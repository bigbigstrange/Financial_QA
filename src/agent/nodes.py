"""Agent 各节点实现。

设计原则：
    1. **无 LLM 也能跑通**：`llm=None` 时路由退化为「总是检索」、评分保留全部证据、
       双层校验直接通过、生成退化为抽取式摘要。这样单元测试与无 Key 环境都能验证图结构。
    2. **单次批量调用完成评分**：参考实现是「逐文档一次 LLM 调用」，10 条证据就是 10 次请求；
       这里改为一次请求批量返回判定数组，成本和延迟都降一个量级。
    3. **两个重试计数分开**：幻觉问题回 `generate`（同一批文档重新生成），
       偏题问题回 `retrieve`（换检索参数），两条路径各自计数、互不干扰。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import prompts
from .state import AgentState, Evidence, NodeTrace, initial_state
from src.system.pilot_budget import PilotBudgetStopped


@dataclass
class AgentConfig:
    """Agent 运行参数（可用环境变量覆盖，便于消融与降本）。"""

    top_k: int = 12
    max_retry_retrieve: int = 2
    max_retry_generate: int = 1
    grade_with_llm: bool = True
    enable_verify: bool = True
    max_grade_candidates: int = 10
    max_generate_evidence: int = 8

    @classmethod
    def from_env(cls) -> "AgentConfig":
        def _int(name: str, default: int) -> int:
            raw = (os.getenv(name) or "").strip()
            return int(raw) if raw.isdigit() else default

        def _bool(name: str, default: bool) -> bool:
            raw = (os.getenv(name) or "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "y"}

        return cls(
            top_k=_int("FINANCE_AGENT_TOP_K", 12),
            max_retry_retrieve=_int("FINANCE_AGENT_MAX_RETRY_RETRIEVE", 2),
            max_retry_generate=_int("FINANCE_AGENT_MAX_RETRY_GENERATE", 1),
            grade_with_llm=_bool("FINANCE_AGENT_GRADE_LLM", True),
            enable_verify=_bool("FINANCE_AGENT_VERIFY", True),
        )


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class AgentNodes:
    """把检索栈与 LLM 包成一组 LangGraph 节点。"""

    def __init__(
        self,
        retriever: Any,
        llm: Any = None,
        *,
        config: Optional[AgentConfig] = None,
        planner: Any = None,
        reranker: Any = None,
        log: Optional[Callable[[str], None]] = None,
        web_search: Any = None,
    ) -> None:
        self.retriever = retriever
        if web_search is None:
            from src.system.web_search import TavilySearch
            web_search = TavilySearch.from_env()
        self.web_search_client = web_search
        self.llm = llm if (llm is not None and getattr(llm, "enabled", bool(llm))) else None
        self.config = config or AgentConfig.from_env()
        self.log = log or (lambda msg: print(msg, flush=True))
        if planner is None:
            from src.system.agent import QueryPlanner

            planner = QueryPlanner()
        self.planner = planner
        if reranker is None:
            from src.system.rerank import EvidenceReranker

            reranker = EvidenceReranker()
        self.reranker = reranker

    def _llm_enabled(self, state: AgentState) -> bool:
        return self.llm is not None and state.get("use_paid_model", True)

    # ---- 工具 -------------------------------------------------------------

    def _chat_json(self, system: str, user: str, *, max_tokens: int = 600,
                   node: str = "chat") -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("LLM 未配置")
        kwargs = {"node": node} if getattr(self.llm, "provider", None) == "ark" else {}
        raw = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
            **kwargs,
        )
        return prompts.extract_json(raw)

    def _chat_text(self, system: str, user: str, *, max_tokens: int = 1200,
                   node: str = "chat") -> str:
        if self.llm is None:
            raise RuntimeError("LLM 未配置")
        kwargs = {"node": node} if getattr(self.llm, "provider", None) == "ark" else {}
        return self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
            **kwargs,
        )

    # ---- 节点：路由 -------------------------------------------------------

    def route_query(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        route, reason = "retrieval", "本次未启用或未配置模型，默认走知识库检索"
        if not self._llm_enabled(state) and state["question"].strip().lower().rstrip("!！。.") in {"你好", "您好", "hi", "hello", "谢谢"}:
            route, reason = "direct", "本地规则：简单问候，无需检索"
        if self._llm_enabled(state):
            try:
                payload = self._chat_json(
                    prompts.ROUTE_SYSTEM,
                    prompts.ROUTE_USER.format(question=state["question"]),
                    max_tokens=200,
                    node="route_query",
                )
                candidate = str(payload.get("route") or "").strip().lower()
                if candidate in {"retrieval", "vectorstore", "websearch", "direct", "clarify"}:
                    route = "retrieval" if candidate == "vectorstore" else candidate
                reason = str(payload.get("reason") or "")[:200]
                if route == "websearch" and (self.web_search_client is None or not state.get("use_web_search", True)):
                    route, reason = "retrieval", "本次未启用网页搜索，回退知识库"
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                reason = f"路由失败，回退 retrieval: {type(exc).__name__}"
        self.log(f"[route] {route} — {reason}")
        return {
            "route": route,
            "route_reason": reason,
            "trace": [NodeTrace(node="route_query", elapsed_ms=int(_now_ms() - started),
                                decision=route, detail={"reason": reason})],
        }

    # ---- 节点：子问题规划 -------------------------------------------------

    def plan_subquestions(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        try:
            subs = list(self.planner.plan(state["question"]) or [])
        except Exception as exc:  # noqa: BLE001
            subs = [state["question"]]
            self.log(f"[plan] 规划失败回退原问题: {exc}")
        if not subs:
            subs = [state["question"]]
        return {
            "sub_questions": subs,
            "trace": [NodeTrace(node="plan_subquestions",
                                elapsed_ms=int(_now_ms() - started),
                                decision=f"{len(subs)} 个子问题",
                                detail={"sub_questions": subs})],
        }

    # ---- 节点：检索 -------------------------------------------------------

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        questions = state.get("sub_questions") or [state["question"]]
        top_k = int(state.get("retrieve_top_k") or self.config.top_k)
        merged: Dict[str, Evidence] = {}
        channels: Dict[str, int] = {}
        for q in questions:
            try:
                if not state.get("use_paid_model", True):
                    from src.system.retrieval import HybridRetriever

                    if isinstance(self.retriever, HybridRetriever):
                        hits = self.retriever.search(q, top_k=top_k, allow_remote_graph=False) or []
                    else:
                        hits = self.retriever.search(q, top_k=top_k) or []
                else:
                    hits = self.retriever.search(q, top_k=top_k) or []
            except Exception as exc:  # noqa: BLE001
                self.log(f"[retrieve] 检索失败: {type(exc).__name__}: {exc}")
                hits = []
            for hit in hits:
                cid = hit.get("chunk_id") or hit.get("block_id")
                if not cid:
                    continue
                prev = merged.get(cid)
                if prev is None:
                    merged[cid] = dict(hit)
                else:
                    prev["fusion_score"] = max(
                        float(prev.get("fusion_score") or 0.0),
                        float(hit.get("fusion_score") or 0.0),
                    )
                for ch in hit.get("retrieval_channels") or []:
                    channels[ch] = channels.get(ch, 0) + 1
                ch = hit.get("retrieval_channel")
                if ch:
                    channels[ch] = channels.get(ch, 0) + 1

        documents = sorted(
            merged.values(),
            key=lambda x: float(x.get("fusion_score") or x.get("score") or 0.0),
            reverse=True,
        )[: max(top_k, self.config.top_k)]
        self.log(f"[retrieve] 召回 {len(documents)} 条（{len(questions)} 个子问题）")
        return {
            "documents": documents,
            "trace": [NodeTrace(node="retrieve", elapsed_ms=int(_now_ms() - started),
                                decision=f"{len(documents)} 条证据",
                                detail={"channels": channels,
                                        "sub_questions": len(questions)})],
        }

    # ---- 节点：网页检索 ---------------------------------------------------

    def web_search(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        documents: List[Evidence] = []
        error = ""
        if self.web_search_client is not None and state.get("use_web_search", True):
            try:
                # Never send local evidence, rewritten queries, or conversation state outward.
                documents = [ev for ev in (self.web_search_client.search(state["question"]) or [])
                             if _supports_named_app(state["question"], ev)]
            except Exception as exc:  # noqa: BLE001
                self.log(f"[web_search] 搜索失败: {type(exc).__name__}")
                code = re.fullmatch(r"Tavily HTTP ([1-5][0-9]{2})", str(exc))
                if isinstance(exc, TimeoutError) or str(exc) == "Tavily search unavailable":
                    error = "网页搜索失败：连接超时或服务不可用，请稍后重试。"
                elif code and code.group(1) in {"401", "403"}:
                    error = f"网页搜索失败（HTTP {code.group(1)}）：请检查服务端 Tavily 密钥。"
                elif code and code.group(1) == "429":
                    error = "网页搜索失败（HTTP 429）：Tavily 额度或速率受限，请稍后重试。"
                else:
                    error = "网页搜索失败：搜索服务暂不可用，请稍后重试。"
        self.log(f"[web_search] 召回 {len(documents)} 条公开网页")
        return {
            "web_searched": True,
            "web_search_error": error,
            "web_documents": documents,
            "local_relevant_documents": list(state.get("relevant_documents") or []),
            "relevant_documents": [],
            "trace": [NodeTrace(node="web_search", elapsed_ms=int(_now_ms() - started),
                                decision="search_failed" if error else f"{len(documents)} 条网页证据")],
        }

    # ---- 节点：证据相关性评分 ---------------------------------------------

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        is_web = bool(state.get("web_searched"))
        documents = list(state.get("web_documents") if is_web else state.get("documents") or [])
        threshold_hits = [ev for ev in documents if _supports_named_app(state["question"], ev)]

        # 先用融合/重排分数做便宜预筛，只把候选送 LLM
        if self.reranker is not None and threshold_hits:
            try:
                if not state.get("use_paid_model", True):
                    reranked = threshold_hits
                else:
                    reranked = self.reranker.rerank(
                        state["question"], threshold_hits,
                        top_k=max(len(threshold_hits), self.config.max_grade_candidates),
                    )
                threshold_hits = [ev for ev in reranked if _supports_named_app(state["question"], ev)]
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"[grade] 预筛失败，使用原始顺序: {exc}")

        candidates = threshold_hits[: self.config.max_grade_candidates]
        grades: Dict[int, bool] = {}
        reasons: Dict[int, str] = {}
        invalid: set[int] = set()
        llm_used = False

        if self._llm_enabled(state) and self.config.grade_with_llm and candidates:
            try:
                payload = self._chat_json(
                    prompts.GRADE_SYSTEM,
                    prompts.build_grade_prompt(state["question"], candidates),
                    max_tokens=max(300, 80 * len(candidates)),
                    node="grade_documents",
                )
                llm_used = True
                for entry in payload.get("grades") or []:
                    if not isinstance(entry, dict):
                        continue
                    if isinstance(entry.get("index"), bool):
                        continue
                    try:
                        idx = int(entry.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if not 1 <= idx <= len(candidates) or not isinstance(entry.get("relevant"), bool):
                        continue
                    if idx in grades or idx in invalid:
                        grades.pop(idx, None)
                        invalid.add(idx)
                        continue
                    grades[idx] = entry["relevant"]
                    reasons[idx] = str(entry.get("reason") or "")[:120]
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"[grade] LLM 评分失败，候选不作为已评分证据: {type(exc).__name__}")

        relevant: List[Evidence] = []
        model_grading = self._llm_enabled(state) and self.config.grade_with_llm
        for idx, ev in enumerate(candidates, start=1):
            # A public web snippet is untrusted: missing/failed grading cannot
            # silently turn every search hit into answer evidence.
            keep = grades.get(idx, False) if is_web or model_grading else True
            item = dict(ev)
            item["relevant"] = keep
            if idx in reasons:
                item["grade_reason"] = reasons[idx]
            if keep:
                relevant.append(item)

        stage_relevant = len(relevant)
        if is_web and relevant:
            local = list(state.get("local_relevant_documents") or [])
            seen = {ev.get("chunk_id") or ev.get("block_id") for ev in local}
            relevant = local + [ev for ev in relevant
                                if (ev.get("chunk_id") or ev.get("block_id")) not in seen]

        filtered = len(documents) - stage_relevant
        graded = len(candidates)
        self.log(f"[grade] 召回 {len(documents)} 条，评分 {graded} 条，"
                 f"本轮保留 {stage_relevant} 条（过滤 {filtered}），合并后 {len(relevant)} 条")
        return {
            # documents 保持完整召回集；graded_count 记录实际送评分的条数。
            # 之前把 documents 改成候选子集，导致「相关数 > 召回数」的反直觉日志。
            "documents": state.get("documents") or [] if is_web else documents,
            "relevant_documents": relevant,
            "graded_count": graded,
            "filtered_count": max(0, filtered),
            "trace": [NodeTrace(node="grade_documents",
                                elapsed_ms=int(_now_ms() - started),
                                decision=f"保留 {stage_relevant}/{len(documents)}",
                                detail={"llm_used": llm_used, "graded": graded})],
        }

    # ---- 节点：查询改写（偏题时换信息源）----------------------------------

    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        question = state["question"]
        new_query = question
        if self._llm_enabled(state):
            try:
                payload = self._chat_json(
                    prompts.REWRITE_SYSTEM,
                    prompts.build_rewrite_prompt(question, state.get("route_reason") or question),
                    max_tokens=200,
                    node="rewrite_query",
                )
                new_query = str(payload.get("query") or "").strip() or question
                if "掌上银行" in question and "掌上银行" not in new_query:
                    new_query = f"掌上银行 {new_query}"
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"[rewrite] 失败，沿用原问题: {type(exc).__name__}")
        retry = int(state.get("retry_retrieve") or 0) + 1
        # 扩召回窗口：偏题时换信息源
        widened = min(int(state.get("retrieve_top_k") or self.config.top_k) + 6, 40)
        self.log(f"[rewrite] 第 {retry} 次重检索: {new_query[:60]}")
        return {
            "sub_questions": [new_query],
            "retry_retrieve": retry,
            "retrieve_top_k": widened,
            "trace": [NodeTrace(node="rewrite_query",
                                elapsed_ms=int(_now_ms() - started),
                                decision=f"retry={retry}",
                                detail={"query": new_query, "top_k": widened})],
        }

    # ---- 节点：生成 -------------------------------------------------------

    def generate(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        evidences = (state.get("relevant_documents") or [])[:self.config.max_generate_evidence]
        generation = ""
        truncated = False
        citations: List[int] = []
        from .visual_answer import pilot_visual_approved

        approved_image = next((ev for ev in evidences
                               if ev.get("image_ref") and pilot_visual_approved(ev)), None)
        if not evidences:
            generation = "知识库中无依据：未检索到与该问题相关的文档内容。"
            status = "no_evidence"
        elif approved_image is not None and self._llm_enabled(state):
            try:
                from .visual_answer import build_image_messages

                root = Path(__file__).resolve().parents[2] / "output" / "kb_v3"
                messages = build_image_messages(state["question"], approved_image, root)
                visual_text = self.llm.chat(messages, temperature=0.0,
                                            max_tokens=500, node="visual_generate").strip()
                truncated = getattr(getattr(self.llm, "last_response", None), "finish_reason", None) == "length"
                if visual_text:
                    citation = evidences.index(approved_image) + 1
                    generation = f"图像模型初步识别（未经双校验）：{visual_text} [{citation}]"
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"[generate] 视觉调用不可用: {type(exc).__name__}")
        elif all(ev.get('evidence_kind') == 'page_image' for ev in evidences):
            if not generation:
                generation = self._extractive_answer(evidences)
        elif self._llm_enabled(state):
            try:
                generation = self._chat_text(
                    prompts.GENERATE_SYSTEM,
                    prompts.build_generate_prompt(state["question"], evidences),
                    max_tokens=1200,
                    node="generate",
                ).strip()
                truncated = getattr(getattr(self.llm, "last_response", None), "finish_reason", None) == "length"
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"[generate] LLM 失败，回退抽取式: {type(exc).__name__}")
        if not generation:
            generation = self._extractive_answer(evidences)
        citations = _extract_citations(generation)
        self.log(f"[generate] {len(generation)} 字，引用 {citations}")
        return {
            "generation": generation,
            "generation_truncated": truncated,
            "citations": citations,
            "trace": [NodeTrace(node="generate", elapsed_ms=int(_now_ms() - started),
                                decision=f"{len(generation)} 字",
                                detail={"evidence": len(evidences),
                                        "citations": citations})],
        }

    @staticmethod
    def _extractive_answer(evidences: List[Evidence]) -> str:
        """抽取式兜底：LLM 不可用时把最相关的证据整理成分点答案。"""
        origin = "公开网页" if all(ev.get("source_url") for ev in evidences) else "知识库"
        lines = [f"{origin}候选原文摘录（未启用生成模型，相关性与答案尚未校验）："]
        for i, ev in enumerate(evidences[:5], start=1):
            text = " ".join((ev.get("content") or "").split())[:220]
            if ev.get('evidence_kind') == 'page_image':
                text = '仅图像证据，未经文字解析；未经图像内容验证／答案验证'
            loc = ev.get("source_doc") or "-"
            page = ev.get('vision_page') or ev.get("page_start", ev.get("page"))
            location = (ev.get("source_url") or
                        (f"第{page}页" if page else (ev.get("section_path") or "无物理页码")))
            lines.append(f"{i}. {text} [{i}]（{loc} {location}）")
        return "\n".join(lines)

    # ---- 节点：双层校验（第一层 · 幻觉）------------------------------------

    def check_hallucination(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        generation = state.get("generation") or ""
        evidences = (state.get("relevant_documents") or [])[:self.config.max_generate_evidence]
        grounded, reason = True, "未启用校验或 LLM 不可用，未经验证"
        verified = False
        if state.get("generation_truncated"):
            grounded, reason = False, "回答达到输出上限而被截断，不能通过校验"
        elif (self._llm_enabled(state) and self.config.enable_verify and evidences
                and any(ev.get('content') for ev in evidences)):
            try:
                payload = self._chat_json(
                    prompts.HALLUCINATION_SYSTEM,
                    prompts.build_hallucination_prompt(state["question"], generation, evidences),
                    max_tokens=300,
                    node="check_hallucination",
                )
                verified = isinstance(payload.get("grounded"), bool)
                grounded = payload["grounded"] if verified else False
                reason = str(payload.get("reason") or "")[:200]
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                grounded = False
                reason = f"校验失败，未经验证: {type(exc).__name__}"
        retry = int(state.get("retry_generate") or 0)
        if not grounded:
            retry += 1
        self.log(f"[check_hallucination] grounded={grounded} verified={verified} — {reason}")
        return {
            "grounded": grounded,
            "hallucination_reason": reason,
            "grounded_verified": verified,
            "verified": verified and grounded and bool(state.get("answer_verified"))
                        and state.get("answers_question") is True,
            "retry_generate": retry,
            "trace": [NodeTrace(node="check_hallucination",
                                elapsed_ms=int(_now_ms() - started),
                                decision=str(grounded) if verified else "未验证",
                                detail={"reason": reason, "verified": verified})],
        }

    # ---- 节点：双层校验（第二层 · 答案相关性）------------------------------

    def check_answer(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        generation = state.get("generation") or ""
        answers, reason = True, "未启用校验或 LLM 不可用，未经验证"
        answer_verified = False
        evidences = state.get('relevant_documents') or []
        if self._llm_enabled(state) and self.config.enable_verify and any(ev.get('content') for ev in evidences):
            try:
                payload = self._chat_json(
                    prompts.ANSWER_SYSTEM,
                    prompts.build_answer_check_prompt(state["question"], generation),
                    max_tokens=300,
                    node="check_answer",
                )
                answer_verified = isinstance(payload.get("answers_question"), bool)
                answers = payload["answers_question"] if answer_verified else False
                reason = str(payload.get("reason") or "")[:200]
            except PilotBudgetStopped:
                raise
            except Exception as exc:  # noqa: BLE001
                answers = False
                reason = f"校验失败，未经验证: {type(exc).__name__}"
        verified = (bool(state.get("grounded_verified")) and state.get("grounded") is True
                    and answer_verified and answers)
        self.log(f"[check_answer] answers_question={answers} verified={verified} — {reason}")
        return {
            "answers_question": answers,
            "answer_reason": reason,
            "answer_verified": answer_verified,
            "verified": verified,
            "trace": [NodeTrace(node="check_answer",
                                elapsed_ms=int(_now_ms() - started),
                                decision=str(answers) if answer_verified else "未验证",
                                detail={"reason": reason, "verified": verified})],
        }

    # ---- 节点：直接回答 / 反问 / 兜底 -------------------------------------

    def direct_answer(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        text = "你好！我可以帮你查询银行业务规则、产品条款与操作流程，请直接提问。"
        if self._llm_enabled(state):
            try:
                text = self._chat_text(
                    prompts.DIRECT_SYSTEM, state["question"], max_tokens=300,
                    node="direct_answer",
                ).strip() or text
            except PilotBudgetStopped:
                raise
            except Exception:  # noqa: BLE001
                pass
        return {
            "generation": text,
            "status": "direct",
            "confidence": 0.5,
            "trace": [NodeTrace(node="direct_answer",
                                elapsed_ms=int(_now_ms() - started), decision="direct")],
        }

    def clarify(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        text = "能否补充一下具体是哪项业务或产品？这样我能更准确地帮你检索知识库。"
        if self._llm_enabled(state):
            try:
                text = self._chat_text(
                    prompts.DIRECT_SYSTEM, prompts.build_clarify_prompt(state["question"]),
                    max_tokens=200,
                    node="clarify",
                ).strip() or text
            except PilotBudgetStopped:
                raise
            except Exception:  # noqa: BLE001
                pass
        return {
            "generation": text,
            "status": "clarify",
            "confidence": 0.4,
            "trace": [NodeTrace(node="clarify",
                                elapsed_ms=int(_now_ms() - started), decision="clarify")],
        }

    # ---- 节点：收尾（写 status / confidence）------------------------------

    def finalize(self, state: AgentState) -> Dict[str, Any]:
        """成功路径的收尾节点。

        `status` 与 `confidence` 必须在**所有出口**都被写入：
        `generate` 成功后直接连 END 就会漏掉这两个字段，调用方拿到 None。
        """
        started = _now_ms()
        status = "completed"
        confidence = derive_confidence({**state, "status": status})
        self.log(f"[finalize] status={status} confidence={confidence}")
        return {
            "status": status,
            "confidence": confidence,
            "trace": [NodeTrace(node="finalize", elapsed_ms=int(_now_ms() - started),
                                decision=(f"证据覆盖参考分={confidence}" if state.get("verified")
                                          else "未经模型校验"))],
        }

    def give_up(self, state: AgentState) -> Dict[str, Any]:
        started = _now_ms()
        reason = state.get("answer_reason") or state.get("hallucination_reason") or ""
        if state.get("web_search_error"):
            text = state["web_search_error"]
            if state.get("route") != "websearch":
                text = "知识库中无足够依据；" + text
        elif state.get("web_documents"):
            text = "公开网页候选未通过相关性核验，不能作为回答依据。"
        elif state.get("route") == "websearch":
            text = "公开网页中无依据：未找到能回答该问题的可信网页摘录。"
        elif state.get("web_searched"):
            text = "知识库中无依据；公开网页也未找到能回答该问题的可信摘录。"
        else:
            text = "知识库中无依据：当前知识库没有能回答该问题的内容。"
        if reason:
            text += f"（{reason}）"
        self.log(f"[give_up] {text[:80]}")
        return {
            "generation": text,
            "status": "no_evidence",
            "confidence": 0.1,
            "verified": False,
            "trace": [NodeTrace(node="give_up",
                                elapsed_ms=int(_now_ms() - started),
                                decision="no_evidence", detail={"reason": reason[:200]})],
        }


def _extract_citations(text: str) -> List[int]:
    out: List[int] = []
    for match in re.finditer(r"\[(\d{1,2})\]", text or ""):
        idx = int(match.group(1))
        if idx not in out:
            out.append(idx)
    return out


_BANK_ALIASES = (
    ("邮政储蓄银行", "邮政储蓄", "邮储"),
    ("建设银行", "建行"),
    ("工商银行", "工行"),
    ("中国银行", "中行"),
    ("农业银行", "农行"),
    ("交通银行", "交行"),
    ("浦东发展银行", "浦发", "浦东发展"),
    ("上海银行",),
    ("赣昌农商银行", "赣昌农商行"),
)


def _supports_named_app(question: str, evidence: Evidence) -> bool:
    """Filter explicit App/bank identities before LLM grading or generation."""
    identity = " ".join(str(evidence.get(key) or "") for key in
                        ("source_doc", "section_path", "content"))
    image_only = (evidence.get("evidence_kind") == "page_image"
                  and bool(evidence.get("image_ref"))
                  and not str(evidence.get("content") or "").strip())
    if "掌上银行" in question and not image_only:
        if "掌上银行" not in identity or ("基金" in question and "基金" not in identity):
            return False
        if any(term in question for term in ("流程", "步骤", "如何", "怎么", "具体操作")):
            actions = ("点击", "选择", "进入", "输入", "提交", "确认", "登录", "勾选", "阅读", "填写")
            if sum(term in identity for term in actions) < 2:
                return False
    targets = [index for index, aliases in enumerate(_BANK_ALIASES)
               if any(alias in question for alias in aliases)]
    if not targets:
        return True
    source = str(evidence.get("source_doc") or "")
    source_banks = [index for index, aliases in enumerate(_BANK_ALIASES)
                    if any(alias in source for alias in aliases)]
    if source_banks and not any(index in targets for index in source_banks):
        return False
    return any(alias in identity for index in targets for alias in _BANK_ALIASES[index])


def derive_confidence(state: AgentState) -> float:
    """由证据数量与双层校验结果推导置信度（可解释，不依赖模型自评）。

    刻意保持粗粒度且可解释：
      - 证据越多越可信，6 条封顶；
      - **未经双层校验时打折** —— LLM 不可用会让两层都「空过」，
        如果把空过当成通过就会给出虚假的满分；
      - 任一校验不通过减半。
    """
    if state.get("status") == "no_evidence":
        return 0.1
    if state.get("route") == "direct":
        return 0.5
    docs = len(state.get("relevant_documents") or [])
    score = min(1.0, docs / 6.0)
    if not state.get("verified"):
        score *= 0.75
    if state.get("grounded") is False:
        score *= 0.5
    if state.get("answers_question") is False:
        score *= 0.5
    return round(score, 2)
