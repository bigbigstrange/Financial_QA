# -*- coding: utf-8 -*-
"""
证据重排：heuristic 默认 + 难例走 DeepSeek API。
已移除本地 CrossEncoder（易 OOM）。
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from .llm_api import DeepSeekAPIClient


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "y"}


def is_hard_case(question: str, sub_questions: Optional[List[str]] = None) -> bool:
    """难例：多意图 / 长问 / 复杂操作与排查。"""
    q = (question or "").strip()
    subs = [s for s in (sub_questions or []) if s and s.strip()]
    if len(subs) >= 2:
        return True
    if len(q) >= 55:
        return True
    if re.search(r"以及|并且|分别|同时|并说明|另外|然后|既要|又要", q):
        return True
    if "基金" in q and re.search(r"内容", q) and re.search(r"流程|认购|购买", q):
        return True
    if re.search(r"我的账户|总资产", q) and re.search(r"以及|分别|明细", q):
        return True
    if re.search(r"USBkey|USB.?[Kk]ey|U盾|证书管理工具", q, re.I) and re.search(
        r"未自动弹出|不弹出|如何处理|注意", q
    ):
        return True
    if re.search(r"原因|如何处理|排查|异常|故障|提示", q) and len(q) >= 28:
        return True
    return False


def _org_keys(question: str) -> List[str]:
    q = question or ""
    mapping = [
        (r"邮政储蓄|邮储", ["邮政储蓄", "邮储", "中国邮政"]),
        (r"浦东发展|浦发", ["浦东发展", "浦发", "上海浦东"]),
        (r"交通银行|交行", ["交通银行", "交行"]),
        (r"建设银行|建行", ["建设银行", "建行"]),
        (r"工商银行|工行", ["工商银行", "工行"]),
        (r"农业银行|农行", ["农业银行", "农行"]),
        (r"中国银行(?!业)", ["中国银行"]),
    ]
    keys: List[str] = []
    for pat, vals in mapping:
        if re.search(pat, q):
            keys.extend(vals)
    return keys


def _token_set(text: str) -> set[str]:
    # 简易中文二元组 + 连续数字/英文词，避免强依赖 jieba
    t = re.sub(r"\s+", "", text or "")
    grams = set()
    for i in range(max(0, len(t) - 1)):
        g = t[i : i + 2]
        if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", g):
            grams.add(g)
    for m in re.findall(r"[A-Za-z0-9\-]{3,}|[\u4e00-\u9fff]{2,}", text or ""):
        grams.add(m)
    return grams


class EvidenceReranker:
    """
    FINANCE_RERANK_BACKEND:
      - hybrid（默认）：简单题 heuristic，难例 api（无 Key 则全程 heuristic）
      - heuristic：仅关键词/机构加权
      - api：全程 API
      - none：仅原 fusion 分数
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        _ = model_name  # 兼容旧调用签名，已不再加载本地重排模型
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FINANCE_API_KEY") or ""
        ).strip()
        backend = (os.environ.get("FINANCE_RERANK_BACKEND") or "hybrid").strip().lower()
        if backend in {"auto", ""}:
            backend = "hybrid"
        self.backend = backend
        self.api_client: Optional[DeepSeekAPIClient] = None
        if backend in {"hybrid", "api"} and api_key:
            self.api_client = DeepSeekAPIClient(api_key=api_key)
            if not self.api_client.enabled:
                self.api_client = None
        print(
            f"[rerank] backend={self.backend} api={'on' if self.api_client else 'off'}",
            flush=True,
        )

    def rerank(
        self,
        question: str,
        evidences: List[Dict],
        top_k: int,
        sub_questions: Optional[List[str]] = None,
    ) -> List[Dict]:
        if not evidences:
            return []
        top_k = max(1, int(top_k))

        if self.backend == "none":
            return self._by_fusion(evidences, top_k)

        if self.backend == "api":
            if self.api_client is not None:
                return self._rerank_api(question, evidences, top_k)
            return self._rerank_heuristic(question, evidences, top_k)

        if self.backend == "heuristic":
            return self._rerank_heuristic(question, evidences, top_k)

        # hybrid
        hard = is_hard_case(question, sub_questions)
        if hard and self.api_client is not None:
            print("[rerank] 难例 → API 轻量重排", flush=True)
            return self._rerank_api(question, evidences, top_k)
        if hard and self.api_client is None:
            print("[rerank] 难例但无 API Key → heuristic", flush=True)
        else:
            print("[rerank] 简单题 → heuristic", flush=True)
        return self._rerank_heuristic(question, evidences, top_k)

    @staticmethod
    def _by_fusion(evidences: List[Dict], top_k: int) -> List[Dict]:
        ranked = sorted(
            evidences,
            key=lambda x: float(x.get("fusion_score", x.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        return ranked[:top_k]

    def _rerank_heuristic(self, question: str, evidences: List[Dict], top_k: int) -> List[Dict]:
        q = question or ""
        q_tokens = _token_set(q)
        org_keys = _org_keys(q)
        want_phone = bool(re.search(r"电话|客服|热线", q))
        want_fund = "基金" in q
        want_debit = "借记卡" in q
        want_credit = "信用卡" in q

        scored: List[Dict] = []
        for ev in evidences:
            item = dict(ev)
            content = item.get("content") or ""
            doc = item.get("source_doc") or ""
            blob = f"{doc} {content}"
            base = float(item.get("fusion_score", item.get("score", 0.0)) or 0.0)
            bonus = 0.0

            c_tokens = _token_set(blob)
            if q_tokens and c_tokens:
                overlap = len(q_tokens & c_tokens) / max(8.0, float(len(q_tokens)))
                bonus += min(0.35, overlap)

            if org_keys:
                if any(k in blob for k in org_keys):
                    bonus += 0.25
                else:
                    bonus -= 0.35

            if want_phone:
                if re.search(r"借记卡客服电话\s*[\d\-]+", content) and re.search(
                    r"信用卡客服电话\s*[\d\-]+", content
                ):
                    bonus += 0.45
                elif re.search(r"(借记卡|信用卡)?客服电话\s*[\d\-]+", content):
                    bonus += 0.28
                if want_debit and "借记卡" in content:
                    bonus += 0.08
                if want_credit and "信用卡" in content:
                    bonus += 0.08

            if want_fund:
                for kw, w in (
                    ("基金超市", 0.12),
                    ("购买金额", 0.12),
                    ("认购", 0.08),
                    ("操作流程", 0.08),
                    ("投资者问卷", 0.06),
                ):
                    if kw in content:
                        bonus += w

            if item.get("type") == "table":
                bonus += 0.05

            item["rerank_score"] = base + bonus
            item["fusion_score"] = item["rerank_score"]
            ch = str(item.get("retrieval_channel") or "text")
            if "+heuristic" not in ch:
                item["retrieval_channel"] = ch + "+heuristic"
            scored.append(item)

        scored.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        return scored[:top_k]

    def _rerank_api(self, question: str, evidences: List[Dict], top_k: int) -> List[Dict]:
        assert self.api_client is not None
        # 先用 heuristic 收窄候选，再 API，控成本
        narrowed = self._rerank_heuristic(question, evidences, top_k=max(top_k, 16))
        cand_n = min(
            len(narrowed),
            int(os.environ.get("FINANCE_API_RERANK_CANDIDATES", "12") or "12"),
        )
        pool = narrowed[:cand_n]
        snippets = []
        for ev in pool:
            src = ev.get("source_doc") or ""
            page = ev.get("page")
            head = f"{src}" + (f" p.{page}" if page is not None else "")
            body = (ev.get("content") or "")[:200]
            snippets.append(f"{head} | {body}".strip(" |"))
        try:
            order = self.api_client.rerank_indices(question, snippets, top_k=top_k)
            ranked: List[Dict] = []
            for rank_i, idx in enumerate(order):
                if idx < 0 or idx >= len(pool):
                    continue
                item = dict(pool[idx])
                item["rerank_score"] = float(len(order) - rank_i) / float(max(1, len(order)))
                item["fusion_score"] = max(
                    float(item.get("fusion_score", item.get("score", 0.0)) or 0.0),
                    float(item["rerank_score"]),
                )
                ch = str(item.get("retrieval_channel") or "text").replace("+heuristic", "")
                item["retrieval_channel"] = ch + "+api_rerank"
                ranked.append(item)
            if not ranked:
                return pool[:top_k]
            print(
                f"[rerank] API 完成：candidates={len(pool)} -> top={min(top_k, len(ranked))}",
                flush=True,
            )
            return ranked[:top_k]
        except Exception as e:
            print(f"[rerank] API 失败，回退 heuristic: {e}", flush=True)
            return pool[:top_k]
