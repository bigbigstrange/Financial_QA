# -*- coding: utf-8 -*-
"""OpenAI-compatible Chat Completions client (DeepSeek or Ark)."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.contracts import ContractError, ErrorCode, ModelSnapshot
from src.models.gateway import ModelGateway, OpenAICompatibleTransport
from .cost_policy import paid_api_disabled


def configured_api_key() -> str:
    if (os.environ.get("FINANCE_LLM_PROVIDER") or "").strip().lower() == "ark":
        direct = (os.environ.get("ARK_API_KEY") or "").strip()
        if direct:
            return direct
        key_file = (os.environ.get("ARK_API_KEY_FILE") or "").strip()
        if key_file:
            lines = Path(key_file).read_text(encoding="utf-8-sig").splitlines()
            if len(lines) != 1 or not lines[0].strip():
                raise ValueError("Ark API Key 文件必须只含非空单行")
            return lines[0].strip()
        return ""
    return (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FINANCE_API_KEY") or "").strip()


class DeepSeekAPIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider = (
            "ark" if (os.environ.get("FINANCE_LLM_PROVIDER") or "").strip().lower() == "ark"
            else "deepseek"
        )
        if self.provider == "ark":
            self.api_key = configured_api_key()
            self.base_url = (base_url or os.environ.get("ARK_BASE_URL") or
                             "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
            self.model = (model or os.environ.get("ARK_MODEL") or
                          "doubao-seed-2-1-pro-260915").strip()
            if self.model != "doubao-seed-2-1-pro-260915":
                raise ValueError("只允许已批准的 Ark 试跑模型")
            if self.base_url != "https://ark.cn-beijing.volces.com/api/v3":
                raise ValueError("Ark 试跑只允许官方 API 地址")
        else:
            self.api_key = (api_key or configured_api_key()).strip()
            self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or
                             "https://api.deepseek.com").rstrip("/")
            self.model = (model or os.environ.get("DEEPSEEK_MODEL") or
                          os.environ.get("FINANCE_GEN_MODEL_API") or "deepseek-v4-flash").strip()
        self.timeout = timeout
        self.last_response = None
        # V4 Flash 默认可能开 thinking；金融问答优先关闭以省时省钱
        thinking_flag = "ARK_DISABLE_THINKING" if self.provider == "ark" else "DEEPSEEK_DISABLE_THINKING"
        self.disable_thinking = (os.environ.get(thinking_flag) or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        snapshot = ModelSnapshot(
            snapshot_id=f"runtime:{self.provider}",
            profile_name="runtime",
            provider=self.provider,
            requested_model=self.model,
            actual_model=self.model,
        )
        transport = OpenAICompatibleTransport(
            api_key=self.api_key,
            base_url=self.base_url,
            provider=self.provider,
            timeout=self.timeout,
        )
        self.gateway = ModelGateway(snapshot, transport, max_retries=0 if self.provider == "ark" else 1)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not paid_api_disabled()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        node: str = "chat",
    ) -> str:
        from .execution_control import check_execution
        check_execution()
        if paid_api_disabled():
            raise RuntimeError("付费模型调用已暂停：FINANCE_DISABLE_PAID_API=1")
        if not self.api_key:
            name = "ARK_API_KEY" if self.provider == "ark" else "DEEPSEEK_API_KEY / FINANCE_API_KEY"
            raise RuntimeError(f"缺少 {name}")

        extra_body: Dict[str, Any] = {}
        if self.disable_thinking:
            extra_body["thinking"] = {"type": "disabled"}
        budget = None
        request_id = ""
        if self.provider == "ark":
            from .pilot_budget import PilotBudget

            image_count = sum(
                part.get("type") == "image_url"
                for message in messages if isinstance(message.get("content"), list)
                for part in message["content"] if isinstance(part, dict)
            )
            if image_count and (image_count != 1 or max_tokens > 500):
                raise ValueError("视觉试跑仅允许一张图且输出不超过 500 tokens")

            default_path = Path(__file__).resolve().parents[2] / "output" / "pilot_budget.sqlite3"
            budget = PilotBudget(
                Path(os.environ.get("FINANCE_PILOT_BUDGET_PATH") or default_path),
                limit=int(os.environ.get("FINANCE_PILOT_TOKEN_LIMIT") or "1500000"),
                model=self.model,
            )
            request_id = uuid.uuid4().hex
            reservation = (50_000 if image_count else
                           2 * len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + max_tokens)
            budget.reserve(request_id, reservation, node=node)
        started = time.perf_counter()
        try:
            response = self.gateway.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                request_id=request_id or None,
            )
        except Exception as exc:
            if budget is not None:
                from .pilot_budget import PilotBudgetStopped

                reason = "http_402" if "HTTP 402" in str(exc) else (
                    "http_429" if "HTTP 429" in str(exc) else "request_uncertain"
                )
                budget.stop(request_id, reason, latency_ms=(time.perf_counter() - started) * 1000)
                raise PilotBudgetStopped("模型调用失败，试跑已暂停") from exc
            raise
        if budget is not None:
            budget.settle(request_id, response, node)
        self.last_response = response
        return response.text

    def rerank_indices(
        self,
        question: str,
        snippets: Sequence[str],
        *,
        top_k: int,
    ) -> List[int]:
        """
        轻量 API 重排：输入候选摘要，返回按相关性降序的 0-based 下标列表。
        失败时抛异常，由调用方回退原分数排序。
        """
        n = len(snippets)
        if n == 0:
            return []
        top_k = max(1, min(int(top_k), n))
        lines = []
        for i, s in enumerate(snippets):
            t = re.sub(r"\s+", " ", (s or "").strip())[:220]
            lines.append(f"{i}. {t}")
        catalog = "\n".join(lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索重排器。根据问题相关性对候选证据排序。"
                    "只输出 JSON 数组，元素为候选编号（整数），按相关性从高到低。"
                    "不要解释，不要 markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"候选证据（编号从 0 开始）：\n{catalog}\n\n"
                    f"请选出最相关的最多 {top_k} 条，输出例如：[3,0,5]"
                ),
            },
        ]
        raw = self.chat(messages, temperature=0.0, max_tokens=256, node="rerank")
        return _parse_index_list(raw, n=n, top_k=top_k)


def _parse_index_list(raw: str, *, n: int, top_k: int) -> List[int]:
    text = (raw or "").strip()
    # 抽取第一个 JSON 数组
    m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        raise ValueError(f"重排响应无 JSON 数组: {text[:200]}")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("重排响应不是数组")
    out: List[int] = []
    seen = set()
    for x in data:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= top_k:
            break
    if not out:
        raise ValueError(f"重排数组无有效下标: {text[:200]}")
    # 补齐未入选的，保持稳定顺序
    for i in range(n):
        if i not in seen:
            out.append(i)
        if len(out) >= n:
            break
    return out[:n]
