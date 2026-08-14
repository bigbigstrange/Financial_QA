# -*- coding: utf-8 -*-
"""DeepSeek OpenAI-compatible Chat Completions client (cloud gen / lite rerank)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx


class DeepSeekAPIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("FINANCE_API_KEY")
            or ""
        ).strip()
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("DEEPSEEK_MODEL")
            or os.environ.get("FINANCE_GEN_MODEL_API")
            or "deepseek-v4-flash"
        ).strip()
        self.timeout = timeout
        # V4 Flash 默认可能开 thinking；金融问答优先关闭以省时省钱
        self.disable_thinking = (os.environ.get("DEEPSEEK_DISABLE_THINKING") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY / FINANCE_API_KEY")

        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"DeepSeek 响应格式异常: {data}") from e

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
        raw = self.chat(messages, temperature=0.0, max_tokens=256)
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
