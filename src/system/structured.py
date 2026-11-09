"""Structured metadata retrieval channel for institution/date/type filters."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .sparse import tokenize_text


_ORG_KEYS = {
    "邮储": "邮政储蓄",
    "邮政储蓄": "邮政储蓄",
    "浦发": "浦东发展",
    "浦东发展": "浦东发展",
    "交行": "交通银行",
    "交通银行": "交通银行",
    "建行": "建设银行",
    "建设银行": "建设银行",
    "工行": "工商银行",
    "工商银行": "工商银行",
    "农行": "农业银行",
    "农业银行": "农业银行",
    "中国银行": "中国银行",
}

_DATE_PATTERN = re.compile(r"\d{4}[-/.年]\d{1,2}[-/.月]?\d{0,2}日?|\d{4}年")


class StructuredRetriever:
    """按来源、章节、机构、日期和文档类型做精确/结构化召回。"""

    def __init__(self, blocks: List[Dict[str, Any]]) -> None:
        self.blocks: List[Dict[str, Any]] = list(blocks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        terms = self._query_terms(query)
        if not terms:
            return []
        out: List[Dict[str, Any]] = []
        for block in self.blocks:
            score, matched = self._match(block, terms)
            if score <= 0:
                continue
            if visibility is not None and not visibility.allowed(
                block.get("document_version") or ""
            ):
                continue
            item = dict(block)
            item["structured_score"] = score
            item["retrieval_channel"] = "structured"
            if matched:
                item["matched_filters"] = matched
            out.append(item)
        out.sort(key=lambda item: item["structured_score"], reverse=True)
        return out[: max(0, int(top_k))]

    def _query_terms(self, query: str) -> Dict[str, str]:
        terms: Dict[str, str] = {}
        for alias, canonical in _ORG_KEYS.items():
            if alias in query:
                terms[canonical] = "org"
        dates = _DATE_PATTERN.findall(query)
        for date in dates:
            terms[date] = "date"
        for token in tokenize_text(query):
            if token not in terms:
                terms[token] = "token"
        return terms

    def _match(
        self,
        block: Dict[str, Any],
        terms: Dict[str, str],
    ) -> tuple[float, List[str]]:
        field_text = " ".join(
            [
                str(block.get("source_doc") or ""),
                str(block.get("document_id") or ""),
                str(block.get("section_path") or ""),
                str(block.get("document_version") or ""),
                str(block.get("type") or "paragraph"),
            ]
        )
        score = 0.0
        matched: List[str] = []
        for term, kind in terms.items():
            if kind == "org" and re.search(term, field_text, re.I):
                score += 1.0
                matched.append(term)
            elif kind == "date" and term in field_text:
                score += 1.0
                matched.append(term)
            elif kind == "token" and term in field_text:
                score += 0.5
                matched.append(term)
        return round(score, 4), matched
