"""Deterministic BM25 sparse retrieval channel over structured blocks."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def tokenize_text(text: str) -> List[str]:
    text = text or ""
    tokens: List[str] = []
    for match in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff~]+", text):
        if len(match) >= 2:
            tokens.append(match)
        # 长中文片段按相邻二元组扩充，保持可复现性
        if len(match) > 2 and re.search(r"[\u4e00-\u9fff]", match):
            tokens.extend(match[i : i + 2] for i in range(len(match) - 1))
    return tokens


class SparseRetriever:
    """轻量 BM25 实现，索引来自 KnowledgeStore 的 block 列表。"""

    def __init__(
        self,
        blocks: List[Dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.metadata: List[Dict[str, Any]] = list(blocks)
        self.documents: List[List[str]] = []
        avg_len = 0.0
        lengths: List[int] = []
        for block in blocks:
            tokens = tokenize_text(str(block.get("content") or ""))
            self.documents.append(tokens)
            lengths.append(len(tokens))
        self.lengths = lengths
        if lengths:
            avg_len = sum(lengths) / float(len(lengths))
        self.avg_len = avg_len
        self._build_postings()

    def _build_postings(self) -> None:
        df: Dict[str, int] = {}
        postings: Dict[str, List[tuple[int, int]]] = {}
        for idx, tokens in enumerate(self.documents):
            for token, count in Counter(tokens).items():
                df[token] = df.get(token, 0) + 1
                postings.setdefault(token, []).append((idx, count))
        self.df = df
        self.postings = postings
        self.n_docs = len(self.documents)

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        tokens = self.documents[doc_idx]
        if not tokens:
            return 0.0
        idf_default = math.log(1.0 + (self.n_docs - 0.5) / (0.5 + 0.5))
        score = 0.0
        for token in set(query_tokens):
            if not token:
                continue
            term_freq = tokens.count(token)
            if term_freq == 0:
                continue
            df = self.df.get(token, 0)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            if idf <= 0:
                idf = idf_default
            denom = term_freq + self.k1 * (
                1.0 - self.b + self.b * self.lengths[doc_idx] / max(1.0, self.avg_len)
            )
            score += idf * (term_freq * (self.k1 + 1.0)) / denom
        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        visibility: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []
        candidate_k = min(len(self.metadata), max(top_k * 4, top_k))
        scores: Dict[int, float] = {}
        for token in set(query_tokens):
            df = self.df.get(token, 0)
            if not df:
                continue
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for idx, term_freq in self.postings[token]:
                denom = term_freq + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[idx] / max(1.0, self.avg_len)
                )
                scores[idx] = scores.get(idx, 0.0) + idf * (
                    term_freq * (self.k1 + 1.0)
                ) / denom
        scored: List[Dict[str, Any]] = []
        for idx, score in sorted(scores.items(), key=lambda pair: (-round(pair[1], 6), pair[0])):
            if score <= 0:
                continue
            item = dict(self.metadata[idx])
            item["sparse_score"] = round(float(score), 6)
            item["retrieval_channel"] = "sparse"
            scored.append(item)
        if visibility is not None:
            scored = [
                item
                for item in scored[:candidate_k]
                if visibility.allowed(item.get("document_version") or "")
            ]
        else:
            scored = scored[:candidate_k]
        return scored[:top_k]
