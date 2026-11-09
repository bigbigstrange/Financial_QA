# -*- coding: utf-8 -*-
"""离线演示引擎（Demo Mode）。

设计目标：**不依赖任何大模型、向量库或密钥**，也能完整跑通
「检索 → 评分 → 生成/拒答 → 引用」这条链路，用于面试现场演示与 CI 回归。

实现要点：
1. 语料来自 ``demo/corpus/*.md``（示例数据，非真实业务文档）；
2. 检索用 idf 加权词面重叠（中文按字 bigram），可解释、无外部依赖；
3. 低于阈值时明确拒答，而不是硬编一个答案；
4. 输出结构与 ``QARuntime.answer`` 对齐，前端无需改动。
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CORPUS_DIR = Path(__file__).resolve().parents[1] / "demo" / "corpus"

TOP_K = 3
# 证据充分性阈值：低于该值判为“证据不足”，走明确拒答
MIN_EVIDENCE_SCORE = 0.35
REFUSAL_ANSWER = "根据现有示例资料无法回答该问题（证据不足，已拒答）。"

_SENT_SPLIT = re.compile(r"(?<=[。！？])")
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str


_STOPWORDS = {
    "什么", "怎么", "怎样", "多少", "哪些", "可以", "需要", "是否", "有没有",
    "请问", "一下", "这个", "那个", "我们", "你们", "他们", "还是", "以及",
}


def _terms(text: str) -> set[str]:
    """优先用 jieba 分词（过滤停用词），缺失时退回中文 bigram。"""
    text = text.lower()
    tokens: set[str] = set()
    for word in re.findall(r"[a-z0-9_%]+", text):
        if len(word) > 1:
            tokens.add(word)
    try:
        import jieba

        for word in jieba.cut(text):
            word = word.strip()
            if len(word) >= 2 and word not in _STOPWORDS and not word.isdigit():
                tokens.add(word)
    except Exception:  # pragma: no cover - jieba 缺失时的兜底
        chars = _CJK.findall(text)
        for i in range(len(chars) - 1):
            tokens.add(chars[i] + chars[i + 1])
    return tokens


@lru_cache(maxsize=1)
def load_corpus() -> tuple[list[Chunk], dict[str, float]]:
    """解析示例语料，返回切块与 idf 表。"""
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        section = path.stem
        local_index = 0
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith(">"):
                continue
            if block.startswith("#"):
                section = block.lstrip("# ").strip()
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}#{local_index}",
                    doc=path.name,
                    section=section,
                    text=block,
                )
            )
            local_index += 1

    total = len(chunks) or 1
    df: dict[str, int] = {}
    for chunk in chunks:
        for term in _terms(chunk.text):
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    return chunks, idf


def search(question: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
    """返回按证据强度排序的切块及其 0~1 分数。"""
    chunks, idf = load_corpus()
    q_terms = _terms(question)
    if not q_terms:
        return []
    # 语料中从未出现的查询词按最大 idf 计权：它们恰恰是最"稀有"的词，
    # 若被当成 0 权重，会让"只命中一个常见词"的查询拿到虚高分。
    max_idf = math.log(1 + (len(chunks) + 0.5) / 0.5)
    q_weight = sum(idf.get(t, max_idf) for t in q_terms) or 1.0
    scored: list[tuple[Chunk, float]] = []
    for chunk in chunks:
        c_terms = _terms(chunk.text + " " + chunk.section)
        hit = q_terms & c_terms
        if not hit:
            continue
        score = sum(idf.get(t, max_idf) for t in hit) / q_weight
        scored.append((chunk, round(min(score, 1.0), 4)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _extractive_answer(chunk: Chunk) -> str:
    sentences = [s for s in _SENT_SPLIT.split(chunk.text) if s.strip()]
    head = "".join(sentences[:2]).strip()
    return f"{head}\n\n（演示模式：答案为示例语料的抽取式拼接，未经大模型生成。）"


def answer_demo(question: str, question_id: str) -> dict[str, Any]:
    """生成与 QARuntime.answer 对齐的结果字典。"""
    started = time.time()
    hits = search(question)
    trace: list[dict[str, Any]] = [{
        "node": "demo_route",
        "elapsed_ms": 0,
        "decision": "vectorstore",
        "detail": {"mode": "demo", "reason": "示例语料仅本地检索通道"},
    }]

    best_chunk, best_score = (hits[0] if hits else (None, 0.0))
    # 字段同时满足两套消费方：
    # - 前端 / MCP 工具：chunk_id、source_doc、section、score、content
    # - 证据持久化（legacy 适配器）：block_id、text、structure_path
    evidence = [
        {
            "chunk_id": c.chunk_id,
            "block_id": c.chunk_id,
            "source_doc": c.doc,
            "section": c.section,
            "structure_path": c.section,
            "text": c.text,
            "content": c.text[:400],
            "score": s,
        }
        for c, s in hits
    ]

    sufficient = best_score >= MIN_EVIDENCE_SCORE
    trace.append({
        "node": "demo_grade",
        "elapsed_ms": int((time.time() - started) * 1000),
        "decision": "generate" if sufficient else "refuse",
        "detail": {"best_score": best_score, "threshold": MIN_EVIDENCE_SCORE},
    })

    if sufficient and best_chunk is not None:
        text = _extractive_answer(best_chunk)
        confidence = round(min(0.55 + best_score * 0.4, 0.95), 3)
        verified = True
    else:
        text = REFUSAL_ANSWER
        confidence = round(best_score * 0.5, 3)
        verified = False

    trace.append({
        "node": "demo_generate",
        "elapsed_ms": int((time.time() - started) * 1000),
        "decision": "answered" if sufficient else "refused",
        "detail": {"evidence_count": len(evidence)},
    })

    return {
        "question_id": question_id,
        "question": question,
        "answer": text,
        "thinking": "",
        "sub_questions": [],
        "knowledge_points": evidence,
        "evidence": evidence,
        "confidence": confidence,
        "agent_mode": "demo",
        "route": "vectorstore",
        "retry_retrieve": 0,
        "retry_generate": 0,
        "trace": trace,
        "verified": verified,
        "latency_sec": round(time.time() - started, 3),
        "_local_only": True,
    }





