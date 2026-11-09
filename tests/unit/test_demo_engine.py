# -*- coding: utf-8 -*-
"""离线演示引擎的行为测试（不依赖大模型与向量库）。"""
from __future__ import annotations

from app.demo_engine import (
    MIN_EVIDENCE_SCORE,
    REFUSAL_ANSWER,
    answer_demo,
    load_corpus,
    search,
)


def test_corpus_is_loaded_with_stable_ids():
    chunks, idf = load_corpus()
    assert len(chunks) >= 12
    assert any(c.chunk_id == "01_fund#0" for c in chunks)
    assert idf, "idf 表不应为空"


def test_factual_question_returns_citation_and_is_verified():
    result = answer_demo("稳盈货币基金A的起购金额是多少？", "t-1")

    assert result["verified"] is True
    assert result["evidence"], "应返回证据"
    top = result["evidence"][0]
    assert top["chunk_id"] == "01_fund#0"
    assert top["score"] >= MIN_EVIDENCE_SCORE
    assert "1 元起购" in result["answer"]
    assert result["agent_mode"] == "demo"


def test_out_of_corpus_question_is_refused_not_fabricated():
    result = answer_demo("信用卡年费是多少？", "t-2")

    assert result["verified"] is False
    assert result["answer"] == REFUSAL_ANSWER
    assert result["confidence"] < 0.5


def test_retrieval_scores_are_sorted_and_bounded():
    hits = search("定期开放产品多久开放一次？")
    assert hits, "应至少命中一个切块"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
