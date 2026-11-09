# -*- coding: utf-8 -*-
"""MCP server 的工具契约测试（不启动子进程，直接调用注册的工具函数）。"""
from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="未安装 mcp SDK 时跳过 MCP 相关测试")

from mcp_server.server import (  # noqa: E402
    ask,
    get_evidence,
    list_documents,
    search_corpus,
)


def test_search_returns_ranked_hits_with_source():
    hits = search_corpus("货币基金起购金额", top_k=3)

    assert hits, "应返回命中"
    assert hits[0]["chunk_id"] == "01_fund#0"
    assert hits[0]["score"] >= hits[-1]["score"], "命中应按分数降序"
    assert hits[0]["document"].endswith(".md")


def test_top_k_is_clamped():
    assert len(search_corpus("基金", top_k=0)) == 1
    assert len(search_corpus("基金", top_k=99)) <= 10


def test_ask_returns_answer_with_citations():
    payload = ask("定期开放产品多久开放一次？")

    assert payload["verified"] is True
    assert payload["refused"] is False
    assert payload["citations"], "回答必须带引用"
    assert payload["citations"][0]["chunk_id"] == "03_wealth#1"
    assert "90 天" in payload["answer"] or "90天" in payload["answer"]


def test_ask_refuses_out_of_corpus_question():
    payload = ask("信用卡年费是多少？")

    assert payload["refused"] is True
    assert payload["verified"] is False
    assert "拒答" in payload["answer"]
    assert payload["citations"] == []


def test_get_evidence_roundtrip():
    payload = ask("稳盈货币基金A的起购金额是多少？")
    chunk_id = payload["citations"][0]["chunk_id"]

    evidence = get_evidence(chunk_id)
    assert evidence["found"] is True
    assert evidence["chunk_id"] == chunk_id
    assert "1 元起购" in evidence["text"]


def test_get_evidence_reports_unknown_id():
    evidence = get_evidence("no-such-chunk")

    assert evidence["found"] is False
    assert "不存在" in evidence["error"]


def test_list_documents_covers_corpus():
    docs = list_documents()

    assert len(docs) >= 4
    assert all(d["chunks"] > 0 for d in docs)
    assert {d["document"] for d in docs} >= {"01_fund.md", "04_service.md"}
