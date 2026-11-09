# -*- coding: utf-8 -*-
"""Financial_QA 的 MCP Server（stdio）。

把「检索 / 问答 / 证据」三件事暴露成标准 MCP 工具，任何支持 MCP 的客户端
（Claude Desktop、Cursor、IDE 插件等）都能直接调用，无需改一行客户端代码。

设计取舍：
- 默认走离线演示语料（``demo/corpus``），**不需要 API Key 就能连上就有结果**；
  接真实知识库时可替换 ``_engine`` 的实现，工具契约不变。
- 三个工具职责单一：``search_corpus`` 只返回命中，``ask`` 返回带引用的答案，
  ``get_evidence`` 按 chunk_id 取原文——便于模型"先搜、再问、后核对"。

启动：
    python -m mcp_server.server
"""
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.demo_engine import MIN_EVIDENCE_SCORE, answer_demo, load_corpus, search

mcp = FastMCP(
    "financial-qa",
    instructions=(
        "金融文档问答服务。检索示例语料并返回带来源的答案。"
        "先调用 search_corpus 定位相关条款；需要完整答案时调用 ask；"
        "需要核对原文时用 get_evidence 按 chunk_id 取回。"
        "证据不足时服务会明确拒答，不要自行编造答案。"
    ),
)


def _chunks() -> dict[str, Any]:
    chunks, _ = load_corpus()
    return {c.chunk_id: c for c in chunks}


@mcp.tool()
def search_corpus(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """检索示例语料，返回最相关的文档切块（含来源与分数）。

    Args:
        query: 用户问题或关键词。
        top_k: 返回条数，默认 3，最大 10。
    """
    top_k = max(1, min(int(top_k), 10))
    hits = search(query, top_k=top_k)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document": chunk.doc,
            "section": chunk.section,
            "score": score,
            "snippet": chunk.text[:160],
        }
        for chunk, score in hits
    ]


@mcp.tool()
def ask(question: str) -> dict[str, Any]:
    """回答一个金融业务问题，返回答案、置信度与引用来源。

    证据不足时会明确拒答（verified=false），不会编造。

    Args:
        question: 用户的完整问题。
    """
    result = answer_demo(question, question_id="mcp")
    citations = [
        {
            "chunk_id": item["chunk_id"],
            "document": item["source_doc"],
            "section": item["section"],
            "score": item["score"],
        }
        for item in result["evidence"]
    ]
    return {
        "answer": result["answer"],
        "verified": result["verified"],
        "confidence": result["confidence"],
        "refused": not result["verified"],
        "threshold": MIN_EVIDENCE_SCORE,
        "citations": citations,
        "latency_ms": round(result["latency_sec"] * 1000, 2),
    }


@mcp.tool()
def get_evidence(chunk_id: str) -> dict[str, Any]:
    """按 chunk_id 取回切块原文与出处，用于核对答案是否被原文支持。

    Args:
        chunk_id: search_corpus 或 ask 返回的 chunk_id，例如 "01_fund#0"。
    """
    chunk = _chunks().get(chunk_id)
    if chunk is None:
        return {
            "found": False,
            "error": f"chunk_id 不存在：{chunk_id}",
            "available": sorted(_chunks())[:10],
        }
    return {
        "found": True,
        "chunk_id": chunk.chunk_id,
        "document": chunk.doc,
        "section": chunk.section,
        "text": chunk.text,
    }


@mcp.tool()
def list_documents() -> list[dict[str, Any]]:
    """列出当前语料的文档与切块数量，用于判断知识库覆盖范围。"""
    chunks, _ = load_corpus()
    stats: dict[str, int] = {}
    for chunk in chunks:
        stats[chunk.doc] = stats.get(chunk.doc, 0) + 1
    return [
        {"document": doc, "chunks": count, "sections": sorted({
            c.section for c in chunks if c.doc == doc
        })}
        for doc, count in sorted(stats.items())
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

