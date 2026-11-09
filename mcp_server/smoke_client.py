# -*- coding: utf-8 -*-
"""MCP 冒烟客户端：真的起一个子进程、走 stdio 协议，验证 tools/list 与 tools/call。

用途：本地自测、面试现场演示"这不是文档，是能跑通的协议"。

    python -m mcp_server.smoke_client
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== tools ===")
            for tool in tools.tools:
                print(f"- {tool.name}: {(tool.description or '').splitlines()[0]}")

            print("\n=== call: ask ===")
            res = await session.call_tool(
                "ask", {"question": "稳盈货币基金A的起购金额是多少？"}
            )
            payload = json.loads(res.content[0].text)
            print(json.dumps(payload, ensure_ascii=False, indent=2)[:700])

            print("\n=== call: get_evidence ===")
            res = await session.call_tool(
                "get_evidence", {"chunk_id": payload["citations"][0]["chunk_id"]}
            )
            ev = json.loads(res.content[0].text)
            print(f"found={ev['found']} doc={ev.get('document')} section={ev.get('section')}")
            print((ev.get("text") or "")[:120], "...")

            print("\n=== call: refusal ===")
            res = await session.call_tool("ask", {"question": "信用卡年费是多少？"})
            refused = json.loads(res.content[0].text)
            print(f"refused={refused['refused']} answer={refused['answer'][:40]}")

    print("\nOK: MCP server 与客户端握手、列工具、调工具全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
