# MCP Server

把 Financial_QA 的「检索 / 问答 / 证据」暴露成标准 **MCP（Model Context Protocol）** 工具，
任何支持 MCP 的客户端（Claude Desktop、Cursor、IDE 插件、自研 Agent）都能直接调用。

**不需要 API Key**：默认走 `demo/corpus` 的示例语料，连上就有结果。

---

## 工具契约

| 工具 | 作用 | 关键返回字段 |
|---|---|---|
| `search_corpus(query, top_k=3)` | 检索语料，返回最相关的切块 | `chunk_id` / `document` / `section` / `score` / `snippet` |
| `ask(question)` | 回答问题并给出引用 | `answer` / `verified` / `confidence` / `refused` / `citations` |
| `get_evidence(chunk_id)` | 按 id 取回原文，用于核对 | `found` / `text` / `document` / `section` |
| `list_documents()` | 列出语料覆盖范围 | `document` / `chunks` / `sections` |

设计要点：**三个工具对应一条"先搜、再问、后核对"的工作流**。
`ask` 在证据不足时返回 `refused=true` 而不是编造答案——这条契约由服务端保证，不依赖提示词。

---

## 接入客户端

以 Claude Desktop（`claude_desktop_config.json`）为例：

```json
{
  "mcpServers": {
    "financial-qa": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/path/to/Financial_QA"
    }
  }
}
```

Cursor / 其他客户端同理：命令 `python -m mcp_server.server`，工作目录设为仓库根目录。

---

## 本地验证

```bash
# 1) 协议冒烟：真的起子进程，走 stdio 完成握手 → 列工具 → 调工具
python -m mcp_server.smoke_client

# 2) 单元测试
pytest tests/unit/test_mcp_server.py -q
```

冒烟输出示例：

```text
=== tools ===
- search_corpus: 检索示例语料，返回最相关的文档切块（含来源与分数）。
- ask: 回答一个金融业务问题，返回答案、置信度与引用来源。
- get_evidence: 按 chunk_id 取回切块原文与出处，用于核对答案是否被原文支持。
- list_documents: 列出当前语料的文档与切块数量，用于判断知识库覆盖范围。

=== call: ask ===
{"answer": "稳盈货币基金 A（示例代码 000000）为货币市场基金，1 元起购…",
 "verified": true, "citations": [{"chunk_id": "01_fund#0", ...}]}

=== call: refusal ===
refused=True answer=根据现有示例资料无法回答该问题（证据不足，已拒答）。
```

---

## 接真实知识库

`server.py` 里的三个工具只依赖 `app.demo_engine` 的两个函数：

```python
from app.demo_engine import search, answer_demo
```

替换成生产检索栈（`src.system.retrieval` + `app.services.qa_runtime`）即可，
工具签名与返回结构不变——这正是把工具层与引擎层分开的目的。
