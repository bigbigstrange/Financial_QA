# 金融多模态知识与业务协作 Agent 平台

> 面向金融业务文档的多模态检索与问答平台：**从 923 份多格式文档，到一条可溯源、可拒答、可审计的答案。**
>
> 不是「接个大模型做问答」，而是把 **证据链、幻觉校验、预算熔断、检查点恢复** 做成工程约束。

[![ci](https://github.com/bigbigstrange/Financial_QA/actions/workflows/ci.yml/badge.svg)](https://github.com/bigbigstrange/Financial_QA/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-MIT-blue.svg)
![python](https://img.shields.io/badge/python-3.11-3776AB.svg)
![tests](https://img.shields.io/badge/tests-490%20passed%20%2F%204%20skipped-brightgreen.svg)
![coverage](https://img.shields.io/badge/coverage-67%25-brightgreen.svg)
![status](https://img.shields.io/badge/status-local%20demo%20verified-orange.svg)

---

## 它解决什么问题

金融业务文档有三个特点，普通 RAG 直接上会翻车：

| 现实问题 | 常见做法 | 本项目的做法 |
|---|---|---|
| 文档格式杂（PDF / Word / Excel / PPT / 扫描件），表格与图表承载关键信息 | 只抽文本，图表信息丢失 | 统一中间表示 **DocIR**（文档版本 / 原文块 / 物理页 / 页图），文本与页图双通道检索 |
| 答错代价高，模型会「编」 | 让模型直接答，事后人工核对 | **依据校验 + 答题校验**双层拦截，证据不足明确拒答 |
| 调用成本不可控 | 跑完才发现账单 | **token 账本 + 预算熔断**，超限即停，不越权重跑 |

一句话：**把"模型能不能答"变成"系统允不允许答"。**

---

## 架构

![架构图](docs/evidence/architecture.png)

**四通道各自保留独立分数，不做归一化相加**：视觉相似度和文本相似度不是同一种量纲，直接相加会让"看起来像"压过"内容对"。

<details>
<summary>Mermaid 源码（可编辑版）</summary>

```mermaid
flowchart TD
    A[PDF / Word / Excel / PPT / 图片] --> B[隔离转换 + 本地解析 + OCR]
    B --> C[DocIR<br/>文档版本 · 原文块 · 物理页 · 页图]
    C --> D[版本化切块 + 知识快照 + 向量索引]

    D --> E1[文本塔<br/>BGE-M3 + FAISS]
    D --> E2[BM25<br/>稀疏检索]
    D --> E3[结构检索<br/>表格 / 条款]
    D --> E4[页图塔<br/>Chinese-CLIP]

    E1 --> F[RRF 融合 + 机构/产品约束 + 重排]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[LangGraph 证据链<br/>路由 → 证据评分 → 有限补查 → 生成]
    G --> H[依据校验 + 答题校验]
    H -->|通过| I[答案 + 原文块 / 页图引用]
    H -->|有限重试| G
    H -->|证据不足| J[明确拒答]
    I --> K[SSE 轨迹 · 内部核查工单]
```

</details>

---

## 效果预览

**零依赖演示（真实运行，非手写）**：两次命中生成引用，一次证据不足明确拒答。

![终端演示](docs/evidence/terminal-demo.gif)

| 问 | 结果 |
|---|---|
| 稳盈货币基金A的起购金额是多少？ | 命中 `01_fund#0`（score 0.65）→ 回答 + 引用 |
| 定期开放产品多久开放一次？ | 命中 `03_wealth#1`（score 0.47）→ 回答 + 引用 |
| 信用卡年费是多少？ | 最高分 0.00 < 阈值 0.35 → **拒答**，不编 |

---

## 关键数字

> 数据来自 2026-09-30 的实际运行记录，评测报告与原始问答均在仓库内可查。

| 指标 | 数值 | 出处 |
|---|---|---|
| 测试 | **490 passed / 4 skipped / 12 subtests**，覆盖率 **67%**（CI 门槛 60%） | `pytest -q --cov`（约 19s） |
| 代码规模 | **221 个 Python 文件**（app 30 / src 46 / tests 81 / scripts 58 / mcp_server 3） | `git ls-files` 口径 |
| 语料规模 | **920 份**业务文件 → **30,406** 检索切块 → **17,957** 张页图 | `docs/baseline/2026-09-29-delivery-audit.md` |
| 真实模型评测 | 50 题完成 **43 题**（第 44 题预算耗尽） | `output/quality/financial50-live-v1.json` |
| 延迟 | **p50 30.5s / p95 60.2s** | 同上 |
| 预算 | 累计 **1,488,843 / 1,500,000** tokens，触发熔断即停 | 预算账本 |
| 解析兼容率 | **100%**（1,146 个文件的历史评测口径） | 早期评测报告 |
| 离线评测（示例语料） | Recall@3 **94.7%**、拒答准确率 **100%**、1.68 ms | `eval/results/metrics.json` |

---

## 快速开始

> **不需要任何 API Key**：演示模式用 `demo/corpus/` 的示例语料跑通「检索 → 评分 → 拒答 → 引用」。

```bash
# ① 零依赖演示（2 分钟）
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
set FINANCE_DEMO_MODE=1 && python -m app.main      # 打开 http://127.0.0.1:8000

# ② 一键容器（首次构建约 5-10 分钟）
docker compose -f docker-compose.demo.yml up --build   # 打开 http://127.0.0.1:8002

# ③ 接入真实模型（可选）
copy .env.example .env      # 填 ARK_API_KEY 后 python -m app.main
```

**评测复现**（离线，不调用付费 API）：

```bash
python scripts/eval_demo.py
# → eval/results/metrics.json + report.md
```

**回归测试**：

```bash
set FINANCE_DISABLE_PAID_API=1
.venv\Scripts\python.exe -m pytest -q
```

> 说明：`data/` 下的原始语料与 `output/` 下的知识产物、向量缓存、模型权重不入库，需本地自行准备（见 `.gitignore` 说明）。
> `docker compose up`（生产形态，含 PostgreSQL）尚未完成端到端验收，不作为演示入口。

---

## 评测（可复现）

```bash
python scripts/eval_demo.py
```

| 指标 | 结果 | 口径 |
|---|---|---|
| Recall@3 | **94.7%**（18/19） | 事实题命中金标准切块即算通过 |
| 拒答准确率 | **100%**（4/4） | 语料外问题必须明确拒答，不许硬编 |
| 平均耗时 | 1.68 ms | 纯本地词面检索（无向量库、无大模型） |
| 数据集 | 23 题（19 事实 / 4 拒答） | `eval/dataset.jsonl` |

![评测结果](docs/evidence/metrics.png)

唯一未命中：**「货币基金保本吗？」** —— 语料原文写的是「不保证本金安全」，词面检索匹配不到「保本」。
**这正是生产版要接向量检索与重排的原因**，而不是把评测集改到好看。

结果文件随仓库提交：`eval/results/metrics.json`、`eval/results/report.md`；
换任何机器重跑，数字应当一致。

---

## MCP Server（可被任何 Agent 客户端调用）

把「检索 / 问答 / 证据」暴露成标准 **MCP（Model Context Protocol）** 工具，**不需要 API Key**：

| 工具 | 作用 | 关键返回 |
|---|---|---|
| `search_corpus(query, top_k)` | 检索语料，定位相关条款 | `chunk_id` / `score` / `snippet` |
| `ask(question)` | 回答并给出引用，证据不足则拒答 | `answer` / `verified` / `citations` |
| `get_evidence(chunk_id)` | 按 id 取回原文核对 | `found` / `text` |
| `list_documents()` | 语料覆盖范围 | `document` / `chunks` |

接入 Claude Desktop / Cursor：

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

验证（真的起子进程、走 stdio 完成握手 → 列工具 → 调工具）：

```bash
python -m mcp_server.smoke_client
```

```text
- ask: 回答一个金融业务问题，返回答案、置信度与引用来源。
- get_evidence: 按 chunk_id 取回切块原文与出处，用于核对答案是否被原文支持。
=== call: refusal ===
refused=True answer=根据现有示例资料无法回答该问题（证据不足，已拒答）。
OK: MCP server 与客户端握手、列工具、调工具全部通过
```

**两个设计取舍**：
1. **工具层与引擎层解耦**——三个工具只依赖 `search()` / `answer_demo()`，换成生产检索栈时契约不变；
2. **拒答由服务端保证**，不依赖提示词——模型再想编，也拿不到证据。

---

## 过程证据

### 一次真实问答的节点时序

![trace](docs/evidence/trace.png)

这是从容器里跑出来的一次真实请求（30,406 切块知识库，本地检索模式）：检索 12 → 18 → 24 条，
三轮评分都未通过，`rewrite_query` 触发两次重试，最后 `give_up` 明确拒答——**而不是硬编一个答案**。

### CI 状态

![ci](docs/evidence/ci-status.png)

三个 job 全绿，覆盖率 67.25%（门槛 60%）。图由 `scripts/render_ci_status.py` 直接读
GitHub Actions API 生成，**不是截图**，所以不会过期。

---

## 两个实现对照

仓库里有两个 Agentic RAG 实现，**故意放在一起**：

| | `agentic-rag-v3/`（最小实现） | 根目录（生产平台） |
|---|---|---|
| 目的 | 把「路由 → 评分 → 双校验 → 重试/拒答」讲清楚 | 把这套编排跑在真实业务语料上 |
| 检索 | Chroma 单塔 | 文本 / BM25 / 结构 / 页图 四通道 + RRF |
| 文档层 | 直接切块 | DocIR（版本 / 原文块 / 物理页 / 页图） |
| 规模 | 5 段示例语料，7 条测试 | 30,406 切块，490 条测试 |
| 依赖 | langgraph + langchain + chroma | + faiss / torch / BGE-M3 / CLIP / FastAPI / Vue |

详见 [agentic-rag-v3/README.md](agentic-rag-v3/README.md)。

---

## 可观测（可选）

```bash
# 把一次 run 的 trace 转成 Langfuse ingestion 格式（默认 dry-run）
python scripts/export_langfuse.py --trace trace.json --dry-run

# 配好 LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 后实际发送
python scripts/export_langfuse.py --trace trace.json
```

不依赖 Langfuse 也能跑：仓库自带 `docs/evidence/trace.png` 与 `/api/runs/{id}/trace` 接口。

---

## 演示视频

录制脚本（分镜 + 旁白 + 剪辑清单）见 [docs/demo-script.md](docs/demo-script.md)。

---

## 目录结构

```text
Financial_QA/
├── app/                # FastAPI 服务层：JWT、会话、run 生命周期、SSE、证据鉴权、工单、运维
├── src/
│   ├── pipeline/       # 多格式解析、DocIR、身份、表格切块、快照、质量门禁
│   ├── system/         # 多通道检索、RRF、机构过滤、模型适配、预算
│   └── agent/          # LangGraph 节点、条件边、提示词、持久节点日志
├── frontend/           # Vue3：登录、问答、轨迹、原文原图、取消与运维
├── demo/corpus/        # 演示用示例语料（虚构条款，非真实业务数据）
├── mcp_server/         # MCP Server：检索 / 问答 / 证据 三个工具（stdio）
├── agentic-rag-v3/     # 对照实现：最小可读的 Adaptive RAG + 自纠正（7 条测试）
├── eval/               # 评测集 + 结果（metrics.json / report.md）
├── tests/              # pytest：单测 + 真实 HTTP 验收
├── scripts/            # 数据处理、评测、备份、回归
├── docs/               # 设计、审计、runbook（26 篇）
└── configs/            # 模型注册表
```

---

## 核心设计

### 1. 文档统一表示 DocIR

所有格式先落到同一层中间表示：**文档版本、原文块、物理页或结构位置、页图**。块级保留 `source_doc / page / block_id`，答案才能回指到原文。

### 2. 四通道检索 + RRF 融合

文本向量（BGE-M3 + FAISS）、BM25、结构检索、页图（Chinese-CLIP）各自保留分数；RRF 融合后做机构与产品约束重排。页图检索失败**不伪造文本代理向量**。

### 3. LangGraph 证据链

```
路由 → 证据评分 → 有限补查 → 生成 → 依据校验 + 答题校验
                    ↑                              |
                    └──────── 有限重试 ────────────┘
```

节点总步数硬限制 **32**，检索与生成重试各自有限额。取消会阻断后续节点，已发送的远端请求不保证撤回或免计费，**迟到的答案不发布**。

### 4. 预算账本与熔断

按 run 记账、按轮结算，累计触达上限即置 `budget_exhausted` 并停止调用。**不允许通过回退账本来获得新额度。**

### 5. 检查点与幂等恢复

run、事件、LangGraph 检查点与节点结果持久化；恢复时复用已落盘结果。未知模型结果、不同代码或配置、过期任务**不自动重放**。

---

## 评测与边界（重要）

这份 README 只写**已经发生**的事，没做到的单独列出来：

**已做到**
- 479 项自动化测试通过，含真实 HTTP 验收（不调用付费 API）
- 920 份文件入库，30,406 切块，17,957 张页图
- 50 题真实模型评测完成 43 题，延迟与 token 账本可查
- 证据不足时明确拒答，答案附原文块与页图引用

**未做到（不宣称）**
- ❌ 500 题准确率门禁**未通过**：题源表缺人工标准答案与证据，正式准确率指标保持"未通过"
- ❌ 39 个"模型校验标记"不等于 39 个人工正确答案
- ❌ 单次 RAG 对照未运行，**不报告 A/B 准确率提升**
- ❌ 图谱 1,792 条关系**未通过语义复核**，默认不上线
- ❌ 多租户私有文档 ACL 未实现（当前知识快照对登录用户共享，run / 会话 / 证据 / 工单按用户隔离）
- ❌ Docker 部署未做端到端验收，公网生产上线未验收

> 这一节的用意：**能说清边界的人，才值得相信他说的数字。**

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/PRD.md](docs/PRD.md) | 可信度改造 PRD（门面 / 证据 / 差异三层任务） |
| [docs/baseline/2026-09-29-delivery-audit.md](docs/baseline/2026-09-29-delivery-audit.md) | 交付审计：实际规模与未完成项 |
| [docs/deployment-readiness.md](docs/deployment-readiness.md) | 部署边界说明 |
| [docs/2026-09-21-agentic-rag-upgrade-master-plan.md](docs/2026-09-21-agentic-rag-upgrade-master-plan.md) | Agentic RAG 升级总纲 |

---

## License

MIT © 2026 张艳






