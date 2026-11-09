# Financial QA 项目可信度改造 PRD

> 面向 Agent 岗求职的「可部署 / 可演示 / 可验证」项目改造方案

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0（初稿） |
| 编写日期 | 2026-11-09 |
| 作者 | 张艳 |
| 目标仓库 | github.com/bigbigstrange/Financial_QA |
| 本地开发目录 | C:\code\个人资料\code\Meeting\Financial_QA |
| 关联项目 | Agentic RAG v3（自适应路由检索） |
| 文档状态 | 待评审 → 评审通过后进入 P0 开发 |


## 1. 背景与问题


### 1.1 现状审计

对已克隆到本地（C:\code\个人资料\code\Meeting\Financial_QA）的仓库做了完整体检，结果如下：

| 检查项 | 现状 | 问题定性 |
|---|---|---|
| 提交历史 | 仅 1 个 commit（2026-08-14 “Initial commit”） | 看起来是“攒完一次性上传”，无法体现开发过程 |
| 代码规模 | 84 个文件 / 45 个 Python 文件（FastAPI + Vue3 平台骨架） | 骨架可用，但缺少可验证的测试 |
| 测试 | 仅有 test/*.xlsx（初赛、复赛测试集） | 有数据、没有跑数据的代码 |
| CI | 无 | 工程规范硬伤 |
| README | 一段架构说明，无图、无命令验证结果、无结论 | 面试官 30 秒内划走 |
| 可运行性 | 需要自备模型 Key 并自建索引 | 没有人会为你跑起来 |
| 可观测 | 有 src/system/observability.py，无对外展示 | 能力存在但不可见 |


### 1.2 尚未上传的核心资产

本地完整版（finance-qa-platform，提交至 2026-09-30）具备以下能力，但均未体现在 GitHub 仓库中：

- 约 150 个 Python 文件，app / src / tests 分层清晰
- 479 项测试通过、4 项跳过（pytest）
- 25 篇设计与审计文档（含交付审计、预算设计、质量门禁）
- LangGraph 六级节点证据链 + 三路条件边 + 双校验
- token 账本与 150 万预算熔断、检查点持久化、取消阻断

### 1.3 问题定义

| 编号 | 问题 | 影响 | 严重度 |
|---|---|---|---|
| P-1 | 简历指标（路由 92%、幻觉 8%→2.1%、延迟 3.2s）在仓库中没有出处 | 面试官一 clone 就发现无法复现，直接判定“数字是编的” | 致命 |
| P-2 | 仓库跑不起来（无 Key、无索引、无演示数据） | 面试官不会为你配置环境，等于没看 | 高 |
| P-3 | 仓库看不懂（无图、无 Demo、无结论） | 30 秒内无法建立“这人做过真实系统”的印象 | 高 |
| P-4 | 简历声称的“ToolMessage 计数实现 MAX_ITERATIONS=2”在 Agentic RAG v3 代码中不存在 | 细节被追问即穿帮 | 中 |


## 2. 目标与非目标


### 2.1 业务目标

支撑 2027 届校招 Agent 开发岗求职：让面试官在打开仓库的 3 分钟内，形成“这个人做过能跑、能验、有边界感的 Agent 系统”的判断。


### 2.2 产品目标（三条硬指标）

| 编号 | 目标 | 量化标准 |
|---|---|---|
| T-1 | 30 秒看懂 | README 首屏包含：一句话定位、架构图、3 个可验证数字、状态徽章 |
| T-2 | 3 条命令跑通 | 干净机器执行 docker compose up → 打开页面 → 完成一次带引用的问答，全程无需申请 Key |
| T-3 | 1 条命令复现指标 | python scripts/eval.py 输出的 metrics.json 与 README 展示数字完全一致 |


### 2.3 成功指标（验收口径）

- README 首屏：≥3 张图 + 1 个 GIF + 3 个徽章（CI / License / Python）
- Demo 模式：无 API Key 情况下，健康检查返回 ready、问答可用（缓存或 mock 回放）
- 评测：50 题数据集，一条命令输出 4 项指标，与 README 误差为 0
- CI：pytest + ruff + 前端 build 全绿，测试覆盖率 ≥ 60%
- 一致性：简历中 100% 的数字都能指向仓库内的具体文件

### 2.4 非目标

- 不做公网 SaaS、不做多租户与计费
- 不追求 SOTA 指标，不做模型训练/微调
- 不重写核心检索算法（DocIR / RRF / 双校验保持不变）
- 不伪造 Git 提交历史或时间戳

## 3. 用户与使用场景

| 角色 | 场景 | 关键诉求 |
|---|---|---|
| 技术面试官 | 筛简历时点开仓库 → 看 README → clone 跑一遍 → 挑刺 | 能否 3 分钟内判断真伪 |
| HR / 招聘 | 看活跃度、项目完整度、文档质量 | 是否有工程素养 |
| 候选人本人 | 面试现场演示、回答追问 | 有一键可复现的证据 |
| 未来协作者 | 接手或复现实验 | 有清晰的结构与命令 |


## 4. 总体方案


### 4.1 核心原则

> **任何一个数字，都必须能在仓库里找到：数据集 + 脚本 + 一条命令 + 输出文件。**

这套原则把“自证”变成“他证”：面试官不需要相信你，他可以自己跑。


### 4.2 三层改造模型

| 层级 | 解决什么 | 对应任务 | 完成后的效果 |
|---|---|---|---|
| 门面层 | 能不能看懂、能不能跑 | P0-1 ~ P0-4 | 陌生人 3 分钟跑起来 |
| 证据层 | 数字可不可信 | P1-1 ~ P1-3 | 一条命令复现全部指标 |
| 差异层 | 凭什么选你 | P2-1 ~ P2-3 | MCP / 可观测 / 第二仓库 |


### 4.3 目标仓库结构

```text
Financial_QA/
├── README.md                  # 首屏：定位 + 架构图 + 3 个数字 + 快速开始
├── LICENSE                    # MIT
├── docker-compose.yml         # api + frontend + demo 数据
├── .github/workflows/ci.yml   # pytest + ruff + build
├── app/                       # FastAPI 服务层（保留）
├── src/                       # 检索 / Agent / 解析
├── eval/
│   ├── dataset.jsonl          # 50 题评测集（问题 + 金标准来源）
│   ├── judge_prompts/         # 4 个评分提示词
│   └── results/               # metrics.json + report.md + 图表
├── scripts/
│   ├── eval.py                # 一条命令跑评测
│   ├── build_index.py         # 构建索引（保留）
│   └── seed_demo.py           # 灌入演示数据
├── docs/
│   ├── architecture.md        # 架构与数据流
│   ├── adr/                   # 关键决策记录（为什么自研 DocIR / RRF / 32 步上限）
│   └── evidence/              # 截图、GIF、trace、评测曲线
└── tests/                     # pytest（迁移本地 479 项中可公开的部分）
```


## 5. 需求清单


### 5.1 P0 · 门面层（预计 2.5 人日）

| 编号 | 需求 | 产出物 | 验收标准 | 工时 |
|---|---|---|---|---|
| P0-1 | 仓库清理与门面重写 | 删除 .env / data / output 等敏感与大文件；补 LICENSE(MIT)、.gitignore；重写 README | 陌生人 3 分钟内知道项目是什么、怎么跑 | 0.5d |
| P0-2 | 一键启动 + Demo 模式 | docker compose（api + frontend）；三级降级：真实模型 → 缓存回放 → mock；/api/health 返回 ready | 干净机器一条命令打开页面并完成一次问答 | 1d |
| P0-3 | CI 流水线 | .github/workflows/ci.yml；README 挂 build 徽章 | PR 上出现绿色 CI 徽章 | 0.5d |
| P0-4 | 演示数据包 | 3-5 份脱敏文档 + 预构建索引（Git LFS 或 Release 附件） | 启动即有数据，无需自行灌库 | 0.5d |


### 5.2 P1 · 证据层（预计 3.5 人日，最关键）

| 编号 | 需求 | 产出物 | 验收标准 | 工时 |
|---|---|---|---|---|
| P1-1 | 评测体系（★） | eval/dataset.jsonl（50 题）、scripts/eval.py、eval/results/metrics.json、自动生成的 report.md | 一条命令输出 4 项指标，与 README 完全一致 | 2d |
| P1-2 | 补齐「简历写了但代码没有」的能力 | state 内迭代计数 MAX_ITERATIONS=2、超时、显式 recursion_limit | 人为构造循环场景，图在 2 次后终止 | 0.5d |
| P1-3 | 过程证据包 | docs/evidence/：架构图、时序图、自纠正 trace、评测柱状图、CI 截图、终端 GIF | README 首屏嵌入 3 图 + 1 GIF | 1d |


### 5.3 P2 · 差异层（预计 4 人日）

| 编号 | 需求 | 产出物 | 验收标准 | 工时 |
|---|---|---|---|---|
| P2-1 | 可观测 | Langfuse 接入或 /ops 页面展示 trace、成本、延迟分布 | 任意一次问答可下钻到节点级耗时与 token | 1d |
| P2-2 | MCP server | 3 个 tool：search / ask / get_evidence；docker run 即用 | Claude/Cursor 等客户端可成功调用 | 2d |
| P2-3 | Agentic RAG v3 独立化 | 换领域语料（50 篇）+ 20 题评测 + 与主项目关系说明 | 形成“最小实现 → 生产平台”的对照叙事 | 1d |


### 5.4 P3 · 传播层（预计 1 人日）

| 编号 | 需求 | 产出物 | 验收标准 | 工时 |
|---|---|---|---|---|
| P3-1 | 演示视频 | 2-3 分钟录屏 + 旁白，B 站/YouTube 链接，简历二维码 | 视频内可完整走完一次带引用问答 | 0.5d |
| P3-2 | 技术文章 | 《我如何把幻觉率从 8% 压到 2.1%：一次可复现的评测》 | 发布并回链仓库 | 0.5d |


## 6. 关键设计


### 6.1 Demo 模式（三级降级）

| 级别 | 触发条件 | 行为 | 目的 |
|---|---|---|---|
| L1 真实模型 | 配置了 API Key | 正常调用，消耗额度 | 完整能力演示 |
| L2 缓存回放 | 无 Key，命中预置问答缓存 | 返回预置答案 + 真实引用 + 标注“缓存” | 零成本演示核心链路 |
| L3 Mock | 无 Key 且未命中缓存 | 返回结构化占位答案，明确标注“演示模式” | 保证永不 500 |

关键约束：任何降级都必须显式标注来源，不允许把缓存答案伪装成模型实时生成——这与项目的“证据可溯源”原则一致。


### 6.2 评测方案

数据集规范（eval/dataset.jsonl）：

- 字段：id、question、gold_sources（文档+页/块）、answer_type（事实/对比/拒答）、difficulty
- 规模：50 题，其中 10 题为“应拒答”负样本，用于验证拒答能力
四项指标定义：

| 指标 | 定义 | 计算方式 |
|---|---|---|
| 路由准确率 | query 被分配到正确通道的比例 | 与人工标注通道比对，取准确率 |
| 上下文相关性 | 进入生成的文档中相关文档占比 | 逐文档二分类（LLM-as-judge），取均值 |
| 幻觉率 | 答案中不被检索证据支持的断言占比 | 断言级判定，抽样 50 题人工复核校准 |
| 回答相关性 | 答案是否回答了问题的比例 | 二分类判定（LLM-as-judge） |

可复现性约束：

- judge 模型固定版本、temperature=0，评判提示词随仓库提交
- 数据集、索引、结果文件全部入库（索引走 LFS）
- 一条命令：python scripts/eval.py --dataset eval/dataset.jsonl --out eval/results

### 6.3 证据包规范

| 证据类型 | 文件 | 生成方式 | 放在哪 |
|---|---|---|---|
| 架构图 | docs/evidence/architecture.svg | Mermaid 源码 + 导出 | README 第 2 屏 |
| 自纠正时序 | docs/evidence/trace-selfrag.png | 一次 run 的 trace 截图 | README 特性段 |
| 评测结果 | docs/evidence/metrics.png | eval.py 自动出图 | README 指标段 |
| CI 状态 | docs/evidence/ci.png | PR 页面截图 | README 徽章下方 |
| 演示录屏 | docs/evidence/demo.gif | 终端录制（vhs / asciinema） | README 快速开始下方 |


### 6.4 CI 设计

```text
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5  # 3.11
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: pytest -q --cov=app --cov=src --cov-report=term
      - run: npm ci --prefix frontend && npm run build --prefix frontend
```


### 6.5 README 信息架构

| 位置 | 内容 | 目的 |
|---|---|---|
| 首屏（不滚动） | 一句话定位 + 3 个徽章 + 架构图 + 3 个数字 | 30 秒判断 |
| 快速开始 | 3 条命令 + 演示 GIF | 证明能跑 |
| 评测复现 | 一条命令 + 结果表 | 证明数字真实 |
| 设计说明 | 链接到 docs/adr/ | 证明有判断 |
| 边界与不足 | 明确写出未实现项 | 证明诚实、可信 |


## 7. 里程碑与排期

| 日期 | 任务 | 当日交付物 | 可对外状态 |
|---|---|---|---|
| D1 | P0-1 清理与 README 重写 | 新 README + 干净目录 | 仓库可公开访问 |
| D2 | P0-2 一键启动 + Demo 模式 | docker compose 跑通 | 面试官可自行运行 |
| D3 | P0-3 CI + P0-4 演示数据 | CI 绿勾 + 演示数据包 | ★ 可以挂到简历上 |
| D4-D5 | P1-1 评测体系 | dataset + eval.py + metrics.json | 数字可复现 |
| D6 | P1-2 迭代上限 + P1-3 证据包 | MAX_ITERATIONS + evidence 目录 | 简历表述 100% 落地 |
| D7 | 缓冲与复盘 | 发布 v1.0 Release | 第一版完成 |
| W2 | P2-1 可观测 / P2-2 MCP | /ops 与 MCP server | 差异化建立 |


## 8. 简历联动

| 任务 | 简历上哪句话会变 | 变化性质 |
|---|---|---|
| P1-1 评测体系 | 路由准确率 92%、幻觉率 2.1% 等指标 | 从“自述”变为“可复现” |
| P1-2 迭代上限 | ToolMessage 计数实现 MAX_ITERATIONS=2 | 从“不存在”变为“真实存在” |
| P0-2 Demo | 新增“提供零依赖演示模式，docker compose 一键启动” | 新增工程能力 |
| P0-3 CI | 新增“CI：pytest + ruff + 前端构建” | 补齐工程规范 |
| P2-2 MCP | 技能栏“熟悉 MCP” → “实现过 MCP server” | 能力升级 |


## 9. 风险与对策

| 风险 | 触发条件 | 对策 |
|---|---|---|
| 伪造历史被发现 | 想快速做出“长期开发”的假象 | 保留真实 initial commit，从今天起逐步提交，commit message 写清楚 |
| 指标虚高被戳穿 | 评测跑出来的数字低于简历 | 以真实结果改写简历：写“从 X 提升到 Y”，而不是绝对值 |
| Demo 跑不起来 | 依赖外部模型或大数据文件 | 三级降级 + 预置索引 + 健康检查 |
| 时间不够 | 面试临近 | P0 三天可交付，先挂出去；P1 边面边补 |
| 数据合规 | 使用真实业务文档 | 全部脱敏：公司名打码、金额替换、账号隐藏，仅保留结构 |


## 10. 验收清单（Definition of Done）

- [ ] 干净机器 docker compose up 成功，页面可访问
- [ ] 无 API Key 时仍可完成一次带引用的问答，并标注“缓存/演示”
- [ ] python scripts/eval.py 输出 metrics.json，与 README 数字一致
- [ ] CI 全绿，覆盖率 ≥ 60%
- [ ] README 首屏含 3 图 + 1 GIF + 3 徽章
- [ ] docs/evidence/ 下 5 类证据齐全
- [ ] 简历中每一个数字都能指向仓库内的文件路径
- [ ] 仓库无密钥、无真实业务数据、无 >5MB 的二进制文件

## 附录 A · Agentic RAG v3 设计解析

代码位置：C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag

整体是 Adaptive RAG 与 Self-RAG 的组合：路由决定“查哪里”，文档评分决定“用不用”，双校验决定“重生成还是换源”。


### A.1 图结构

```text
1) 入口：route_question（条件入口，二选一分类）
   |- vectorstore -> retrieve -> grade_documents
   |- websearch   -> web_search

2) grade_documents：逐篇二分类，任一不相关即置 web_search=True
   |- 存在不相关文档 -> web_search（补充外部来源）-> generate
   |- 全部相关       -> generate

3) generate -> hallucination_grader（答案是否基于检索文档）
   |- not supported -> 回到 generate（用同一批文档重生成）
   |- 通过          -> answer_grader（是否回答了问题）

4) answer_grader
   |- useful     -> END
   |- not useful -> web_search（换信息源）-> generate
```


### A.2 三个关键设计决策

| 决策 | 代码位置 | 设计意图 |
|---|---|---|
| 路由字段用 Literal 强约束 | chains/router.py:11 | 把开放生成降维为二选一分类，避免自由文本解析失败 |
| 逐文档二分类而非整体打分 | nodes/grade_documents.py:25-36 | 一篇不相关即触发补充检索，宁可多搜一次也不放进脏上下文 |
| 两级校验分离 | graph.py:56-78 | 幻觉→重新生成；偏题→换信息源；错误类型决定补救动作 |


### A.3 三个硬伤（必须修）

| 问题 | 证据 | 影响 | 修复任务 |
|---|---|---|---|
| 无迭代上限 | 全仓库无 MAX_ITERATIONS，仅 server.py:48 的 recursion_limit=8 | not supported → generate 可能反复循环 | P1-2 |
| 语料为教程样例 | ingestion.py:28-49 仅 5 段英文示例（agent memory / prompt engineering / adversarial attacks） | 与“金融问答”名不符实 | P2-3 |
| 无评测 | 无 eval 脚本、无数据集、无指标输出 | 简历指标无出处 | P1-1 |


## 附录 B · 简历—代码一致性差异表

| 简历表述 | 仓库现状 | 处理方式 |
|---|---|---|
| 路由准确率 92%、无效检索减少 30% | 无评测脚本 | P1-1 建立评测后重新测量，用真实值替换 |
| 幻觉率 8% → 2.1%、相关性 67% → 89% | 无评测脚本 | 同上，并保留 judge 提示词与数据集 |
| ToolMessage 计数实现 MAX_ITERATIONS=2 | 代码中不存在 | P1-2 实现后保留；实现前从简历删除 |
| 端到端延迟 3.2s | 无 benchmark 脚本 | P1-1 增加延迟统计（p50/p95） |
| 479 项测试通过 | GitHub 仓库无 tests 目录 | P0-3 迁移可公开测试并接入 CI |

---

## 11. 执行记录

> 口径：只有**仓库里能找到证据**的才标 ✅；做了但没验收的标 🟡；没动的标 ⬜。

| 编号 | 任务 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| P0-1 | 仓库清理与门面重写 | ✅ | 删除 `.env` / 业务数据；补 MIT LICENSE、`.gitignore`；README 重写 |
| P0-2 | 一键启动 + Demo 模式 | 🟡 | `app/demo_engine.py` + `demo/corpus/` + `docker-compose.demo.yml` 已就绪，**容器未实跑验收** |
| P0-3 | CI 流水线 | ✅ | `.github/workflows/ci.yml`：ruff + pytest + 前端构建，三次全绿；失败会写 check-run 注解 |
| P0-4 | 演示数据包 | 🟡 | 已有 4 篇示例语料（16 切块）；**预构建索引未提供**（演示模式不需要，但真实 KB 演示仍缺） |
| P1-1 | 评测体系 | 🟡 | 已完成**示例语料** 23 题（Recall@3 94.7% / 拒答 100%）；**真实语料 50 题、92% / 2.1% 仍无出处** |
| P1-2 | 迭代上限与超时 | ⬜ | `recursion_limit=32` 已存在且有测试；**`MAX_ITERATIONS=2` 未实现**（该表述属 Agentic RAG v3，尚未落地） |
| P1-3 | 过程证据包 | 🟡 | 已有架构图 / 评测图 / 终端 GIF（3 类）；**缺 trace 时序图、CI 截图** |
| P2-1 | 可观测 | ⬜ | 未接 Langfuse，`/ops` 页面未补 trace 下钻 |
| P2-2 | MCP server | ✅ | `mcp_server/`：4 个 tool，stdio 实测握手通过，7 条契约测试 |
| P2-3 | Agentic RAG v3 独立化 | ⬜ | 未换领域语料、未补 20 题评测 |
| P3-1 | 演示视频 | ⬜ | 未录制 |
| P3-2 | 技术文章 | ⬜ | 未撰写 |

### 验收清单（DoD）对照

| 检查项 | 状态 | 说明 |
|---|---|---|
| 干净机器 `docker compose up` 成功 | ⬜ | 仅提供 compose 文件，未实跑 |
| 无 Key 可完成带引用问答 | ✅ | 演示模式实测 |
| `eval.py` 与 README 数字一致 | ✅ | 示例语料口径一致 |
| CI 全绿 | ✅ | — |
| 覆盖率 ≥ 60% | ⬜ | 未接 `pytest-cov` |
| README 首屏 3 图 + 1 GIF + 3 徽章 | ✅ | 架构图 / 评测图 / GIF + 5 个徽章 |
| `docs/evidence/` 5 类证据齐全 | 🟡 | 3 / 5 |
| 简历每个数字可指向文件 | ⬜ | 92% / 2.1% / 3.2s 仍无法在仓库复现 |
| 仓库无密钥、无真实业务数据 | ✅ | `git ls-files` 检查通过；面试准备脚本已移除 |

### 下一批优先级

1. **P1-2**：在 Agentic RAG v3 补 `MAX_ITERATIONS=2` + 超时（否则简历那句话站不住）
2. **真实语料 50 题评测**：把 92% / 2.1% 变成可复现数字，或按真实值改写简历
3. **P0-2 验收**：`docker compose -f docker-compose.demo.yml up --build` 实跑一遍
4. P2-1 可观测 / P2-3 v3 独立化
5. P3 视频与文章
