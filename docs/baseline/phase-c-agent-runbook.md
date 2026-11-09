# 阶段 C：Agent 编排运行手册

> 状态：图与接入已实现并通过自动化测试；端到端真实问答与前端面板待验收
> 日期：2026-09-21
> 适用仓库：`finance-qa-platform`
> 上位方案：`docs/superpowers/specs/2026-09-21-agentic-rag-upgrade-master-plan.md`

## 1. 本阶段目标

把「一次性检索 → 直接生成」升级为带**自纠正闭环**的 Agent 编排：

```
路由分发 → 子问题规划 → 混合检索 → 证据相关性评分
   → 条件分支（有相关 / 全无关） → 生成
   → 双层校验（幻觉 → 回生成 · 偏题 → 换信息源）
   → 生成答案 + 可回放的决策轨迹
```

两条纠正路径**必须分开**：
- **幻觉**（`grounded=False`）→ 回 `generate`：证据没问题，是生成跑偏了，换信息源无意义；
- **偏题**（`answers_question=False`）→ 回 `rewrite_query` → 重检索：换检索参数与信息源。

## 2. 开关与配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FINANCE_AGENT_MODE` | `legacy` | `graph`=LangGraph 编排；`legacy`=原有轻量 DocumentAgent |
| `FINANCE_AGENT_TOP_K` | 12 | 每次检索的候选数（多子问题会合并去重） |
| `FINANCE_AGENT_MAX_RETRY_RETRIEVE` | 2 | 偏题后允许的重检索次数 |
| `FINANCE_AGENT_MAX_RETRY_GENERATE` | 1 | 幻觉后允许的重生成次数 |
| `FINANCE_AGENT_GRADE_LLM` | 1 | 是否用 LLM 做证据相关性评分（关掉则全部保留） |
| `FINANCE_AGENT_VERIFY` | 1 | 是否启用双层校验（关掉则两层都直接通过） |

**成本控制**：证据评分是**一次批量调用**（不是逐条一次），10 条证据只花 1 次请求。
无 LLM 时图退化为「总是检索 + 抽取式作答」，不会失败。

## 3. 启动

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'

# 数据库增量迁移（新增 runs.agent_trace_json 列，幂等）
.\.venv\Scripts\python.exe -c "from app.db.init_db import init_db; init_db(); print('ready')"

# 启动 API（走 KB v3 + GPU）
.\.venv\Scripts\python.exe -m app.main
```

关键 `.env` 项（全量模式）：

```
FINANCE_WEB_DEMO=0
FINANCE_EMBED_DEVICE=cuda
KB_PATH=output/kb_v3/kb_v3.json
FINANCE_AGENT_MODE=graph
```

## 4. 调用与验证

```powershell
$login = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/auth/login' `
  -ContentType 'application/json' -Body '{"username":"admin","password":"admin123"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }

# 一次带 Agent 编排的问答
$run = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/runs' `
  -Headers $headers -ContentType 'application/json' `
  -Body '{"question":"农业银行2024年一季度营业收入和归母净利润分别是多少？"}'

$run.run_id
$run.route              # 路由决策：retrieval | direct | clarify
$run.retry_retrieve     # 偏题重检索次数
$run.retry_generate     # 幻觉重生成次数
$run.trace | Format-Table node, elapsed_ms, decision

# 单独读决策轨迹
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/runs/$($run.run_id)/trace" -Headers $headers
```

期望的轨迹形状：

```
node                  elapsed_ms  decision
----                  ----------  --------
route_query                   12  retrieval
plan_subquestions            340  1 个子问题
retrieve                     820  8 条证据
grade_documents             1100  保留 8/12
generate                    2400  420 字
check_hallucination          600  True
check_answer                 580  True
```

## 5. 验收标准

| # | 验收项 | 阈值 | 验证方式 |
|---:|---|---|---|
| 1 | 10 类问题全部正常返回 | 无异常 | 手工提问（事实/数值/流程/多意图/无关/问候…） |
| 2 | `route_query` 准确率 | ≥ 85% | 人工标注 50 题期望路由后比对 |
| 3 | 证据评分有效性 | 过滤后噪声率下降 ≥ 30% | 对比过滤前后的证据与问题相关性 |
| 4 | **双层校验真的会触发** | 3 个陷阱题 | 构造「证据里没有的数字」验证回 `generate`；构造「答非所问」验证回 `rewrite_query`，且 `retry_*` 正确递增不超上限 |
| 5 | 无依据不编造 | 5 个库外问题 | 全部输出「知识库中无依据」而非编造 |
| 6 | 轨迹可查 | 全部节点出现 | `GET /api/runs/{id}/trace` |
| 7 | 端到端时延 | 简单题 < 8s，难例 < 25s | 记录 baseline 对比 |
| 8 | **可回退** | 全绿 | `FINANCE_AGENT_MODE=legacy` 后现有 202 个测试全过、接口形状不变 |

## 6. 自动化测试

```powershell
# Agent 图结构与条件边（16 个）
.\.venv\Scripts\python.exe -m unittest tests.unit.test_agent_graph -v

# Agent 轨迹持久化（5 个）
.\.venv\Scripts\python.exe -m unittest tests.unit.test_run_agent_trace -v

# 全量回归（202 个）
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app src scripts tests
```

覆盖的关键场景：
- 快乐路径节点顺序、trace 计时、多子问题去重
- direct / clarify 短路（不进入检索）
- **幻觉回 generate 不回 retrieve**、幻觉达上限放弃
- **偏题回 rewrite_query**、偏题无限循环回归、偏题达上限放弃
- 无 LLM 降级、LLM 抛异常不崩图、检索抛异常不崩图
- 轨迹落库与读回、legacy 默认值、失败运行、畸形 JSON 容错

## 7. 回退

```powershell
# 1) 停 API
# 2) 切换模式（无需改代码）
$env:FINANCE_AGENT_MODE='legacy'
.\.venv\Scripts\python.exe -m app.main
```

`runs.agent_trace_json` 列会被 legacy 代码忽略，**不要删除**（保留审计数据）。

## 8. 已知遗留

1. **前端轨迹面板未做**：`/trace` 接口就绪，Vue 侧待渲染；
2. **checkpointer 未挂**：`AsyncSqliteSaver` 已安装但未接入图，暂无断点续跑；
3. **SSE 未做**：轨迹只能在运行结束后一次性读取，没有节点级实时推送；
4. **`/api/chat` 未单独验证**：它内部经 `RunService`，理论上与 `/api/runs` 一致。

## 9. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| **检索永远返回 0 条、答案总是「知识库中无依据」** | `_EmptyImageRetriever.search()` 曾不接受 `visibility` 参数，而 `FusionRetriever` 无条件传它 → 抛 `TypeError` 被上层吞掉。`FINANCE_SKIP_IMAGE=1` / 演示模式下**必然触发** | 已修复：`search(self, query, top_k=6, **kwargs)`；并加契约测试 `tests/unit/test_retriever_contracts.py` 锁定「所有会被传入 visibility 的检索器都必须接受它」 |
| **编码极慢且显存接近打满** | `FINANCE_EMBED_BATCH_SIZE` 取了 CUDA 默认值 64；**torch 2.1.0 下 transformers 不启用 SDPA**，退回 eager attention 后显存/耗时随批大小增长 | `FINANCE_EMBED_BATCH_SIZE=16`；并把 `max_seq_length` 从 BGE-M3 默认的 8192 降到 512（`FINANCE_EMBED_MAX_SEQ_LEN`），实测峰值从约 8.2GB 降到 2~3GB |
| **CUDA out of memory** | 显存不足：BGE-M3（数 GB）+ MinerU VLM（约 2.5GB）+ 其它常驻服务（如 index-tts WebUI 约 6GB）**三者叠加超过 10GB** | 重任务串行执行；跑索引/评测前确认空闲显存 ≥ 4GB |

## 10. 2026-09-23 Doubao 有界试跑记录

- 目标模型：`doubao-seed-2-1-pro-260915`，Ark Chat Completions。密钥继续留在用户本地文件；项目忽略的 `.env` 只保存文件路径、模型与预算，不复制密钥。
- 持久配置 `FINANCE_DISABLE_PAID_API=1`；探针脚本仅在自身进程内临时启用。上线服务尚未打开付费模型。
- 探针命令：`.venv\Scripts\python.exe -m scripts.probe_ark_pilot --send`，仅 1 次，请求输出上限 16 tokens。成功返回目标模型；用量台账记录本次 **53 tokens**，本地 1,500,000-token 上限尚余 **1,499,947 tokens**。
- 探针前完整离线回归：`323 passed, 4 skipped, 7 subtests passed`。模型提供方、错误不重试、用量缺失停机、跨进程预算预留、掌上银行错源拒答等测试均通过。
- **等待火山控制台资源包剩余额度核对。** 用户此前报告剩余 2,499,950 tokens；不能直接从本地 53 tokens 推定云端一定扣减 53。核对前不继续付费问答或图谱抽取。
- 探针后用户刷新控制台：“已采”由 **50→103**，恰好增加 53；但资源包剩余仍为 **2,499,950**、已抵扣仍为 **2,530,755**。这只证明采集计数吻合，**不证明资源包抵扣或零费用**。已请求核对本次推理的费用明细；明确前继续禁止付费请求。
- 用户进一步说明账户**未充值、只打算使用免费额度**；但没有提供该调用的费用明细，资源包数字依旧没有扣减证据。因而截至本记录，**总共只有 1 次真实 Ark 调用（53 tokens）**，不启动第二个付费请求。SQLite 台账核对：1 条 `probe/settled/53`，停止标志为空；本地预算剩余 1,499,947，仅代表本地安全线，不代表云端免费余额。
- 图谱：`scripts/run_kg_pilot.py` 已准备隔离批次 `output/lightrag_pilot/batch_0629a5a79bab413480e90fed6e4de981`，只含活动知识库 `0040ff6d423b4b6c918d3502210ce2fe.xls` 第 1 页 2 个块，附来源/页码 manifest；**没有向 LightRAG 发送请求**，旧图谱未清理。独立 LightRAG 尚未证明逐调用 Ark 用量能进入本地台账，故“图谱待计量”，不能称为抽取完成。
- 视觉：可信页图路径、摘要、格式、大小与页码校验及固定 50,000-token 预算预留已写入；默认不启用 `FINANCE_PILOT_VISUAL`，**未发生付费图像调用**，只命中视觉索引时仍标“未经图像内容验证”。图片内容的双校验也尚未打通。
- 实时轨迹：后端 `POST /api/runs/stream` 带原有鉴权，在运行期间发送 `started → node* → done/error`，节点事件只含名称、决策摘要和耗时；前端新增显式“实时试跑轨迹（不写入会话）”开关，普通 `/api/chat` 会话路径不变。试跑模式结果落运行记录但不写入当前会话消息，用户界面已有提示。
- 离线回归：Python 全套 **338 passed、4 skipped、7 subtests passed**（后续小修需重新跑最终回归）；前端实时流、图片预览、健康计数测试和生产构建通过。在线健康接口仍报活动快照 920 份（918 文本 + 2 仅图像）、`paid_api_disabled=true`。当前运行中的 8001 服务进程尚未重启，OpenAPI 暂未加载新 `/runs/stream` 路由；部署验证仍待完成。
- 最新回归（上述记录之后）：Python 全套 **342 passed、4 skipped、7 subtests passed**，前端四项脚本测试与 `npm --prefix frontend run build` 通过；修复了“两个校验节点执行过但判断为 false，却错误标成 verified=true”的问题，并强制 Ark 试跑同一时刻最多 1 笔未结算请求。
- 8001 本项目服务已重启加载新代码。首次重启继承不到旧进程端口环境，误用默认 8000，与另一应用端口冲突；该失败进程已自行退出，没有触碰另一应用。随后显式 `PORT=8001`、`FINANCE_DISABLE_PAID_API=1` 启动成功。在线核对：`/api/health` 报 `ready=true`、920 份（918 文本、2 仅图像）、付费关闭；OpenAPI 含 `/api/runs/stream`；首页返回新前端 bundle；未登录 POST 该流返回 401。未运行任何新的 Ark 推理。
- 密钥检查：用户本地密钥文件中的原值不出现在本轮拥有的代码差异或新增文件中，`.env` 仍被 Git 忽略。此检查不等于对仓库历史或全部用户自有文件的密钥审计。
- 最终离线回归（代码修订后重新执行）：Python 全套 **356 passed、4 skipped、7 subtests passed**；前端 `run-stream`、`chat-live`、`image-preview`、`health-count` 四项测试及生产构建均通过。独立代码审阅给出了具体问题，已按测试修复模型 ID 核验、预算异常传递、错银行证据兜底、SSE 停机原因、校验真假及视觉页图授权等；审阅代理因用量限制未给出最终批准结论，不能称为独立审阅通过。
- 图谱准备批次已更正为 `output/lightrag_pilot/batch_a9090de62b2a4abd95c605401251c676`：它从当前活动 `output/kb_full/kb_v3_manifest.json` 对应的文本块取 2 个块，manifest 的快照 ID 为 `snap_2e0f216e294bd7baba5709c6a62b691b`，状态仍是 `prepared_only`。前两批引用过时路径，已被此批取代；保留以便审计，均未发送到 LightRAG。
- 修订版 8001 服务已再次重启，显式设置 `PORT=8001`、`FINANCE_DISABLE_PAID_API=1`，核对 `/api/health` 为就绪、920 份（918 文本 + 2 仅图像）、视觉索引 17,957 页、付费关闭；OpenAPI 有 `/api/runs/stream`，首页加载新 bundle，未授权流请求返回 401。日志为 `output/pilot_api_final_2026-09-23.*.log`。本地预算账本复核仍仅一条 `probe/settled/53`、无停止原因；重启及验证**没有新增 Ark 调用**。
- 尚未完成的真实验收：6 道付费链路问题、逐调用可审计的 LightRAG 抽取、人工批准具体页图后的视觉模型调用，以及最终答案准确率评测。用户只使用免费额度，而本次 53-token 探针的资源包抵扣/费用行尚无明确证据，因此继续保持付费关闭；不得把离线测试通过称作上述在线能力已验收。
- 随后用户明确要求开启付费模型并自行在前端试跑。已将忽略的 `.env` 中 `FINANCE_DISABLE_PAID_API` 从 `1` 改为 `0`，仅重启核实归属本项目的 8001 进程。健康接口确认 `paid_api_disabled=false`、920 份文档；本地客户端确认提供方 `ark`、模型 `doubao-seed-2-1-pro-260915`、可用；试跑预算仍为 1,500,000 tokens，已用 53、无预留、未停机。启动与核对均未发起新的模型请求。用户先前免费资源包的费用明细尚未证实，前端真实调用可能消耗额度或产生费用；按用户本次授权开放，但须继续观察控制台。

## 11. 2026-09-24 集成、真实图谱试跑与停机记录

- `scripts/build_budgeted_kg.py` 对活动快照 `snap_2e0f216e294bd7baba5709c6a62b691b` 的 30,406 个文本块提供逐块 SQLite 检查点。开始远程请求前即写 `started`；进程中断后转 `needs_review`，不重试不确定付费调用。抽出的关系必须指向原始块、文档版本和页码，并保留逐字原文证据。`FINANCE_KG_PILOT_ENABLED=1` 才把已完成子集接入 RRF 检索；当前仍保持默认关闭，不声称全量图谱已上线。
- 先用 1 块真实 Ark 调用验证了模型与用量；接着尝试最多 80 个新块，但在第 39 个新块发生 `request_uncertain` 后立即停机。检查点最终 **27 completed、12 no_relations、1 needs_review**，其余活动文本块未抽取。已存 86 条原始关系；加上低质量关系过滤后，在线试验通道可用 **27 条关系／12 个块**。对“工商银行2023年末不良率是多少”本地图通道命中两份工行文档的原文证据和页码。此结果只证明局部可追溯链路，不是答案准确率或全库图谱验收。
- 共享 Ark 台账最终 `spent_tokens=45,668`、`reserved_tokens=0`、`stopped=true`、`stop_reason=request_uncertain`，上限为 1,500,000。最后一笔调用的真实云端消耗未知，**不得按剩余额度自动恢复或创建新账本绕过熔断**。需要核对火山控制台请求/费用明细后，由用户决定是否允许继续；当前前端付费模型虽在配置上开启，实际调用会被本地预算门禁拒绝。
- 新增真实 Tavily Web 搜索适配器及 `web_search` 路由、无关本地证据降级、可点击公开来源链接与不安全 URL 过滤；仅向搜索服务发送用户原问题。旧 Agent `.env` 的 Tavily 值实为占位符，一次真实请求返回 HTTP 401；该占位值已从金融项目 `.env` 移除。**在线 Web 搜索尚未验收**，需要有效 Tavily 密钥，或另行确认火山方舟 Web Search 插件开通及独立计费后再接入。
- 问答运行新增有界排队、持久化 SSE 事件/重连重放、取消、SQLite 图检查点和安全恢复、内部复核工单及管理员工单列表。正在进行的远程模型调用不能从本地强制撤回；取消后丢弃结果，付费中断不自动重放。生产配置、Compose 本机绑定/卷、密钥保护与 Linux 依赖标记已做静态验收；目标主机的完整镜像构建、卷恢复和 HTTPS 仍待实际部署验证。
- `test/merged.xlsx` 是 500 行、400 个唯一题号、485 个唯一问题，仅含 `question_id` 与 `question`，没有标准答案；现存 `output/phase3/gold_review.csv` 的 15 行仍全部为 `pending_review`。因此目前不能报告 500 题答案准确率，人工金标准审核仍是质量验收阻塞项。
- 最终离线验证：Python **415 passed、4 skipped、12 subtests passed**；前端四项脚本检查及生产构建通过。8001 服务仅在无运行中问答时重启，健康接口仍为 920 份（918 文本、2 仅图像），OpenAPI 包含 `/api/runs/stream`、`/api/runs/review-tickets`，首页 200。健康状态与前端新增独立预算停机提示，预算熔断未解除。
- 2026-09-24：用户提供有效 Tavily Key，已仅写入被 Git 忽略的本地 `.env`，未记录明文。一次 Basic 搜索连通性检查返回 5 条公开网页，`tests/unit/test_web_search_agent.py` 9 项通过。仅重启本项目 8001 后端，`/api/health` 报 `ready=true`、920 份文档，首页 200。Ark 预算仍为 `request_uncertain` 熔断；Tavily 接通不代表付费问答链路已恢复，未为本次验收调用 Ark。
- 2026-09-24：继续完善 Tavily 失败可观测性。`web_search` 节点将 401/403、429、超时/不可用映射为不含密钥的明确提示；正常 0 结果仍显示无依据。`/api/health` 新增仅含布尔值的 `web_search_configured`，前端状态栏显示“已配置/未配置/状态未知”，与 Ark 预算暂停分开。测试：Python `420 passed, 4 skipped, 12 subtests passed`，前端五项脚本和生产构建通过。仅重启 8001 本项目后端，健康接口确认 `ready=true`、920 份文档、`web_search_configured=true`、预算仍为 `request_uncertain`。未追加任何真实 Tavily 或 Ark 调用。
- 2026-09-24：用户明确要求解除 `request_uncertain` 熔断并继续发起模型调用。在保留 55 笔已结算调用及 1 笔 `uncertain` 记录的前提下，先备份原账本为 `output/pilot_budget_before_user_resume_2026-09-24.sqlite3`，再仅清除 `state.stopped_reason`；没有重试未确认的图谱块，也没有新建账本。调用现有 `scripts.probe_ark_pilot --send` 一次，Doubao-Seed-2.1-pro 成功结算 53 tokens；账本现为 56 笔 settled、1 笔 uncertain、累计 45,721 tokens、`stopped=false`。8001 健康接口显示预算停止原因为空、付费总开关仍为开、知识库 920 份；当时系统内存约 86%（soft），完整前端问答尚未验收。未确认的云端实际扣量仍需用户在火山控制台核对；原本地 1,500,000-token 上限不变。
- 2026-09-24：用户随后明确要求重试上述未确认的图谱任务。本次只针对 `ck_2a38eed26c45d53362ce2d036a0fee98` 一块：确认它是唯一 `needs_review`，且活动快照、文档版本与原文 SHA-256 均匹配；备份 `output/kg_pilot/checkpoint_before_user_retry_2026-09-24.sqlite3` 和 `output/pilot_budget_before_kg_retry_2026-09-24.sqlite3` 后，手动标记为 `started` 并调用 Ark 一次。返回有效 JSON，但没有通过逐字原文证据约束的关系，检查点变为 `no_relations`、该次结算 **1,221 tokens**；旧预算账本的 `uncertain` 调用记录未删除，也未重试其他块。重试前账本已另有一次前端问答的六个节点结算 **3,937 tokens**，与本次图谱调用分开；重试后累计 **50,879 tokens**、`stopped=false`，图谱检查点为 27 `completed`、13 `no_relations`、0 `needs_review`。这不代表全量图谱完成。

## 12. 2026-09-24 约 300 份文档的有界图谱批次

- 用户确认先做约 300 份，而不是对全部 923 份业务文件跑付费图谱。固定选择活动快照中 **20 份含题库候选证据的较短文本文件 + 280 份其余较短文本文件**，共 300 个文档版本、1,458 个文本块；选择及逐块 ID 固化在 `output/kg_pilot/selection_300.json`。只处理文本块，不把纯图片内容称作已理解；在线知识库维持 920 份，图谱线上开关保持原状态。
- `scripts/build_budgeted_kg.py` 新增整文档选择、固定清单与整文档进度统计。先跑 10 份整文档并核算：到该点所选块 155 个已结算、1 个格式失败、账本累计 180,672 tokens；这一数值含此前图谱与前端用量，本批新增 141 次调用约 10.6 万 tokens，均值约 752 tokens/块。估计余下 1,303 块约需 98 万 tokens；这不是云端免费额度保证，仍由共用 1,500,000-token 本地上限及服务端 402/429 控制。
- 后续发现 5 个 `invalid_response` 块的模型输出均碰到原 `max_tokens=450` 上限，判定截断是强证据；不重试这些已结算失败块。新请求输出上限改为 700；单批允许隔离并跳过至多 2 个格式错误，第 3 个即暂停。`request_uncertain`、402/429、预算停机仍立即停止。当前冻结清单中最多可能有 5 份文档因失败块不能计为整文档完成，最终数字须以检查点实测为准。
- 当前阶段回归：Python 全套 **425 passed、4 skipped、12 subtests passed**。批次仍在推进，最终完整文档数、关系数及预算余额待运行结束后复核；不能把文本块已结算、`no_relations` 或 300 份已选中等同于 300 份图谱关系完整覆盖。
- 最终离线检查（同一活动快照 `snap_2e0f216e294bd7baba5709c6a62b691b`）：冻结的 **300 份、1,458 块**全部有检查点；**294 份整文档完成，6 份部分完成**。所选块中 735 个 `completed`（有原文关系）、716 个 `no_relations`、7 个 `invalid_response`；7 个失败块分布在 6 份文档，均为已结算请求，**没有自动重试**。全量检查点另有历史试跑记录，不能直接把其总计当作这 300 份的统计。
- 来源与证据审计：所选范围原始存储 1,745 条关系，覆盖 236 份文档；所有块的快照、文档版本、来源文件、原文 SHA-256 均与活动文本一致，关系整段证据也在原文中。但 25 条旧试跑关系的 `relation` 字段并未逐字出现在证据段（如“证券代码”为“证券代码为”），因此只把 **1,720 条**计为满足严格三字段逐字证据契约的可用关系，覆盖仍为 236 份文档。`KGPilotRetriever.search` 已增加同一入口过滤，旧检查点原样保留而不进入在线检索；图谱通道仍未启用，原 920 份在线知识库不变。
- 共享预算账本最终 **1,017,875 / 1,500,000 tokens**，`reserved=0`、`stopped=false`；选中块已结算的检查点用量合计 945,079 tokens。共享账本还含前端问答与先前探针，两个数字不可相减来推断云端免费资源包余额。在线健康接口仍 `ready=true`、920 份文档（918 文本 + 2 纯图片）、Tavily 已配置，付费模型开关保持用户原设定。最终 Python 回归 **427 passed、4 skipped、12 subtests passed**。
- 未完成/未声称通过：6 份部分完成文档的 7 个 `invalid_response` 块需人工决定是否针对性重试；图谱没有自动上线，图谱关系的金融语义正确性和问答答案准确率没有人工金标准验收。用户要求“约 300 份”已由 294 份完整处理文档达到，但不是严格的 300/300 完整覆盖，更不是 923 份全库图谱。
- 待人工判定的精确块 ID：`ck_0a82e8c3dd3da14838e3c387ab4b34b1`、`ck_100a5bddb977e20f5f762b7a9f3a8b86`、`ck_2125e8478715eea06d78cb94e026cf79`、`ck_44f83035db1181ec4cf008de7315c362`、`ck_612ce9bb2ed70ac11cb9ca28e34f961f`、`ck_769e7c69debd7c8e0416757581166230`、`ck_f8657ea4af0e14f9e9358632b300e2c1`。未经新授权不得批量重试或清空这些状态。
