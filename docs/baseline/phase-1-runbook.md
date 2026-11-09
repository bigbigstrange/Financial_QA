# 阶段 1：核心契约与 Run 兼容层运行手册

阶段 1 在现有金融问答引擎外增加统一数据契约、模型网关和同步 Run 层。旧 `/api/chat` 仍可使用，并与新 `/api/runs` 共用同一条执行链路。

## 1. 能力边界

本阶段已经启用：

- DocIR 向后兼容字段和统一 Evidence 契约；
- 不可变 ModelSnapshot；
- 普通、JSON、流式模型调用的 ModelGateway；
- `pending → running → completed/failed` Run 状态落库；
- 同一用户范围内的幂等键；
- `POST /api/runs`、`GET /api/runs/{run_id}`；
- `/api/chat` 通过 RunService 执行并返回 `run_id`。

本阶段尚未启用：生产 LangGraph、多 Agent 编排、SSE、取消、审批、断点续跑、文档解析修复、稀疏检索和检索融合升级。

## 2. 初始化或升级本地数据库

`runs` 是新增表，当前开发环境继续使用 SQLAlchemy `create_all()` 的增量建表能力：

```powershell
.\.venv\Scripts\python.exe -c "from app.db.init_db import init_db; init_db(); print('database ready')"
```

该命令只创建缺失表，不删除现有用户、会话或消息。正式切换 PostgreSQL 前应在后续阶段引入 Alembic，当前不要手工改生产表结构。

## 3. 运行验证

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_contracts -v
.\.venv\Scripts\python.exe -m unittest tests.unit.test_model_gateway -v
.\.venv\Scripts\python.exe -m unittest tests.unit.test_run_service -v
.\.venv\Scripts\python.exe -m unittest tests.integration.test_runs_api -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app src scripts tests
git diff --check
```

Agent 参考项目仍执行阶段 0 的离线路由门禁：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
& 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag\.venv\Scripts\python.exe' -m unittest graph.tests.test_graph_routes -v
```

## 4. 启动 API

开发机可先关闭预热，验证鉴权、数据库和接口契约：

```powershell
$env:FINANCE_SKIP_WARMUP='1'
.\.venv\Scripts\python.exe -m app.main
```

服务地址：`http://127.0.0.1:8000`，OpenAPI：`http://127.0.0.1:8000/docs`。

启用真实问答前按阶段 0 手册设置 `KB_PATH`、`CACHE_DIR`、`HF_HOME` 和模型环境变量，然后取消 `FINANCE_SKIP_WARMUP`。

## 5. 调用新旧接口

先登录取得 access token：

```powershell
$login = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/auth/login' `
  -ContentType 'application/json' `
  -Body '{"username":"admin","password":"admin123"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }
```

创建同步 run：

```powershell
$run = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/runs' `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body '{"question":"手机银行证书如何更新？","idempotency_key":"manual-smoke-0001"}'
$run
```

查询本人 run：

```powershell
Invoke-RestMethod -Method Get `
  -Uri "http://127.0.0.1:8000/api/runs/$($run.run_id)" `
  -Headers $headers
```

旧接口继续可用，并在原字段之外返回 `run_id`：

```powershell
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/chat' `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body '{"question":"手机银行证书如何更新？","top_k":12}'
```

## 6. 三题真实连通性回归

沿用阶段 0 的同一知识库、向量缓存、三个问题和 20-block 设置：

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:HF_HOME='C:\code\个人资料\code\Agentic_RAG\model_cache\huggingface'
$env:FINANCE_WEB_DEMO='0'
$env:FINANCE_MAX_BLOCKS='20'
$env:FINANCE_SKIP_IMAGE='1'
$env:FINANCE_SKIP_GEN='1'
$env:FINANCE_EMBED_BATCH_SIZE='16'
```

把新运行结果写为 `output/baseline/phase-1-qa-report.json`。验收标准是三题均无运行错误、三题均返回证据；这只是连通性门禁，不代表答案正确率。

## 7. 对比阶段 0 与阶段 1

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_regression.py `
  --compare-before output\baseline\qa_report.json `
  --compare-after output\baseline\phase-1-qa-report.json `
  --comparison-output output\baseline\phase-1-compatibility.json
```

报告只记录：

- 答案文本是否变化；
- 有序 Evidence ID 是否变化；
- 是否新增错误；
- 每题延迟变化。

文本相同不等于正确，文本变化也不等于退化。所有变化都应结合原始 Evidence 人工复核。

2026-09-19 阶段验收使用同一 20-block 快照执行三题：`error_count=0`、`evidence_response_rate=1.0`、答案变化 0、证据块顺序变化 0。首题包含进程内 BGE 冷启动时间，因此单独记录，不与热请求延迟混为一谈。

## 8. 错误码和 HTTP 状态

| 错误码 | HTTP | 含义 |
|---|---:|---|
| `invalid_request` | 400 | 请求内容不合法 |
| `run_not_found` | 404 | run 不存在或不属于当前用户 |
| `run_conflict` | 409 | 状态或幂等冲突 |
| `inference_busy` | 429 | 推理并发已满 |
| `memory_limit` | 503 | 内存水位拒绝推理 |
| `model_timeout` | 503 | 模型调用超时 |
| `model_unavailable` | 503 | 模型服务不可用 |
| `internal_error` | 500 | 未分类内部错误 |

服务端会保存失败 run；创建接口的错误响应包含 `run_id`，可用于审计。读取其他用户的 run 统一返回 404，避免泄露资源是否存在。

## 9. 回滚

代码回滚到阶段 0 提交后，新增 `runs` 表不会被旧代码读取，可暂时保留。回滚步骤：

1. 停止 API；
2. 切回阶段 0 代码提交；
3. 用阶段 0 环境变量重新启动；
4. 执行阶段 0 的 14 项测试和三题连通性验证。

不要在回滚时直接删除 `runs` 表；其中保存了错误和模型快照审计信息。确认不再需要后再通过受控数据库迁移清理。
