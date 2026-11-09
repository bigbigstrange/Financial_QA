# 阶段 2：文档版本与证据链路运行手册

> 状态：核心链路已实现并通过自动化测试；全量语料迁移与激活待实施
> 日期：2026-09-20

## 1. 环境

阶段 2 使用真实 Python 3.11 虚拟环境，替代错误的 `.conda`（Python 3.10）：

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

当前验证环境：CPython 3.11.13（uv 管理），全部锁定依赖已安装。

## 2. 自动化验证

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m compileall -q app src scripts tests
```

最近一次结果：103 个测试全部通过，编译检查无语法错误。首次运行前还会包含阶段 0/1 的 48 个旧测试，全部保持通过。

## 3. 已落地模块

### 身份与 DocIR v2

- `src/pipeline/identity.py`：`ParseContext`、`make_document_version`、`make_block_id`。
- `src/pipeline/docir.py`：schema v2、可空页号、`structural_locator`、`validate_v2()`、原子 JSON 读写。
- `src/pipeline/tables.py`：表头继承/扁平化、单位、父表、行组切块与确定性检索文本。

### 解析器

- Word：OOXML body 单次遍历，段落-表格原始顺序，`page=None`，嵌套表格进单元格定位。
- PDF：真实页码、一位小数 bbox、结构化 locator、OCR 阈值与表格文本去重。
- Excel：工作表行定位、行组、截断标记。
- PPT：幻灯片号与 shape 定位。
- 文本、图片、旧 `.doc/.rtf`：可空页号 + `legacy_identity/legacy_layout` 标记。

### 质量与文件安全

- `src/pipeline/quality.py`：文档/快照质量报告、硬失败门禁、`process_document()`。
- `src/pipeline/file_validation.py`：大小、签名、OOXML 成员、ZIP 解压上限、宏文件拒绝。

### 目录、快照与可见性

- `app/models`：`DocumentRecord`、`DocumentVersionRecord`、`KnowledgeSnapshotRecord`、`EvidenceRecord`，`RunRecord.knowledge_snapshot_id`。
- `app/db/schema_upgrades.py`：幂等增量迁移。
- `app/services/document_catalog.py`：版本去重、状态机、激活/删除/回滚。
- `app/services/document_service.py`：上传→解析→质量门禁→不可变 manifest→原子激活。
- `src/pipeline/snapshots.py`：不可变 SnapshotManifest 与原子写入。
- `src/system/retrieval.py`：manifest 快照加载；各检索通道支持可见性过滤。
- `src/system/index_fingerprint.py`：缓存兼容指纹契约。

### API

- `POST /api/documents`、`POST /api/documents/{id}/versions`、`GET /api/documents/{id}`、`DELETE /api/documents/{id}`、`POST /api/documents/{id}/rollback/{version_id}`。
- `GET /api/evidence/{evidence_id}`：仅所有者或管理员可读，身份不匹配返回稳定 409。

### Run 固定快照

- `RunService` 在推理前持久化 `knowledge_snapshot_id`。
- v2 Evidence 使用 `ev_<sha256(snapshot|version|block)>` 身份。
- 完成事务同时写入 `EvidenceRecord`。

## 4. 最小演示流程

```powershell
# 1) 启动数据库升级
& '.\.venv\Scripts\python.exe' -c "from app.db.init_db import init_db; init_db(); print('ready')"

# 2) 启动 API（先跳过模型预热只验证文档 API 也可）
$env:FINANCE_SKIP_WARMUP='1'
& '.\.venv\Scripts\python.exe' -m app.main

# 3) 管理员上传
$login = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/auth/login' `
  -ContentType 'application/json' `
  -Body '{"username":"admin","password":"admin123"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }
$body = Get-Content -LiteralPath 'D:\tmp\rules.txt' -Raw
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/documents' `
  -Headers $headers -Form @{ file = Get-Item 'D:\tmp\rules.txt' }
```

## 5. 尚未完成 / 下一步

- 全量语料候选迁移：`scripts/migrate_phase2_knowledge.py` 已提供候选构建与 `--activate` 门禁，但尚未在 924 个原始文件上跑完整候选和回归。
- 将 `IndexFingerprint` 真正接入 FAISS 缓存元数据读写（当前只有契约与测试）。
- `scripts/preprocess.py` / `scripts/audit_parser_baseline.py` 尚未改造成 v2 聚合报告。
- 固定阶段 0 问题集的 v2 回归对比与解释字段。
- Agent 参考项目的 `graph/tests` 回归。
- 全量迁移通过后，再执行删除/回滚/历史证据的真实烟测和手工验收。

候选迁移示例（默认不切换 active）：

```powershell
& '.\.venv\Scripts\python.exe' scripts/migrate_phase2_knowledge.py `
  --input 'data/documents' `
  --candidate-dir 'output/phase2' `
  --database-url 'sqlite:///C:/code/个人资料/code/Agentic_RAG/finance-qa-platform/data_app.db'
```

旧 `.conda` 环境仍保留，但不应再作为阶段 2 开发环境；任务和手册中残留的 `.conda\python.exe` 命令应统一替换为 `.venv\Scripts\python.exe`。
