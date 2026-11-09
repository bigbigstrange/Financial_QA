# Phase 1 Core Contracts and Run Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以金融问答仓库为唯一生产入口，建立 DocIR、Evidence、Run、ModelSnapshot 和错误码契约，将生成模型调用收敛到 ModelGateway，并让旧 `/api/chat` 与新 `/api/runs` 走同一条可审计运行链路。

**Architecture:** 本阶段保持现有同步 `qa_runtime.answer()` 行为，增加轻量的 SQLAlchemy Run 记录与 Pydantic 领域契约，不引入任务队列、SSE 或 LangGraph 生产图。`RunService` 在执行前冻结模型快照，调用旧问答引擎，把旧 evidence 映射为统一 Evidence 后原子写入完成或失败状态；`/chat` 继续返回原响应，只把底层推理入口替换为相同的 RunService。模型侧以可注入 transport 的 `ModelGateway` 统一普通、流式和 JSON 调用，旧 `DeepSeekAPIClient` 暂保留为薄兼容适配器。

**Tech Stack:** Python 3.11、Pydantic 2、FastAPI、SQLAlchemy 2、httpx、pytest/unittest（沿用仓库现状）、SQLite 开发环境。

**Spec:** `docs/superpowers/specs/2026-09-19-financial-agentic-rag-integration-design.md`

## Global Constraints

- 以 `finance-qa-platform` 为主仓库；Agent 参考项目本阶段不直接复制到生产入口。
- Python 固定使用 3.11；不得为本阶段新增消息队列、向量数据库或服务拆分。
- 现有 `/api/chat` 请求和响应字段保持兼容，旧问答答案与证据不得因包装层无理由变化。
- 新运行接口仅实现同步 `POST /api/runs` 与归属校验后的 `GET /api/runs/{run_id}`；SSE、取消、审批和断点续跑由后续阶段实现。
- `user_id`、角色、权限范围和模型快照由服务端生成，客户端请求不得覆盖。
- ModelGateway 记录实际 provider/model、耗时和 token 元数据，但日志和异常不得包含 API Key、Authorization 头或完整敏感正文。
- ModelSnapshot 创建后不可变；运行必须绑定显式快照，不能在执行中读取变化后的 `active_profile`。
- Evidence 的原始分数保留在独立字段；未知旧字段进入 `metadata`，不得把 RRF、向量相似度或重排分数包装成“正确率”。
- 所有新行为先写失败测试，再写最小实现；每个任务通过自己的测试后再提交。
- 保留阶段 0 输出和未跟踪的用户面试资料，不得批量 `git add .`。

## Review Focus

- 同一用户重复提交相同 `idempotency_key` 时，应返回原 run 且不再次调用问答引擎；Task 5 覆盖。
- 用户读取他人的 run 时应返回 404，避免泄露 run 是否存在；Task 6 覆盖。
- 旧 evidence 缺少 `block_id`、页码或分数时仍可转换，并显式留下质量标记；Task 2 覆盖。
- 上游 HTTP 返回非 JSON、空 choices、超时或流中断时，ModelGateway 应给出稳定错误码且不泄露凭据；Task 4 覆盖。
- Run 完成结果写库失败时不能对外呈现 `completed`；Task 5 用事务回滚测试覆盖。

---

## File Map

| File | Responsibility |
|---|---|
| `src/contracts.py` | 领域枚举和 Pydantic 契约：ErrorCode、Evidence、ModelSnapshot、RunStatus、RunResult |
| `src/pipeline/docir.py` | 在保留现有构造方式的前提下补齐 DocIR 版本、来源和结构定位字段 |
| `src/models/gateway.py` | 可注入 transport 的模型网关，负责超时、有限重试、普通/流式/JSON 调用和调用元数据 |
| `src/system/llm_api.py` | 旧 DeepSeekAPIClient 薄适配器，内部只调用 ModelGateway |
| `app/models/__init__.py` | 新增持久化 RunRecord，不把大段证据拆表到本阶段 |
| `app/schemas/__init__.py` | 新增 RunCreateRequest、RunOut、ErrorResponse，并给 ChatResponse 增加可选 run_id |
| `app/services/run_service.py` | 模型快照冻结、幂等创建、旧结果映射、状态转换和事务边界 |
| `app/api/runs.py` | 新运行接口和资源归属校验 |
| `app/api/chat.py` | 旧 `/chat` 改为复用 RunService，保持原返回结构 |
| `app/main.py` | 注册 runs router |
| `tests/unit/test_contracts.py` | 契约验证、序列化、兼容性测试 |
| `tests/unit/test_model_gateway.py` | transport、重试、流式、JSON 和脱敏测试 |
| `tests/unit/test_run_service.py` | 状态机、幂等、快照、事务和旧结果映射测试 |
| `tests/integration/test_runs_api.py` | 新接口鉴权、归属和 `/chat` 兼容测试 |
| `docs/baseline/phase-1-runbook.md` | 本阶段启动、迁移、回归和回滚命令 |

### Task 1: Establish Versioned Domain Contracts

**Files:**
- Create: `src/contracts.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_contracts.py`

**Interfaces:**
- Consumes: Pydantic 2 already installed by FastAPI.
- Produces: `ErrorCode`, `ContractError`, `Evidence`, `ModelSnapshot`, `RunStatus`, `RunResult`, and `utc_now()`.

- [ ] **Step 1: Write the failing contract tests**

```python
from pydantic import ValidationError

from src.contracts import Evidence, ModelSnapshot, RunResult, RunStatus


def test_evidence_keeps_score_types_separate():
    item = Evidence(
        evidence_id="ev-1",
        document_id="doc-1",
        document_version="v1",
        block_id="b1",
        text="监管要求",
        page_no=3,
        dense_score=0.8,
        rerank_score=2.4,
        retrieval_channels=["dense", "rerank"],
    )
    assert item.dense_score == 0.8
    assert item.rerank_score == 2.4
    assert "confidence" not in item.model_dump()


def test_model_snapshot_is_immutable():
    snapshot = ModelSnapshot(
        snapshot_id="sha256:abc",
        profile_name="stable-v1",
        provider="deepseek",
        requested_model="deepseek-chat",
        actual_model="deepseek-chat",
    )
    try:
        snapshot.actual_model = "changed"
        raise AssertionError("snapshot must be frozen")
    except ValidationError:
        pass


def test_run_result_rejects_completed_without_answer():
    try:
        RunResult(run_id="r1", status=RunStatus.COMPLETED, answer=None)
        raise AssertionError("completed run requires answer")
    except ValidationError:
        pass
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_contracts -v`

Expected: FAIL because `src.contracts` does not exist.

- [ ] **Step 3: Implement the minimal versioned contracts**

Create `src/contracts.py` with these exact public names and field rules:

```python
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    RUN_NOT_FOUND = "run_not_found"
    RUN_CONFLICT = "run_conflict"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_RESPONSE_INVALID = "model_response_invalid"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_BUSY = "inference_busy"
    MEMORY_LIMIT = "memory_limit"
    INTERNAL_ERROR = "internal_error"


class ContractError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(min_length=1)
    document_id: str | None = None
    document_version: str | None = None
    block_id: str | None = None
    text: str = ""
    source_name: str | None = None
    page_no: int | None = Field(default=None, ge=1)
    structure_path: str | None = None
    bbox: list[float] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    structured_score: float | None = None
    fusion_rank: int | None = Field(default=None, ge=1)
    fusion_score: float | None = None
    rerank_score: float | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    snapshot_id: str
    profile_name: str
    provider: str
    requested_model: str | None = None
    actual_model: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    reranker_model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    parser_config: dict[str, Any] = Field(default_factory=dict)
    evaluation_run_id: int | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = None
    thinking: str | None = None
    model_snapshot: ModelSnapshot | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    latency_sec: float | None = Field(default=None, ge=0)
    memory: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_terminal_payload(self):
        if self.status == RunStatus.COMPLETED and self.answer is None:
            raise ValueError("completed run requires answer")
        if self.status == RunStatus.FAILED and self.error_code is None:
            raise ValueError("failed run requires error_code")
        return self
```

- [ ] **Step 4: Run the contract tests**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_contracts -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract slice**

```powershell
git add src/contracts.py tests/unit/__init__.py tests/unit/test_contracts.py
git commit -m "feat: define core agent contracts"
```

### Task 2: Evolve DocIR Without Breaking Existing Parsers

**Files:**
- Modify: `src/pipeline/docir.py`
- Extend: `tests/unit/test_contracts.py`

**Interfaces:**
- Consumes: existing `DocBlock`, `DocPage`, and `DocIR` constructors used by `src/pipeline/parsers.py`.
- Produces: version-aware DocIR dictionaries and `evidence_from_legacy(raw, position)` returning `Evidence`.

- [ ] **Step 1: Add failing backward-compatibility and legacy evidence tests**

```python
from src.contracts import evidence_from_legacy
from src.pipeline.docir import DocBlock, DocIR, DocPage


def test_existing_docir_constructor_still_serializes():
    doc = DocIR(doc_id="old", source_path="a.pdf", source_format="pdf")
    doc.add_page(DocPage(page_no=1, blocks=[DocBlock("b1", "text", "内容", 1)]))
    data = doc.to_dict()
    assert data["document_id"] == "old"
    assert data["document_version"]
    assert data["pages"][0]["blocks"][0]["block_id"] == "b1"


def test_legacy_evidence_missing_location_is_explicit():
    item = evidence_from_legacy({"text": "内容", "source_doc": "a.pdf"}, position=0)
    assert item.evidence_id
    assert item.page_no is None
    assert "missing_block_id" in item.quality_flags
    assert "missing_location" in item.quality_flags
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_contracts -v`

Expected: FAIL because the new aliases and converter are absent.

- [ ] **Step 3: Add only backward-compatible DocIR fields**

Update the dataclasses so existing positional arguments remain valid. Append optional fields after all current fields:

```python
@dataclass
class DocIR:
    doc_id: str
    source_path: str
    source_format: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pages: List[DocPage] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    document_version: str = "legacy-v1"
    source_hash: Optional[str] = None
    parser_name: Optional[str] = None
    parser_version: Optional[str] = None
    language: Optional[str] = None
    access_scope: List[str] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        return self.doc_id
```

In `to_dict()`, add `document_id` while retaining `doc_id`; do not rename persisted legacy keys in this phase. Add optional `section_path`, `parent_id`, `caption`, `table_headers`, `table_rows`, `unit`, `ocr_used`, and `quality_flags` to `DocBlock`, after the existing fields.

- [ ] **Step 4: Implement deterministic legacy Evidence mapping**

Add `evidence_from_legacy()` to `src/contracts.py`. Derive `evidence_id` from a SHA-256 hash of canonical JSON containing `source_doc`, `block_id`, `page`, text, and position. Map legacy `score` to `metadata["legacy_score"]`, not to a calibrated score field. Mark missing IDs/locations in `quality_flags` and preserve all unrecognized keys in `metadata["legacy"]`.

- [ ] **Step 5: Run parser and contract regression tests**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_contracts tests.test_audit_parser_baseline -v`

Expected: PASS, including old constructor coverage.

- [ ] **Step 6: Commit the DocIR compatibility slice**

```powershell
git add src/pipeline/docir.py src/contracts.py tests/unit/test_contracts.py
git commit -m "feat: version document and evidence contracts"
```

### Task 3: Freeze Model Profiles Into Immutable Snapshots

**Files:**
- Modify: `app/services/model_registry.py`
- Modify: `configs/model_registry.json`
- Create: `tests/unit/test_model_registry_snapshot.py`

**Interfaces:**
- Consumes: `ModelRegistry.get_profile(name)` and `ModelSnapshot`.
- Produces: `ModelRegistry.snapshot(name=None, evaluation_run_id=None) -> ModelSnapshot`.

- [ ] **Step 1: Write the failing snapshot tests**

Use a temporary registry JSON with a `stable-v1` profile. Assert that:

```python
snapshot = registry.snapshot("stable-v1", evaluation_run_id=7)
assert snapshot.profile_name == "stable-v1"
assert snapshot.embedding_model == "BAAI/bge-m3"
assert snapshot.evaluation_run_id == 7
assert snapshot.snapshot_id.startswith("sha256:")
assert registry.snapshot("stable-v1").snapshot_id == registry.snapshot("stable-v1").snapshot_id
```

Also mutate the registry after the first snapshot and assert that the already returned object stays unchanged.

- [ ] **Step 2: Run the test and confirm missing method failure**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_model_registry_snapshot -v`

Expected: FAIL with `ModelRegistry has no attribute snapshot`.

- [ ] **Step 3: Add explicit profile fields to the registry**

Each profile in `configs/model_registry.json` must include:

```json
{
  "provider": "deepseek",
  "requested_model": "deepseek-v4-flash",
  "generation_parameters": {"temperature": 0.2, "max_tokens": 1024},
  "embedding_model": "BAAI/bge-m3",
  "reranker_model": null,
  "prompt_versions": {"answer": "legacy-v1"},
  "retrieval_config": {"top_k": 12},
  "parser_config": {"docir": "legacy-v1"}
}
```

Keep the old `embed_model`, `gen_model`, and other keys while runtime compatibility still depends on them.

- [ ] **Step 4: Implement canonical hashing and frozen snapshot creation**

Serialize only snapshot fields with `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, hash the UTF-8 bytes, prefix with `sha256:`, and construct `ModelSnapshot`. The method must copy nested dictionaries before returning.

- [ ] **Step 5: Run registry and contract tests**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_model_registry_snapshot tests.unit.test_contracts -v`

Expected: PASS.

- [ ] **Step 6: Commit the immutable snapshot slice**

```powershell
git add app/services/model_registry.py configs/model_registry.json tests/unit/test_model_registry_snapshot.py
git commit -m "feat: freeze model profile snapshots"
```

### Task 4: Introduce ModelGateway and Preserve the DeepSeek Adapter

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/gateway.py`
- Modify: `src/system/llm_api.py`
- Create: `tests/unit/test_model_gateway.py`

**Interfaces:**
- Consumes: `ModelSnapshot`, `ContractError`, `ErrorCode`, httpx.
- Produces: `ModelRequest`, `ModelResponse`, `ModelChunk`, `ModelTransport`, `OpenAICompatibleTransport`, and `ModelGateway`.
- Compatibility: `DeepSeekAPIClient.chat()` and `.rerank_indices()` keep their current signatures.

- [ ] **Step 1: Write transport-free gateway tests**

Create a fake transport that records requests and returns deterministic responses. Tests must cover:

```python
response = gateway.generate(
    messages=[{"role": "user", "content": "问题"}],
    temperature=0.0,
    max_tokens=32,
)
assert response.text == "答案"
assert response.provider == "fake"
assert response.actual_model == "fake-1"
assert response.latency_ms >= 0
```

Add tests for one retry on `TimeoutError`, no retry on invalid request, JSON parsing from fenced text, streamed chunk order, interruption mapped to `MODEL_UNAVAILABLE`, and `repr(error)` not containing the test API key.

- [ ] **Step 2: Run tests and verify missing gateway failure**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_model_gateway -v`

Expected: FAIL because `src.models.gateway` does not exist.

- [ ] **Step 3: Implement the transport boundary**

Define `ModelTransport` as a Protocol with:

```python
def complete(self, request: ModelRequest) -> ModelResponse: ...
def stream(self, request: ModelRequest) -> Iterator[ModelChunk]: ...
```

`ModelRequest` carries messages, requested model, temperature, max tokens, response format and request ID. `ModelResponse` carries text, requested/actual model, provider, prompt/completion token counts, latency and request ID. `ModelGateway` accepts a transport and `max_retries=1`; retries only timeout, HTTP 429 and HTTP 5xx with bounded backoff. Tests inject `sleep=lambda _: None`.

- [ ] **Step 4: Implement the OpenAI-compatible DeepSeek transport**

Use one injected `httpx.Client` or client factory. Set Authorization at request time, but never include headers or response bodies in exception messages. Validate `choices[0].message.content`; read provider usage when present. For streaming, parse `data:` lines, stop on `[DONE]`, ignore keep-alives, and close the response on generator exit.

- [ ] **Step 5: Convert DeepSeekAPIClient to a thin adapter**

Keep constructor and public methods. Construct `OpenAICompatibleTransport` plus `ModelGateway`; `chat()` returns `gateway.generate(...).text`. Keep `_parse_index_list()` unchanged so rerank behavior remains stable. Remove direct `httpx.Client` creation from `DeepSeekAPIClient`.

- [ ] **Step 6: Run focused and existing regression tests**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_model_gateway tests.test_benchmark_regression -v`

Expected: PASS.

- [ ] **Step 7: Verify no business module creates vendor clients directly**

Run: `rg -n "httpx\.Client|ChatOpenAI|OpenAI\(" app src`

Expected: only `src/models/gateway.py` owns the HTTP client; compatibility modules may import `ModelGateway` but may not create an HTTP client.

- [ ] **Step 8: Commit the gateway slice**

```powershell
git add src/models src/system/llm_api.py tests/unit/test_model_gateway.py
git commit -m "feat: centralize model calls in gateway"
```

### Task 5: Persist and Execute Synchronous Runs

**Files:**
- Modify: `app/models/__init__.py`
- Modify: `app/schemas/__init__.py`
- Create: `app/services/run_service.py`
- Create: `tests/unit/test_run_service.py`

**Interfaces:**
- Consumes: `qa_runtime.answer(question, question_id)`, `model_registry.snapshot()`, `evidence_from_legacy()`.
- Produces: `RunRecord`, `RunCreateRequest`, `RunOut`, and `RunService.create_and_execute(db, user_id, request) -> RunResult`.

- [ ] **Step 1: Write failing RunService tests with a temporary SQLite database**

Use `StaticPool` and `check_same_thread=False`. Inject a fake QA runtime and fake registry. Cover:

```python
result = service.create_and_execute(db, user_id=1, request=request)
assert result.status == RunStatus.COMPLETED
assert result.answer == "旧答案"
assert fake_qa.calls == 1

same = service.create_and_execute(db, user_id=1, request=request)
assert same.run_id == result.run_id
assert fake_qa.calls == 1
```

Add tests that a different user may reuse the same idempotency key without collision, MemoryError becomes `memory_limit`, busy RuntimeError becomes `inference_busy`, unexpected exceptions become `internal_error`, snapshot JSON is frozen before QA execution, and a commit failure rolls the transaction back without returning a completed result.

- [ ] **Step 2: Run tests and verify missing service failure**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_run_service -v`

Expected: FAIL because `RunService` and `RunRecord` do not exist.

- [ ] **Step 3: Add the minimal RunRecord table**

Use a UUID string primary key and these fields: `id`, `user_id`, `conversation_id`, `idempotency_key`, `status`, `question`, `answer`, `evidence_json`, `confidence`, `thinking`, `model_snapshot_json`, `error_code`, `error_message`, `latency_sec`, `created_at`, `started_at`, `finished_at`. Add a unique constraint on `(user_id, idempotency_key)` and indexes on `user_id`, `status`, and `created_at`.

- [ ] **Step 4: Add API schemas without mutable defaults**

```python
class RunCreateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=30)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    show_thinking: bool = False


class RunOut(BaseModel):
    run_id: str
    status: RunStatus
    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = None
    model_snapshot: ModelSnapshot | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    latency_sec: float | None = None
```

Change existing list/dict schema defaults in the touched file to `Field(default_factory=...)` to prevent cross-request mutation.

- [ ] **Step 5: Implement state transitions and transaction boundaries**

`create_and_execute()` must:

1. normalize and validate the question;
2. return an existing record for `(user_id, idempotency_key)`;
3. freeze the model snapshot;
4. insert `pending`, commit, then set `running` and commit;
5. call QA with `question_id=run_id`;
6. convert at most eight legacy evidence items;
7. write result fields and set `completed` in the same commit;
8. on a mapped failure, roll back pending writes, reload the run, set `failed` plus stable error code, and commit;
9. never store `_memory` inside evidence or snapshot JSON.

Expose `get_owned(db, run_id, user_id)` and `to_result(record, include_thinking=False)`.

- [ ] **Step 6: Run service tests and database initialization smoke test**

Run: `.venv\Scripts\python.exe -m unittest tests.unit.test_run_service -v`

Run: `.venv\Scripts\python.exe -c "from app.db.init_db import init_db; init_db(); print('ok')"`

Expected: all tests PASS and initialization prints `ok`.

- [ ] **Step 7: Commit the run persistence slice**

```powershell
git add app/models/__init__.py app/schemas/__init__.py app/services/run_service.py tests/unit/test_run_service.py
git commit -m "feat: persist synchronous agent runs"
```

### Task 6: Add Run API and Route Legacy Chat Through It

**Files:**
- Create: `app/api/runs.py`
- Modify: `app/api/chat.py`
- Modify: `app/main.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_runs_api.py`

**Interfaces:**
- Consumes: `RunService`, current auth dependency, current conversation/message tables.
- Produces: authenticated `POST /api/runs`, `GET /api/runs/{run_id}`, and backward-compatible `POST /api/chat` with optional `run_id`.

- [ ] **Step 1: Write failing API integration tests**

Build a test FastAPI app with dependency overrides for DB and current user, and inject a fake RunService. Assert:

- `POST /api/runs` returns 200 with completed run, evidence, and model snapshot.
- blank question returns 422 without invoking the service.
- `GET /api/runs/{id}` returns the owner record.
- requesting another user's run returns 404, not 403.
- repeated idempotency key returns the same `run_id`.
- `/api/chat` still contains `conversation_id`, `user_message`, `assistant_message`, `elapsed_sec`, `memory`, and `profile`, and now may include `run_id`.
- `/api/chat` persists exactly one user and one assistant message while invoking RunService exactly once.

- [ ] **Step 2: Run API tests and verify route failures**

Run: `.venv\Scripts\python.exe -m unittest tests.integration.test_runs_api -v`

Expected: FAIL because the runs router is absent and chat still calls `qa_runtime` directly.

- [ ] **Step 3: Implement the runs router**

```python
router = APIRouter(prefix="/runs", tags=["runs"])

@router.post("", response_model=RunOut)
def create_run(...):
    result = run_service.create_and_execute(db, user_id=user.id, request=body)
    return RunOut.model_validate(result.model_dump())

@router.get("/{run_id}", response_model=RunOut)
def get_run(...):
    record = run_service.get_owned(db, run_id=run_id, user_id=user.id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "运行不存在"})
    return RunOut.model_validate(run_service.to_result(record).model_dump())
```

Map `invalid_request` to 400, `inference_busy` to 429, memory/model unavailable errors to 503, and unexpected errors to 500. Return stable code and safe message in `detail`.

- [ ] **Step 4: Refactor `/chat` to call RunService once**

Keep conversation ownership checks and message persistence. Replace `qa_runtime.answer()` with `RunCreateRequest` plus `run_service.create_and_execute()`. Serialize `Evidence.model_dump()` into the assistant message, map model profile from `result.model_snapshot.profile_name`, and add `run_id: str | None = None` to `ChatResponse`. Do not expose thinking unless `show_thinking` is true.

- [ ] **Step 5: Register the router and run API tests**

Add `runs` to `from app.api import ...` and call `app.include_router(runs.router, prefix=settings.api_prefix)`.

Run: `.venv\Scripts\python.exe -m unittest tests.integration.test_runs_api -v`

Expected: PASS.

- [ ] **Step 6: Run the full Python suite**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all phase 0 and phase 1 tests PASS.

- [ ] **Step 7: Commit the API compatibility slice**

```powershell
git add app/api/runs.py app/api/chat.py app/main.py app/schemas/__init__.py tests/integration
git commit -m "feat: add run API behind legacy chat"
```

### Task 7: Prove Baseline Compatibility and Document Operations

**Files:**
- Create: `docs/baseline/phase-1-runbook.md`
- Modify: `README.md`
- Modify: `scripts/benchmark_regression.py`
- Extend: `tests/test_benchmark_regression.py`

**Interfaces:**
- Consumes: stage 0 `output/baseline/qa_report.json` and the new run API.
- Produces: a comparison mode that reports answer/evidence changes without labeling them as correctness, plus operator runbook.

- [ ] **Step 1: Write a failing comparison test**

Add a pure `compare_reports(before, after)` test using two three-case fixtures. Assert it reports:

```python
assert summary["case_count"] == 3
assert summary["answer_changed_count"] == 1
assert summary["evidence_id_changed_count"] == 1
assert summary["new_error_count"] == 0
assert "accuracy" not in summary
```

The function must reject mismatched question IDs rather than compare by list position.

- [ ] **Step 2: Run the test and verify missing function failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_benchmark_regression -v`

Expected: FAIL because `compare_reports` is absent.

- [ ] **Step 3: Implement report comparison and CLI flags**

Add `--compare-before PATH` and `--compare-after PATH`. Compare normalized answer text, ordered evidence IDs, error presence and latency deltas. Write comparison JSON under `output/baseline/phase-1-compatibility.json`. Do not infer semantic correctness from equality.

- [ ] **Step 4: Write the runbook**

Document exact commands for:

1. creating/upgrading the SQLite table with `init_db()`;
2. running focused and full tests;
3. starting the API with model warmup disabled;
4. authenticating and calling `/api/runs` and `/api/chat`;
5. running the same three-case connectivity smoke used in phase 0;
6. comparing phase 0 and phase 1 reports;
7. rolling back the code commit while leaving additive `runs` table data harmless;
8. stating that Agent graph, SSE, cancellation, approvals, document fixes and hybrid retrieval are not enabled in phase 1.

- [ ] **Step 5: Link the runbook from README**

Add a “阶段 1” link beside the existing architecture and phase 0 links.

- [ ] **Step 6: Run complete verification**

Run:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q app src scripts tests
git diff --check
git status --short
```

Then run the Agent reference offline gate unchanged:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
& 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag\.venv\Scripts\python.exe' -m unittest graph.tests.test_graph_routes -v
```

Expected: all commands exit 0; `git status` lists only intended phase 1 changes plus the pre-existing unrelated user files.

- [ ] **Step 7: Run the real compatibility smoke**

Use the existing external Hugging Face cache and 20-block settings from the phase 0 runbook. Execute the same three questions through `RunService`, store `output/baseline/phase-1-qa-report.json`, and compare it with the phase 0 report. Acceptance requires zero new runtime errors and evidence-bearing responses for all three cases. Any answer/evidence change must be listed for human review; equality is not called accuracy.

- [ ] **Step 8: Commit documentation and verification support**

```powershell
git add README.md docs/baseline/phase-1-runbook.md scripts/benchmark_regression.py tests/test_benchmark_regression.py
git commit -m "docs: add phase one compatibility gate"
```

## Phase 1 Acceptance Checklist

- [ ] `POST /api/runs` executes the current finance QA path and returns a persisted run ID.
- [ ] `GET /api/runs/{run_id}` enforces owner scope and returns a stable contract.
- [ ] `/api/chat` uses the same RunService and preserves its prior response fields.
- [ ] Every run stores an immutable model snapshot captured before inference.
- [ ] Existing model calls reach the provider through ModelGateway.
- [ ] DocIR legacy constructors and serialized keys remain readable.
- [ ] Evidence mapping preserves source text and raw metadata while making missing provenance explicit.
- [ ] Idempotency prevents duplicate execution for the same user and key.
- [ ] Stable error codes distinguish invalid input, busy inference, memory pressure, model failure and internal failure.
- [ ] Phase 0 tests, Agent offline route tests, compile checks and phase 1 tests pass.
- [ ] Three-case real smoke has zero new errors and its changes are recorded without unsupported accuracy claims.
- [ ] Unrelated user documents and tools remain untracked and unmodified.

## Deferred Deliberately

The following belong to later architecture phases and are intentionally absent from this plan: production LangGraph migration, planner/executor/verifier nodes, SSE event replay, cancellation propagation, approval workflows, PostgreSQL migration tooling, Redis queues, document parser repairs, stable block ID generation, sparse retrieval, RRF/DBSF fusion, reranker migration, ACL filtering, prompt registry, automated LLM evaluation and Kubernetes deployment.
