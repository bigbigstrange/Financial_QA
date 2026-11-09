# Doubao Bounded Full-Chain Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely connect the authorized Doubao-Seed-2.1-pro model and demonstrate the existing financial RAG's complete chain on a small, traceable sample without running paid full-corpus extraction.

**Architecture:** Reuse the existing OpenAI-compatible transport and LangGraph; place a single metered, fail-closed call boundary between all online Agent calls and Ark. Keep the active 920-document KB untouched. Run LightRAG extraction in an isolated pilot directory only after its separate metering is proved; use selected image pages for genuine VLM input; expose node progress via authenticated SSE.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, SQLite stdlib, httpx, pytest, Vue 3/TypeScript, existing LightRAG server and local retrieval indexes.

**Spec:** `docs/superpowers/specs/2026-09-23-doubao-bounded-full-chain-pilot-design.md`.

## Global Constraints

- Project root: `C:\code\个人资料\code\Agentic_RAG\finance-qa-platform`; use `.venv\Scripts\python.exe` for Python commands.
- Keep `FINANCE_DISABLE_PAID_API=1` in persistent `.env`; run the smallest Ark probe with a one-process override only after every local budget test passes. Do not silently clear this flag in persistent config.
- Use `doubao-seed-2-1-pro-260915` at `https://ark.cn-beijing.volces.com/api/v3` only; do not relabel Ark results as DeepSeek or switch to another paid model automatically.
- This pilot's cumulative target ceiling is 1,500,000 tokens, with approximately 1,000,000 tokens reserved from the user-reported 2,499,950-token resource pack. Local enforcement cannot guarantee cloud-side zero overage; check console after every small paid batch.
- Never print, log, commit, return to browser, or put in tests the key from the user-supplied file. Test only non-sensitive excerpts; no whole raw business document uploads.
- Preserve the 920-document active KB, prior LightRAG data, dirty worktree and user files. Never use LightRAG `--clear`; no full-corpus paid extraction or `web_search`.
- Stop paid work on HTTP 402/429, missing `usage`, model mismatch, unexpected resource-pack deduction, untraceable evidence, or exhausted budget; report exact stopping point. Do not count an unverified fallback as a passed double check.

## Review Focus

1. Ark returns text but omits `usage`: the request must be recorded as uncertain and the next paid request refused (Task 2).
2. Ark returns a different model ID: response must not be accepted or silently charged to the Pro pilot (Task 1).
3. HTTP 402 or 429: no retry, no switch to another paid provider, visible stop reason (Tasks 1–2).
4. “掌上银行” is an App name, not a bank name: query rewriting and evidence gating must not invent a bank (Task 3).
5. Image-only evidence has no readable text: show a visual citation only after actual image-model invocation; otherwise label it unverified (Task 6).

## File map

- `src/models/gateway.py`: normalize Ark `usage` and actual model; the transport remains the only HTTP boundary.
- `src/system/llm_api.py`: select Ark or existing DeepSeek without false provider attribution; retain paid disable switch.
- `src/system/pilot_budget.py` (new): SQLite token reservations and redacted usage ledger, shared by all online calls.
- `src/system/agent.py`, `src/agent/nodes.py`, `src/agent/prompts.py`, `app/services/qa_runtime.py`: preserve query entity, reject wrong-source evidence, propagate real verification and budget-stop state.
- `scripts/probe_ark_pilot.py` (new): one tiny non-sensitive call, no automatic batch or key display.
- `scripts/run_kg_pilot.py` (new): create isolated, chunk-whitelisted LightRAG input and audit source mapping; never touch the old graph.
- `src/agent/visual_answer.py` (new): resolve and limit trusted page images for model input; `nodes.py` uses it only for a selected visual case.
- `app/api/runs.py`, `app/services/run_service.py`, `app/services/qa_runtime.py`, `src/agent/graph.py`, `frontend/src/api/client.ts`, `frontend/src/views/ChatView.vue`: authenticated live progress stream and honest verification display.
- `tests/unit/` and `tests/integration/`: isolated fake transport, temporary SQLite and fake retrieval; real paid calls are manual gates only.

## Task 1: Ark provider and response contract

**Files:** Modify `src/models/gateway.py`, `src/system/llm_api.py`, `app/services/qa_runtime.py`; test `tests/unit/test_ark_client.py` (new).

**Interfaces:** `DeepSeekAPIClient(...).chat(...) -> str` remains compatible with existing callers. `ModelResponse.total_tokens: int | None` is added. Ark is selected by `FINANCE_LLM_PROVIDER=ark`, `ARK_API_KEY`, `ARK_BASE_URL`, and `ARK_MODEL`; the DeepSeek path remains unchanged.

- [ ] **Step 1: Write failing offline tests.** Use `httpx.MockTransport` to assert `ModelResponse(provider="ark", actual_model="doubao-seed-2-1-pro-260915", prompt_tokens=10, completion_tokens=2, total_tokens=12)` and to reject a different `data.model`. Assert `repr(client.gateway.transport)` never contains the fake key. Assert 402 and 429 raise non-retryable `ContractError` without a second request.

  ```python
  def test_ark_usage_and_model(monkeypatch):
      monkeypatch.setenv("FINANCE_LLM_PROVIDER", "ark")
      monkeypatch.setenv("ARK_API_KEY", "fake-test-key")
      client = DeepSeekAPIClient()
      assert client.gateway.snapshot.provider == "ark"
      assert client.model == "doubao-seed-2-1-pro-260915"
      assert "fake-test-key" not in repr(client.gateway.transport)
      def reply(request):
          return httpx.Response(200, json={"model": client.model,
              "choices": [{"message": {"content": "连接正常"}}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
      client.gateway.transport._client_factory = lambda: httpx.Client(
          transport=httpx.MockTransport(reply))
      assert client.gateway.generate([{"role": "user", "content": "测试"}]).total_tokens == 12
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_ark_client.py -q`; expect failures for absent Ark selection/total tokens.
- [ ] **Step 3: Implement minimal adapter.** Add `total_tokens=_optional_int(usage.get("total_tokens"))` in `OpenAICompatibleTransport.complete`, add `total_tokens: int | None = None` to `ModelResponse`, select provider/URL/model/key in `DeepSeekAPIClient.__init__`, set `ModelSnapshot.provider` and transport provider to `ark`, and use `ModelGateway(..., max_retries=0)` for Ark. In `chat`, reject `response.actual_model != self.model` before returning text. Make HTTP 429 non-retryable in pilot mode; do not expose response body or key in errors. Update `QARuntime._settings_demo_env` so an Ark key enables the API backend without setting a DeepSeek model.

  ```python
  provider = "ark" if os.getenv("FINANCE_LLM_PROVIDER") == "ark" else "deepseek"
  if provider == "ark":
      key = os.getenv("ARK_API_KEY", "")
      url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
      model = os.getenv("ARK_MODEL", "doubao-seed-2-1-pro-260915")
  response = self.gateway.generate(messages, max_tokens=max_tokens,
                                   temperature=temperature, extra_body=extra_body)
  if provider == "ark" and response.actual_model != model:
      raise ContractError(ErrorCode.MODEL_RESPONSE_INVALID, "返回模型与试跑模型不符")
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_ark_client.py tests/unit/test_local_mode.py -q`; expect all pass, then commit only Task 1 files with `feat: support Ark model provenance`.

## Task 2: One fail-closed token ledger for online calls

**Files:** Create `src/system/pilot_budget.py`; modify `src/system/llm_api.py`; test `tests/unit/test_pilot_budget.py` (new).

**Interfaces:** `class PilotBudgetStopped(RuntimeError)`; `PilotBudget(path: Path, limit: int = 1_500_000).reserve(request_id: str, tokens: int) -> None`, `.settle(request_id: str, response: ModelResponse, node: str) -> None`, `.stop(request_id: str, reason: str) -> None`, `.summary() -> dict`. `DeepSeekAPIClient.chat(..., node: str = "chat") -> str` records each Ark request; existing callers still work.

- [ ] **Step 1: Write failing tests.** Use `tmp_path` SQLite and two `PilotBudget` objects: only one reservation succeeds when their total exceeds 1,500,000; `settle` rejects `total_tokens=None` and a mismatched model; no prompt/key is present in the DB; a failed call releases its reservation but marks the pilot stopped on 402/429 or uncertain usage. Exercise a 1,500,001-token request rejection before transport invocation.

  ```python
  def test_atomic_limit(tmp_path):
      path = tmp_path / "pilot.sqlite3"
      first, second = PilotBudget(path), PilotBudget(path)
      first.reserve("a", 900_000)
      with pytest.raises(PilotBudgetStopped):
          second.reserve("b", 600_001)
      assert second.summary()["reserved_tokens"] == 900_000
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_pilot_budget.py -q`; expect failures before implementation.
- [ ] **Step 3: Implement the ledger.** Use SQLite `BEGIN IMMEDIATE` around reservation and settlement; store request ID, provider, model, node, reserved, input/output/total, status and timestamp only. Pre-reserve `2 * len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + max_tokens` tokens; do not reserve image requests through this text estimator. Refuse if spent plus outstanding reservations plus requested amount exceeds limit, or any prior entry is `uncertain`/`stopped`. Defer retries; Ark call wrapper reserves once, settles once, and on missing `usage` or HTTP 402/429 records stop and raises. Never publish budget details containing messages.

  ```python
  with sqlite3.connect(self.path, timeout=30) as db:
      db.execute("BEGIN IMMEDIATE")
      spent, pending = self._totals(db)
      if spent + pending + tokens > self.limit or self._is_stopped(db):
          raise PilotBudgetStopped("试跑预算不足或已暂停")
      db.execute("INSERT INTO calls(request_id,reserved,status) VALUES(?,?,'reserved')",
                 (request_id, tokens))
      db.commit()
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_pilot_budget.py tests/unit/test_ark_client.py -q`; expect all pass, then commit only Task 2 files with `feat: cap and record Ark pilot usage`.

## Task 3: Evidence identity and real double-check semantics

**Files:** Modify `src/system/agent.py`, `src/agent/nodes.py`, `src/agent/prompts.py`, `app/services/qa_runtime.py`; test `tests/unit/test_agent_graph.py` and new `tests/unit/test_pilot_qa_guard.py`.

**Interfaces:** `QueryPlanner.plan(question) -> list[str]` keeps the named App in both subquestions. `AgentNodes` exposes `verified=False` unless both checks completed successfully; a budget-stop exception never becomes extractive “verified” output.

- [ ] **Step 1: Write failing tests.** Assert both subquestions for the sample 掌上银行/基金 question contain `掌上银行`; fake retrieval returns only 交通银行企业网银 and generic slides, and the graph returns evidence-insufficient without a bank-specific procedure. Fake first verifier success + second verifier exception must yield `verified=False`. A fake budget-stop in any LLM node must abort paid work and be visible in the run result; no retry loop.

  ```python
  def test_mobile_app_identity_survives_plan():
      question = "请说明掌上银行中基金功能的服务内容以及基金认购流程"
      assert all("掌上银行" in q for q in QueryPlanner().plan(question))

  def test_second_check_failure_is_not_verified():
      class BrokenLLM:
          enabled = True
          def chat(self, *args, **kwargs):
              raise RuntimeError("offline")
      nodes = AgentNodes(retriever=object(), llm=BrokenLLM(),
                         reranker=object(), planner=QueryPlanner(), log=lambda _: None)
      state = initial_state("基金认购")
      state.update({"generation": "答案", "grounded_verified": True,
                    "documents": [{"chunk_id": "c1", "content": "基金认购"}]})
      assert nodes.check_answer(state)["verified"] is False
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_pilot_qa_guard.py -q`; expect failures.
- [ ] **Step 3: Implement targeted guards.** Remove hardcoded bank/app-unaware second subquery. Before `generate`, require evidence to support the named product/channel and answer each requested part; otherwise return a short evidence-insufficient response with citations only to genuinely relevant material. Track two separate check results (`grounded_verified`, `answer_verified`) and set final `verified` only when both are true. Do not catch `PilotBudgetStopped` under broad fallback handlers. Keep finite graph retry limits and no web search.

  ```python
  if "掌上银行" in q and "基金" in q and "认购" in q:
      return ["掌上银行基金功能的主要服务内容", "掌上银行基金认购的具体操作流程"]
  answer_verified = False
  try:
      payload = self._chat_json(prompts.ANSWER_SYSTEM,
                                prompts.build_answer_check_prompt(question, generation),
                                max_tokens=300)
      answer_verified = True
  except PilotBudgetStopped:
      raise
  except Exception:
      answer_verified = False
  verified = bool(state.get("grounded_verified")) and answer_verified
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_pilot_qa_guard.py tests/unit/test_agent_graph.py tests/unit/test_run_agent_trace.py -q`; expect all pass, then commit only Task 3 files with `fix: preserve app identity and verification truth`.

## Task 4: Configure key safely and make one Ark probe

**Files:** Create `scripts/probe_ark_pilot.py`; modify ignored project `.env` locally only after tests; test `tests/unit/test_probe_ark_pilot.py` (new). No key is committed.

**Interfaces:** `probe()` issues one `DeepSeekAPIClient.chat([{"role":"user","content":"请只回复：连接正常"}], max_tokens=16, node="probe")`; it prints model, returned token counts and request ID only, never message payload or secret.

- [ ] **Step 1: Write failing test.** With a fake client, assert exactly one request, `max_tokens=16`, no secret in stdout; paid-disabled state raises before transport use.

  ```python
  def test_probe_is_one_bounded_call(fake_client, capsys):
      probe(fake_client)
      assert fake_client.calls == 1
      assert fake_client.last_max_tokens == 16
      assert "fake-test-key" not in capsys.readouterr().out
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_probe_ark_pilot.py -q`; expect failure.
- [ ] **Step 3: Implement probe and local config.** Read the user-provided key file only in memory; reject empty/multiline content and do not echo it. Set `FINANCE_LLM_PROVIDER=ark`, `ARK_MODEL=doubao-seed-2-1-pro-260915`, Ark URL and token cap in ignored `.env`, but leave `FINANCE_DISABLE_PAID_API=1`. The probe command uses a one-process `FINANCE_DISABLE_PAID_API=0` override; no persistent switch. Check user-visible console resource-pack balance before and after; if wrong model, absent usage or unexpected deduction, stop here.

  ```python
  def probe(client: DeepSeekAPIClient) -> str:
      return client.chat([{"role": "user", "content": "请只回复：连接正常"}],
                         max_tokens=16, node="probe")
  # The CLI obtains the key from the supplied local file, not an argv value or log line.
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_probe_ark_pilot.py -q`; expect pass. Then run **one** manual probe only after Tasks 1–3 pass; record usage/balance in `docs/baseline/phase-c-agent-runbook.md` without key. Commit the probe/test only with `feat: add one-shot Ark connectivity probe`.

## Task 5: Isolated, chunk-whitelisted LightRAG pilot

**Files:** Create `scripts/run_kg_pilot.py`; modify `scripts/build_lightrag_kg.py` only if a reusable selection helper is necessary; test `tests/unit/test_kg_pilot.py` (new).

**Interfaces:** `select_chunks(chunks: list[dict], allowlist: dict[str, set[int]]) -> list[dict]` keeps only named source/page pairs. `prepare_pilot(chunks: list[dict], allowlist: dict[str, set[int]], root: Path) -> Path` writes a manifest and mini-documents in a new `output/lightrag_pilot/` directory, without touching `output/lightrag/`. `lightrag_usage_auditable() -> bool` checks for a separately metered Ark route before any insertion.

- [ ] **Step 1: Write failing tests.** Feed a synthetic 833-chunk manual; allow pages 132–151 and assert only those pages survive. Assert path resolution never targets `output/lightrag/`, manifest records original source and page, and missing page provenance rejects a chunk. Assert default `--dry-run` makes zero LightRAG requests and `--clear` is unavailable.

  ```python
  def test_page_whitelist_excludes_rest_of_manual():
      rows = [{"source_doc": "manual", "page_start": p, "content": str(p)}
              for p in range(1, 834)]
      selected = select_chunks(rows, {"manual": set(range(132, 152))})
      assert len(selected) == 20
      assert {r["page_start"] for r in selected} == set(range(132, 152))
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_kg_pilot.py -q`; expect failure.
- [ ] **Step 3: Implement minimal scope tool.** Load active snapshot chunks; select a small non-sensitive allowlist; write uniquely named pilot documents with original source/page markers and a manifest. Start an isolated LightRAG service only with an independently auditable Ark usage path. If its calls cannot be metered or capped, stop **before** paid insertion and record “图谱待计量”, rather than claiming Task 5 passed. For metered calls, submit one mini-document at a time, check budget and console after each, and verify returned entities/relations map to manifest pages.

  ```python
  def select_chunks(chunks: list[dict], allowlist: dict[str, set[int]]) -> list[dict]:
      return [row for row in chunks
              if row.get("source_doc") in allowlist
              and row.get("page_start") in allowlist[row["source_doc"]]
  pilot_root = ROOT / "output" / "lightrag_pilot"
  if not lightrag_usage_auditable():
      raise RuntimeError("图谱待计量：禁止付费插入")
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_kg_pilot.py tests/unit/test_lightrag_channel.py -q`; expect pass. Only then manually try one mini-document if metering is proven; commit Task 5 code/tests with `feat: isolate bounded graph pilot`.

## Task 6: Genuine visual answer on a tiny image sample

**Files:** Create `src/agent/visual_answer.py`; modify `src/agent/nodes.py`, `src/models/gateway.py`; test `tests/unit/test_visual_pilot.py` (new).

**Interfaces:** `build_image_messages(question: str, evidence: dict, allowed_root: Path) -> list[dict]` resolves exactly one trusted page image under `allowed_root`, reads it into an Ark-supported `image_url` data URL, and includes source/page text. The existing client accepts `messages: list[dict[str, Any]]`.

- [ ] **Step 1: Write failing tests.** A path outside the trusted visual-page root is rejected; missing or >5-MB image is rejected; a valid tiny PNG yields one text + one image part; image-only evidence with API disabled returns “未经图像内容验证”, never a fabricated visual answer. Mock Ark `usage` must be present before claiming visual verification.

  ```python
  def test_reject_image_outside_root(tmp_path):
      allowed = tmp_path / "pages"
      allowed.mkdir()
      other = tmp_path / "private.png"
      other.write_bytes(b"not-an-image")
      with pytest.raises(ValueError):
          build_image_messages("图中是什么？", {"image_path": str(other)}, allowed)
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_visual_pilot.py -q`; expect failure.
- [ ] **Step 3: Implement the resolver and model call.** Keep local CLIP retrieval unchanged. For one selected page image, use the same Ark client and budget ledger but reserve a conservative fixed 50,000 tokens before sending; use `max_tokens<=500`. Save actual returned usage and source/page citation; if image usage is unavailable or the evidence identity fails, stop paid visual trials and label the response unverified.

  ```python
  image_path = Path(evidence["image_path"]).resolve(strict=True)
  root = allowed_root.resolve(strict=True)
  if not image_path.is_relative_to(root) or image_path.stat().st_size > 5_000_000:
      raise ValueError("图像不在可信页图目录或过大")
  data_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
  text_part = {"type": "text", "text": question}
  image_part = {"type": "image_url", "image_url": {"url": data_url}}
  messages = [{"role": "user", "content": [text_part, image_part]}]
  ```
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/unit/test_visual_pilot.py tests/unit/test_chinese_vision.py -q`; expect pass. Try only one non-sensitive visual case after Ark budget validation; commit Task 6 files with `feat: verify selected visual evidence with Ark`.

## Task 7: Live SSE and representative acceptance run

**Files:** Modify `app/api/runs.py`, `app/services/run_service.py`, `app/services/qa_runtime.py`, `src/agent/graph.py`, `frontend/src/api/client.ts`, `frontend/src/views/ChatView.vue`; test `tests/integration/test_runs_sse.py` (new), `frontend/tests/` current runner; update `docs/baseline/phase-c-agent-runbook.md`.

**Interfaces:** `POST /api/runs/stream` accepts `RunCreateRequest` with the same authentication as `POST /runs`; it creates the run and emits `event: started` with its ID, `event: node` after each LangGraph node, then `event: done` with verified status or `event: error` with a redacted reason. `RunService.create_and_execute(..., on_node: Callable[[dict], None] | None = None)` forwards events through `QARuntime.answer(..., on_node=...)`; `run_event_iter(body: RunCreateRequest, user_id: int)` runs inference in a worker with its own DB session and yields SSE strings; existing `/trace` remains available for historical runs. A post-completion `GET` cannot provide live events, so do not implement SSE as a trace replay.

- [ ] **Step 1: Write failing tests.** An unauthenticated caller gets 401; authenticated caller receives `started`, ordered `node` events and one terminal event from `POST /runs/stream`; no API key/prompt/evidence body appears in event data. Frontend test verifies live nodes render before final answer and displays “未验证” on budget stop.

  ```python
  def test_run_stream_requires_auth(client):
      response = client.post("/api/runs/stream", json={"question": "基金认购"})
      assert response.status_code == 401
  ```
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/integration/test_runs_sse.py -q` and `npm --prefix frontend test -- --run`; expect failures for absent SSE.
- [ ] **Step 3: Implement the smallest authenticated event feed.** Have `run_agent` use `graph.stream(..., stream_mode="values")` only when an `on_node` callback is provided; each yielded value is the complete state, so emit only the newly appended `trace` entries as redacted `{node, decision, elapsed_ms}` events and return the last value as the final state. Run inference in a worker thread and push each event to a bounded queue; `StreamingResponse` drains the queue, sends one terminal event and closes. Do not hold a DB session across the streaming generator; create/commit the run using a fresh session inside the worker. Frontend uses authenticated `fetch` streaming, not bare `EventSource`, because auth requires headers. Preserve `/chat` and `/trace` behavior. When no paid model is available, show the local mode/verification state honestly.

  ```python
  @router.post("/stream")
  def stream_run(body: RunCreateRequest,
                 user: User = Depends(get_current_user)):
      return StreamingResponse(run_event_iter(body, user.id),
                               media_type="text/event-stream")
  ```
- [ ] **Step 4: Run** the SSE tests, all focused Agent/budget tests, frontend tests and type/build check (`npm --prefix frontend run build`). Then run at most six representative questions in small batches: text, cross-document, table, image, no-evidence, 掌上银行. Compare run traces/citations and actual token ledger with console balance; stop at the first safety failure. Write measured totals and remaining gaps in the runbook, then commit only Task 7 files with `feat: stream audited agent progress`.

## Final verification gate

- [ ] Run `.venv\Scripts\python.exe -m pytest tests/unit/test_ark_client.py tests/unit/test_pilot_budget.py tests/unit/test_pilot_qa_guard.py tests/unit/test_kg_pilot.py tests/unit/test_visual_pilot.py tests/integration/test_runs_sse.py -q` and `npm --prefix frontend run build`; report exact failures, not inferred success.
- [ ] Confirm `/api/health` still reports the active 920-document snapshot and compare a fresh `/api/runs/{id}/trace` against frontend SSE; verify no old KB or LightRAG data were cleared.
- [ ] Compare SQLite usage ledger, model-returned usage and Ark console deduction; explicitly label any discrepancy, and keep paid calls off until resolved.
- [ ] Report which methods were truly exercised, which were skipped by safety gates, spend/remaining balance and specific manual/paid follow-up. Never claim full-corpus graph or overall answer-accuracy acceptance from this pilot.
