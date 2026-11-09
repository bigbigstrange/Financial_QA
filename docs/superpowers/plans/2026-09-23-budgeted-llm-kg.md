# Budgeted LLM Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract auditable entity relationships from active chunks with Ark under the existing shared 1.5M-token ceiling, then offer the completed subset as an opt-in retrieval channel.

**Architecture:** A single-process CLI checkpoints each chunk in SQLite before calling the existing metered Ark client; valid triples point back to original chunk IDs and evidence spans. A read-only retriever maps completed triples to active chunks and feeds the existing hybrid RRF. No independent LightRAG insertion.

**Tech Stack:** Python 3, SQLite stdlib, installed openai-compatible gateway, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-budgeted-llm-kg-design.md`

## Global Constraints

- Use only active `output/kb_full` snapshot chunks, preserve all 920 online documents.
- Use `PilotBudget` max 1,500,000 tokens and Doubao-Seed-2.1-pro; stop on uncertain paid response.
- Never send a second request for a `started` or `needs_review` chunk.
- Do not activate the pilot graph until provenance and retrieval tests pass.

## Review Focus

- An interrupted remote call may have consumed tokens; resume must not retry it.
- Model JSON may contain fabricated evidence; reject triples whose evidence is not a substring of its source chunk.
- A wrong snapshot/version or missing chunk ID must never enter online graph retrieval.
- Empty/invalid model output must not be called a completed graph extraction.
- Budget exhaustion must stop the loop with exact progress, not swallow the failure and continue.

---

### Task 1: Checkpointed extractor

**Files:** Create `scripts/build_budgeted_kg.py`, `tests/unit/test_budgeted_kg.py`.

**Interfaces:** `extract_chunk(row: dict, llm: Any) -> list[dict]`, `KGCheckpoint(path: Path)` with `start`, `complete`, `fail`, `summary`; CLI `--max-chunks`, `--dry-run`, `--checkpoint`.

- [ ] Write tests for valid evidence, fabricated evidence rejection, resumed completed skip, started→needs_review, model error stop and the paid-disabled preflight.
- [ ] Run `.venv/Scripts/python.exe -m pytest tests/unit/test_budgeted_kg.py -q` and observe failures from missing implementation.
- [ ] Implement the minimum SQLite state machine and JSON parsing. Each relation stores `snapshot_id`, `document_version`, `source_doc`, `page`, `chunk_id`, `head`, `relation`, `tail`, `evidence`; only evidence present verbatim in source text is accepted. Use the installed `DeepSeekAPIClient.chat(..., node=f"kg:{chunk_id}")` and its `last_response` usage; never log the key or raw model output.
- [ ] Run focused tests, dry-run the real active snapshot to count the prioritized candidate pool, and verify the active snapshot was not changed.

### Task 2: Opt-in graph retrieval

**Files:** Create `src/system/kg_pilot_channel.py`, `tests/unit/test_kg_pilot_channel.py`; modify only the retrieval-stack assembly in `src/system/retrieval.py`.

**Interfaces:** `build_kg_pilot_retriever(blocks: list[dict]) -> KGPilotRetriever | None`; `search(query: str, top_k: int, visibility=None) -> list[dict]`.

- [ ] Write tests: only completed triples with matching active `chunk_id` appear; query entity tokens find a source chunk; visibility filters denied versions; no flag means original channel unchanged.
- [ ] Run tests to watch missing feature fail.
- [ ] Read checkpoint in read-only mode, build a small in-memory entity-to-chunk map, and return original source chunks with `retrieval_channel="graph_llm_pilot"` and provenance. Let existing `HybridRetriever` add its RRF; prefer pilot over LightRAG only if `FINANCE_KG_PILOT_ENABLED=1`.
- [ ] Run focused and full unit tests.

### Task 3: Bounded real pilot and evidence

**Files:** Output only under `output/kg_pilot/`; update existing run record with counts.

**Interfaces:** Existing `output/pilot_budget.sqlite3` token ledger; no new account endpoint.

- [ ] Read budget summary and provider configuration without printing secrets; ensure official Ark endpoint and approved model.
- [ ] Run a single-chunk real extraction first; confirm `completed` checkpoint, actual usage, an evidence-backed triple and local graph query. If the call is uncertain or quota error, stop and report exact state.
- [ ] If successful, run a bounded prioritized subset within the shared token budget. Never override 1.5M; stop on budget gate and record completed/needs_review/remaining counts.
- [ ] Re-run `.venv/Scripts/python.exe -m pytest -q`, check health and active snapshot counts; do not claim full KG coverage or answer accuracy.
