# Visual Evidence Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop page-level visual result crowding and wrong-bank answer contamination, and establish an honest offline audit of the 500 uploaded questions.

**Architecture:** Keep the active 920-document snapshot and existing RRF. Make Chinese-CLIP return one representative evidence per distinct page, selected by lexical overlap with the question, while retaining image-only pages. Reuse the existing named-App grading guard. Build a read-only workbook/corpus audit that measures question coverage without claiming answer accuracy.

**Tech Stack:** Python 3, NumPy, unittest/pytest, openpyxl, existing FAISS/Chinese-CLIP and LangGraph code.

**Spec:** `docs/superpowers/specs/2026-09-23-visual-evidence-quality-design.md`

## Global Constraints

- Do not rebuild or switch the active knowledge snapshot; keep its 920 indexed documents.
- Do not process the 3 unconverted Office files.
- Do not call paid model APIs or Web search in this phase.
- Preserve the 213 pre-existing dirty worktree entries; stage only files attributable to this task.
- Treat `test/merged.xlsx` as read-only; it has questions but no gold answers.

## Review Focus

- A page with several chunks must contribute only one hit; another page must still appear in top-k.
- The same chunk spanning two pages must be returned once, on its higher-ranked page.
- An image-only page must still surface as `page_image` with source/page metadata.
- A hidden document version must never appear merely because its page ranks highly.
- Duplicate question IDs or texts in the workbook must remain separate rows; the audit must not infer answer accuracy.

---

### Task 1: Visual page diversity and representative selection

**Files:**
- Modify: `src/system/vision.py:129-167`
- Create: `tests/unit/test_visual_page_diversity.py`

**Interfaces:**
- Consumes: `ChineseVisionRetriever.search(query, top_k, visibility)` and page inventory records with `chunks`, `page_id`, `document_version`, `page`.
- Produces: `resolve_hits(scores, top_k=6, visibility=None, query="") -> list[dict]`; one hit per page and no duplicate `chunk_id`.

- [ ] **Step 1: Write failing tests.** Add fixtures with two chunks on page A and one on page B, scores `[0.9, 0.8]`, query `基金认购`; assert results are one hit from each page and A's representative contains `基金认购`. Add a repeated cross-page chunk, a chunkless image page, and a denied version. Core assertion:

```python
hits = retriever.resolve_hits(np.array([0.9, 0.8]), top_k=2, query="基金认购")
assert [row["page_id"] for row in hits] == ["page_a", "page_b"]
assert hits[0]["chunk_id"] == "a_fund"
```
- [ ] **Step 2: Run the focused test and confirm failure.** Run `.venv/Scripts/python.exe -m pytest tests/unit/test_visual_page_diversity.py -q`; expected first test fails because current resolver returns both A chunks.
- [ ] **Step 3: Implement minimal page-level selection.** Pass `query` from `search` into `resolve_hits`. Rank each page's eligible chunks by count of meaningful query character bigrams present in `content`/`section_path`/`source_doc`, then by original order. Emit at most one representative for each page, keep global chunk de-duplication, and retain the existing chunkless `page_image` fallback and visibility filter. No new dependency. Selection shape:

```python
query_terms = {query[i:i + 2] for i in range(len(query) - 1) if all("\u4e00" <= ch <= "\u9fff" for ch in query[i:i + 2])}
eligible = [row for row in candidates if row_id(row) not in seen and allowed(row)]
best = max(eligible, key=lambda row: sum(term in evidence_text(row) for term in query_terms))
```
- [ ] **Step 4: Run focused and existing vision tests.** Run `.venv/Scripts/python.exe -m pytest tests/unit/test_visual_page_diversity.py tests/unit/test_chinese_vision.py tests/unit/test_visual_score_isolation.py -q`; all must pass. If older tests encoded multi-chunk-per-page behavior, update only their expected contract with explanation.

### Task 2: Named-App insufficient-evidence regression

**Files:**
- Create: `tests/unit/test_named_app_evidence_gap.py`
- Read, modify only if regression reveals a defect: `src/agent/nodes.py:217-300,593-609`

**Interfaces:**
- Consumes: `AgentNodes.grade_documents`, `build_graph`, `run_agent`; existing `_supports_named_app` rule.
- Produces: a stable `status="no_evidence"` for a “掌上银行＋基金认购流程” question when documents have either App marketing text without funds or other-bank fund procedures.

- [ ] **Step 1: Add a failing regression if one is missing.** Model actual corpus-shaped candidates: an agricultural-bank App introduction without fund instructions, a transport-bank fund procedure, and a generic fund primer; assert no relevant documents and no invented steps. Also assert a genuinely matching App+fund manual remains eligible.

```python
state = initial_state("掌上银行基金认购流程")
state["documents"] = [app_marketing, other_bank_fund_manual, generic_fund_primer]
assert AgentNodes(None, llm=None).grade_documents(state)["relevant_documents"] == []
```
- [ ] **Step 2: Run the focused test.** Run `.venv/Scripts/python.exe -m pytest tests/unit/test_named_app_evidence_gap.py tests/unit/test_pilot_qa_guard.py -q`; if already green, do not rewrite production guard.
- [ ] **Step 3: Fix only a demonstrated gap.** If the test fails, adjust the existing `_supports_named_app`/grading path narrowly; preserve comparison questions and non-App questions. If it passes, make no production change in this task.
- [ ] **Step 4: Re-run agent graph regression.** Run `.venv/Scripts/python.exe -m pytest tests/unit/test_named_app_evidence_gap.py tests/unit/test_pilot_qa_guard.py tests/unit/test_agent_graph.py -q`.

### Task 3: Read-only 500-question inventory audit

**Files:**
- Create: `scripts/audit_question_coverage.py`
- Create: `tests/unit/test_question_coverage.py`
- Read only: `test/merged.xlsx`, `output/kb_full/chunks.jsonl`

**Interfaces:**
- Consumes: `audit(workbook_path: Path, chunks_path: Path) -> dict`, sheet `Sheet1` columns `question_id`, `question`.
- Produces: JSON report with `rows`, `unique_questions`, `unique_question_ids`, `duplicate_question_rows`, `duplicate_id_rows`, `question_rows` keyed by sheet row, and explicit `answer_accuracy="not_evaluated"`; for the known App case include count of exact phrase hits and count of chunks containing both `掌上银行` and `基金`.

- [ ] **Step 1: Write fixture tests.** Create a temporary workbook with duplicate IDs/questions and a tiny JSONL corpus; assert all rows remain, exact phrase counts are deterministic, and no field claims answer correctness.

```python
report = audit(workbook_path, chunks_path)
assert report["rows"] == 3
assert report["unique_question_ids"] == 2
assert report["answer_accuracy"] == "not_evaluated"
```
- [ ] **Step 2: Run and confirm failure.** Run `.venv/Scripts/python.exe -m pytest tests/unit/test_question_coverage.py -q`; expected import failure before script exists.
- [ ] **Step 3: Implement audit with standard library plus installed openpyxl.** Use read-only workbook loading and streaming JSONL; never mutate source files. CLI accepts `--workbook`, `--chunks`, `--output` and writes only the requested report path. Entry point:

```python
def audit(workbook_path: Path, chunks_path: Path) -> dict:
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        rows = [(i, str(values[0] or ""), str(values[1] or "").strip())
                for i, values in enumerate(workbook["Sheet1"].iter_rows(min_row=2, values_only=True), start=2)
                if values[1]]
    finally:
        workbook.close()
    ids = {question_id for _, question_id, _ in rows}
    questions = {question for _, _, question in rows}
    app_hits = 0
    app_fund_hits = 0
    with chunks_path.open(encoding="utf-8") as stream:
        for line in stream:
            content = str(json.loads(line).get("content") or "")
            app_hits += "掌上银行" in content
            app_fund_hits += "掌上银行" in content and "基金" in content
    return {"rows": len(rows), "unique_question_ids": len(ids),
            "unique_questions": len(questions),
            "duplicate_id_rows": len(rows) - len(ids),
            "duplicate_question_rows": len(rows) - len(questions),
            "question_rows": [{"sheet_row": i, "question_id": qid, "question": q} for i, qid, q in rows],
            "app_phrase_chunks": app_hits, "app_and_fund_chunks": app_fund_hits,
            "answer_accuracy": "not_evaluated"}
```
- [ ] **Step 4: Test and run on real input.** Run focused tests, then `.venv/Scripts/python.exe scripts/audit_question_coverage.py --workbook test/merged.xlsx --chunks output/kb_full/chunks.jsonl --output output/eval/question_coverage_2026-09-23.json`; check exactly 500 question rows and `answer_accuracy=not_evaluated`.

### Task 4: Verification and handoff

**Files:**
- Review: only files changed in Tasks 1-3 and the generated JSON report.

**Interfaces:** No new runtime interface.

- [ ] **Step 1: Run regression suite.** `.venv/Scripts/python.exe -m pytest tests/unit/test_visual_page_diversity.py tests/unit/test_chinese_vision.py tests/unit/test_visual_score_isolation.py tests/unit/test_named_app_evidence_gap.py tests/unit/test_pilot_qa_guard.py tests/unit/test_agent_graph.py tests/unit/test_question_coverage.py -q`.
- [ ] **Step 2: Inspect `git diff` and `git status --short --untracked-files=all`.** Verify no active snapshot, Office files, `.env`, or unrelated dirty entries were altered.
- [ ] **Step 3: Commit only isolated new tests/script/plan.** Do not stage the pre-existing untracked `src/system/vision.py` or other user-owned dirty files without an attributable diff boundary; explicitly report any verified code fix left uncommitted for that reason.
- [ ] **Step 4: Report evidence.** State tests, 500-row audit, whether first App question has source coverage, and that answer accuracy and subsequent Web/reliability/deployment stages remain open.
