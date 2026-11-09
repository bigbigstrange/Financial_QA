# Phase 2 Document and Evidence Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned DocIR v2 pipeline whose Word/PDF/table evidence can be traced to immutable source locations, whose updates activate atomically, and whose deleted documents disappear from new searches immediately.

**Architecture:** Native parsers produce immutable DocIR v2 artifacts and quality reports; SQLAlchemy records own document/version/snapshot visibility; FAISS consumes immutable snapshot manifests and remains a replaceable acceleration layer. Each Run persists the active knowledge snapshot before inference, and Evidence records retain enough identity to read the exact source block later.

**Tech Stack:** Python 3.11, dataclasses, Pydantic 2, python-docx 1.1, PyMuPDF 1.23, Camelot, openpyxl 3.1, python-pptx 0.6, SQLAlchemy 2.0, FastAPI 0.115, FAISS 1.7, unittest.

**Spec:** `docs/superpowers/specs/2026-09-19-phase-2-document-evidence-design.md`

## Global Constraints

- Preserve the current `knowledge_base.json`, old FAISS files, `/api/chat`, and `/api/runs` as the rollback baseline.
- Do not add Docling, Qdrant, Redis, Celery, BM25, RRF, rerankers, or LangGraph in this phase.
- `document_id` is a server-assigned logical UUID; a file name is display metadata and never an identity.
- `document_version` changes when content, DocIR schema, parser implementation/configuration, or chunker version changes.
- `block_id` is derived only from `document_version`, canonical structural locator, and block type.
- Word without a verified rendered page map uses `page=None` and `section_path + element_index`.
- PostgreSQL/SQLite catalog state is authoritative for visibility; DocIR, manifests, page images, and FAISS files are immutable derived artifacts.
- New Runs persist `knowledge_snapshot_id` before model or retrieval execution.
- A delete transaction must affect the next new search without waiting for physical index or file cleanup.
- All new artifact writes use a temporary sibling followed by `os.replace`; failed candidates never overwrite active artifacts.
- Use `C:\code\个人资料\code\Agentic_RAG\finance-qa-platform\.conda\python.exe` for project Python commands.
- Every new test module uses `unittest.TestCase`; file tests create `self.temp_dir = TemporaryDirectory()` and `self.tmp_path = Path(self.temp_dir.name)`, database tests create an in-memory SQLite `self.session`, and API tests follow `tests/integration/test_runs_api.py` with `TestClient` and dependency overrides.

## Review Focus

- A DOCX containing empty paragraphs, nested tables, and paragraph-table-paragraph interleaving must retain body order and stable locators; Task 3 pins this with a generated fixture.
- Two uploads with the same display name but different logical document IDs must never collide, while a byte-identical retry for one document must deduplicate; Tasks 1 and 8 test both cases.
- Merged/multi-row headers and a table larger than both row and token limits must keep header/unit context in every child group; Task 2 exercises these inputs.
- A stale FAISS cache may still contain a deleted version, but query-time visibility must remove it and refill from allowed candidates; Task 10 tests this directly.
- A process crash after candidate files are written but before database activation must leave the old snapshot active and permit a clean retry; Task 9 tests the transaction boundary.

---

## File Structure

### Parsing and artifacts

- `src/pipeline/identity.py`: canonical hashing, `ParseContext`, version IDs, block IDs, and locators.
- `src/pipeline/docir.py`: DocIR v2 dataclasses, nullable page, serialization, validation, and legacy reader boundary.
- `src/pipeline/tables.py`: table normalization, header/unit metadata, deterministic retrieval text, and row groups.
- `src/pipeline/parsers.py`: format parsers that consume `ParseContext` and emit DocIR v2.
- `src/pipeline/quality.py`: per-version and snapshot quality reports plus hard-gate validation.
- `src/pipeline/file_validation.py`: extension/MIME/signature/ZIP-limit checks before parsing.
- `src/pipeline/snapshots.py`: canonical immutable manifest model and atomic writer/reader.
- `scripts/preprocess.py`: candidate artifact generation using the new components.
- `scripts/migrate_phase2_knowledge.py`: side-by-side full-corpus migration and activation command.
- `scripts/audit_parser_baseline.py`: DocIR v2/snapshot audit metrics.

### Catalog, API, and runtime

- `app/models/__init__.py`: document, version, snapshot, evidence, and Run snapshot records.
- `app/db/schema_upgrades.py`: idempotent additive upgrade for an existing phase-1 database.
- `app/services/document_catalog.py`: catalog transactions and state transitions.
- `app/services/document_service.py`: validation → parse → quality → snapshot → activation orchestration.
- `app/services/document_visibility.py`: SQLAlchemy implementation of retrieval visibility.
- `app/services/evidence_reader.py`: exact DocIR block lookup with authorization.
- `app/api/documents.py`: minimal admin create/update/delete/rollback/status endpoints.
- `app/api/evidence.py`: authenticated evidence read endpoint.
- `app/schemas/__init__.py`: document/snapshot/evidence response schemas.
- `src/system/index_fingerprint.py`: exact cache compatibility contract.
- `src/system/retrieval.py`: snapshot loading, table text ingestion, cache validation, and visibility filtering.
- `src/system/qa_engine.py`: snapshot descriptor passed to retrieval construction.
- `app/services/qa_runtime.py`: one runtime per requested immutable snapshot with bounded replacement.
- `app/services/run_service.py`: persist snapshot before calling the runtime and persist Evidence rows.

### Tests and documentation

- `tests/unit/test_docir_v2.py`
- `tests/unit/test_table_chunking.py`
- `tests/unit/test_word_parser_v2.py`
- `tests/unit/test_pdf_parser_v2.py`
- `tests/unit/test_office_parsers_v2.py`
- `tests/unit/test_parse_quality.py`
- `tests/unit/test_file_validation.py`
- `tests/unit/test_document_catalog.py`
- `tests/unit/test_snapshot_activation.py`
- `tests/unit/test_snapshot_retrieval.py`
- `tests/unit/test_run_knowledge_snapshot.py`
- `tests/integration/test_documents_api.py`
- `tests/integration/test_evidence_api.py`
- `docs/baseline/phase-2-runbook.md`

---

### Task 1: DocIR v2 Identity and Serialization

**Files:**
- Create: `src/pipeline/identity.py`
- Modify: `src/pipeline/docir.py`
- Modify: `src/pipeline/__init__.py`
- Test: `tests/unit/test_docir_v2.py`

**Interfaces:**
- Produces: `ParseContext.for_file(file_path, document_id, parser_config_hash, chunker_version) -> ParseContext`.
- Produces: `make_document_version(context, parser_name, parser_version) -> str`.
- Produces: `make_block_id(document_version, structural_locator, block_type) -> str`.
- Produces: `DocIR.validate_v2() -> list[str]`, returning hard-gate failure codes.
- Preserves: existing `DocIR(doc_id=..., source_path=..., source_format=...)` construction for legacy tests.

- [ ] **Step 1: Write deterministic identity and round-trip tests**

```python
def test_same_context_produces_same_version_and_block_id(self):
    source = self.tmp_path / "same.docx"
    source.write_bytes(b"same bytes")
    ctx = ParseContext.for_file(source, "doc-a", "cfg-1", "chunk-v2")
    version = make_document_version(ctx, "word_structured", "2")
    assert version == make_document_version(ctx, "word_structured", "2")
    assert make_block_id(version, "body/000001/paragraph", "paragraph") == make_block_id(
        version, "body/000001/paragraph", "paragraph"
    )

def test_same_name_different_logical_documents_do_not_collide(self):
    source = self.tmp_path / "rules.docx"
    source.write_bytes(b"same bytes")
    one = ParseContext.for_file(source, "doc-a", "cfg-1", "chunk-v2")
    two = ParseContext.for_file(source, "doc-b", "cfg-1", "chunk-v2")
    assert make_document_version(one, "word_structured", "2") != make_document_version(
        two, "word_structured", "2"
    )

def test_word_location_does_not_require_fake_page(self):
    block = DocBlock(
        block_id="blk-a", block_type="paragraph", text="正文", page=None,
        structural_locator="body/000002/paragraph", element_index=2,
    )
    assert block.page is None
    assert block.structural_locator == "body/000002/paragraph"
```

- [ ] **Step 2: Run the focused test and confirm the new interfaces are absent**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_docir_v2 -v`

Expected: FAIL because `src.pipeline.identity` and the new DocBlock fields do not exist.

- [ ] **Step 3: Implement canonical identities and backward-compatible DocIR v2 fields**

```python
@dataclass(frozen=True)
class ParseContext:
    document_id: str
    source_name: str
    source_sha256: str
    source_size_bytes: int
    parser_config_hash: str
    chunker_version: str

def make_document_version(ctx, parser_name, parser_version):
    return "dv_" + stable_digest(
        ctx.document_id, ctx.source_sha256, "2", parser_name, parser_version,
        ctx.parser_config_hash, ctx.chunker_version,
    )[:32]

def make_block_id(document_version, structural_locator, block_type):
    return "blk_" + stable_digest(
        document_version, structural_locator, block_type
    )[:32]
```

Add `schema_version=2`, source metadata, `element_index`, `structural_locator`, nullable `page`, and `validate_v2()` without removing the phase-1 constructor fields.

- [ ] **Step 4: Run DocIR and existing contract tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_docir_v2 tests.unit.test_contracts -v`

Expected: PASS; the legacy constructor test and all v2 identity tests pass.

- [ ] **Step 5: Commit the identity foundation**

```powershell
git add src/pipeline/identity.py src/pipeline/docir.py src/pipeline/__init__.py tests/unit/test_docir_v2.py
git commit -m "feat: add deterministic docir v2 identities"
```

### Task 2: Structured Table Normalization and Row Groups

**Files:**
- Create: `src/pipeline/tables.py`
- Modify: `src/pipeline/__init__.py`
- Test: `tests/unit/test_table_chunking.py`

**Interfaces:**
- Consumes: `make_block_id()` and DocIR v2 `DocBlock` from Task 1.
- Produces: `normalize_table(rows, *, document_version, locator, section_path=None, caption=None, unit=None, max_rows=20, max_chars=6000) -> list[DocBlock]`.
- Contract: indexable child blocks use `block_type="table_rows"`; the first returned block is the `block_type="table"` parent.

- [ ] **Step 1: Write table context, grouping, and merged-header tests**

```python
def test_large_table_repeats_header_unit_and_parent(self):
    rows = [["期限", "金额"]] + [[str(i), str(i * 10)] for i in range(45)]
    blocks = normalize_table(
        rows, document_version="dv-a", locator="body/000003/table",
        caption="期限结构", unit="万元", max_rows=20,
    )
    parent, *groups = blocks
    assert [len(group.table_rows) for group in groups] == [20, 20, 5]
    assert all(group.parent_id == parent.block_id for group in groups)
    assert all("[单位] 万元" in group.text for group in groups)
    assert all(group.table_headers == ["期限", "金额"] for group in groups)

def test_multi_row_header_is_flattened_without_losing_original_rows(self):
    rows = [["", "2025年", "2025年"], ["项目", "数量", "金额"], ["存款", "2", "100"]]
    blocks = normalize_table(rows, document_version="dv-a", locator="sheet/S1/rows/1-3")
    child = blocks[1]
    assert child.table_headers == ["项目", "2025年/数量", "2025年/金额"]
    assert child.meta["header_rows"] == rows[:2]
```

- [ ] **Step 2: Run the table tests and verify import failure**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_table_chunking -v`

Expected: FAIL because `src.pipeline.tables` does not exist.

- [ ] **Step 3: Implement deterministic table blocks**

```python
def serialize_group(section_path, caption, unit, headers, rows, start_row):
    lines = []
    if section_path:
        lines.append(f"[章节] {section_path}")
    if caption:
        lines.append(f"[表格] {caption}")
    if unit:
        lines.append(f"[单位] {unit}")
    lines.append("[列] " + " | ".join(headers))
    lines.extend(
        f"[行 {start_row + offset}] " + " | ".join(row)
        for offset, row in enumerate(rows)
    )
    return "\n".join(lines)
```

Detect one- or multi-row headers deterministically, preserve original header rows in metadata, forward-fill header spans only in the normalized view, split by both row and character limits, and assign parent/child locators and IDs.

- [ ] **Step 4: Run the table and DocIR tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_table_chunking tests.unit.test_docir_v2 -v`

Expected: PASS with stable IDs and all row groups retaining context.

- [ ] **Step 5: Commit table normalization**

```powershell
git add src/pipeline/tables.py src/pipeline/__init__.py tests/unit/test_table_chunking.py
git commit -m "feat: preserve structured table row groups"
```

### Task 3: Word Body-Order Parser

**Files:**
- Modify: `src/pipeline/parsers.py:59-130`
- Test: `tests/unit/test_word_parser_v2.py`

**Interfaces:**
- Consumes: `ParseContext`, identity helpers, and `normalize_table()`.
- Produces: `WordDocumentParser.parse(file_path, *, context=None) -> DocIR`.
- Contract: every Word block has `page=None`, body-based locator, element index, and current heading path.

- [ ] **Step 1: Generate mixed and nested DOCX fixtures in tests**

```python
def test_word_preserves_paragraph_table_paragraph_order(self):
    path = self.tmp_path / "mixed.docx"
    document = Document()
    document.add_heading("第一章", level=1)
    document.add_paragraph("表前")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "项目", "金额"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "存款", "100"
    document.add_paragraph("表后")
    document.save(path)
    context = ParseContext.for_file(path, "doc-word", "cfg-v2", "chunk-v2")
    parsed = WordDocumentParser().parse(path, context=context)
    assert [b.block_type for b in parsed.pages[0].blocks] == [
        "heading", "paragraph", "table", "table_rows", "paragraph"
    ]
    assert all(b.page is None for b in parsed.pages[0].blocks)
    assert parsed.pages[0].blocks[-1].section_path == "第一章"
```

Add a second test with an empty body paragraph before the table and a nested table in a cell; assert later locators retain their actual OOXML body/cell indices.

- [ ] **Step 2: Run Word tests and observe the old reordered output**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_word_parser_v2 -v`

Expected: FAIL because tables are appended after all paragraphs and Word blocks use page 1.

- [ ] **Step 3: Implement one-pass OOXML body traversal**

```python
for element_index, child in enumerate(document.element.body.iterchildren()):
    if isinstance(child, CT_P):
        paragraph = Paragraph(child, document)
        blocks.extend(parse_word_paragraph(paragraph, element_index, section_stack, version))
    elif isinstance(child, CT_Tbl):
        table = Table(child, document)
        blocks.extend(normalize_word_table(table, element_index, section_stack, version))
```

Track headings by level, retain empty element indices without emitting empty blocks, recurse through nested cell children with `.../cell/R/C/...` locators, and put the ordered blocks in one logical `DocPage(page_no=None)` container.

- [ ] **Step 4: Run Word, table, and legacy contract tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_word_parser_v2 tests.unit.test_table_chunking tests.unit.test_contracts -v`

Expected: PASS; no fake Word page survives in v2 output.

- [ ] **Step 5: Commit ordered Word parsing**

```powershell
git add src/pipeline/parsers.py tests/unit/test_word_parser_v2.py
git commit -m "fix: preserve word document body order"
```

### Task 4: PDF Locations, Table De-duplication, and OCR Provenance

**Files:**
- Modify: `src/pipeline/parsers.py:132-320`
- Test: `tests/unit/test_pdf_parser_v2.py`

**Interfaces:**
- Consumes: Task 1 identities and Task 2 table normalization.
- Produces: `PDFDocumentParser.parse(file_path, *, context=None) -> DocIR`.
- Produces: `_needs_ocr(text_blocks, page_area, min_chars=20, min_coverage=0.0005) -> bool`.

- [ ] **Step 1: Write synthetic PDF location and OCR-threshold tests**

```python
def test_pdf_keeps_real_page_bbox_and_deterministic_locator(self):
    path = self.tmp_path / "two-pages.pdf"
    pdf = fitz.open()
    for text in ("first page", "second page"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    context = ParseContext.for_file(path, "doc-pdf", "cfg-v2", "chunk-v2")
    doc = PDFDocumentParser().parse(path, context=context)
    second = doc.pages[1].blocks[0]
    assert second.page == 2
    assert len(second.bbox) == 4
    assert second.structural_locator.startswith("page/000002/text/")

def test_low_text_coverage_triggers_ocr_but_normal_text_does_not(self):
    assert _needs_ocr([], 100_000)
    assert not _needs_ocr([(0, 0, 500, 100, "有效文本" * 30)], 100_000)
```

Mock Camelot results whose text overlaps a PyMuPDF block and assert only the normalized table row block is indexable for the overlapping region.

- [ ] **Step 2: Run the PDF tests and verify missing v2 provenance**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_pdf_parser_v2 -v`

Expected: FAIL because locators, OCR threshold metadata, and table overlap handling are absent.

- [ ] **Step 3: Implement PDF v2 blocks and conservative quality flags**

```python
locator = (
    f"page/{page_no:06d}/text/{source_index:06d}/bbox/"
    + ",".join(f"{value:.1f}" for value in bbox)
)
flags = ["uncertain_reading_order"] if has_layout_ambiguity(text_blocks) else []
```

Keep native page order, round bbox values to one decimal, mark ambiguous multi-column layouts instead of guessing, correlate table/text regions by bbox or normalized text overlap, and attach OCR engine/language/version metadata when fallback runs.

- [ ] **Step 4: Run PDF and table tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_pdf_parser_v2 tests.unit.test_table_chunking -v`

Expected: PASS without requiring an installed Tesseract binary for non-OCR fixtures.

- [ ] **Step 5: Commit PDF provenance**

```powershell
git add src/pipeline/parsers.py tests/unit/test_pdf_parser_v2.py
git commit -m "feat: add reliable pdf evidence locations"
```

### Task 5: Excel, PPT, Text, Image, and Legacy Parser Locators

**Files:**
- Modify: `src/pipeline/parsers.py:357-end`
- Test: `tests/unit/test_office_parsers_v2.py`

**Interfaces:**
- Consumes: Task 1 identity contract and Task 2 tables.
- Produces: every default parser accepts optional `context` and emits either v2 locators or an explicit `legacy_identity` quality flag.

- [ ] **Step 1: Write office locator and truncation tests**

```python
def test_excel_uses_sheet_rows_instead_of_fake_page(self):
    path = self.tmp_path / "sheet.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "资产负债表"
    sheet.append(["项目", "金额"])
    sheet.append(["现金", 10])
    workbook.save(path)
    context = ParseContext.for_file(path, "doc-xlsx", "cfg-v2", "chunk-v2")
    doc = ExcelDocumentParser().parse(path, context=context)
    child = next(b for b in doc.pages[0].blocks if b.block_type == "table_rows")
    assert child.page is None
    assert child.structural_locator.startswith("sheet/资产负债表/rows/")

def test_ppt_uses_slide_and_shape_locator(self):
    path = self.tmp_path / "slides.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "风险说明"
    presentation.save(path)
    context = ParseContext.for_file(path, "doc-ppt", "cfg-v2", "chunk-v2")
    doc = PowerPointDocumentParser().parse(path, context=context)
    assert doc.pages[0].blocks[0].structural_locator.startswith("slide/000001/shape/")
```

Add tests that Excel row-limit truncation emits `truncated_rows` and that legacy `.doc/.rtf` blocks carry their actual parser name plus `legacy_layout`.

- [ ] **Step 2: Run office parser tests against the old fake page IDs**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_office_parsers_v2 -v`

Expected: FAIL on page, locator, table group, or quality flag assertions.

- [ ] **Step 3: Bring every default parser onto the v2 context contract**

```python
locator = f"sheet/{quote(sheet_name, safe='')}/rows/{start:06d}-{end:06d}"
slide_locator = f"slide/{slide_no:06d}/shape/{shape_id:06d}"
```

Use `page=None` for text/image/Excel/Word structures without a verified page map, real slide numbers for PPT, table row groups for Excel/PPT tables, and explicit quality flags for COM/legacy layout fallbacks.

- [ ] **Step 4: Run all parser unit tests**

Run: `& '.\.conda\python.exe' -m unittest discover -s tests/unit -p 'test_*parser*v2.py' -v`

Expected: PASS for Word, PDF, Excel, PPT, text/image, and legacy cases.

- [ ] **Step 5: Commit remaining parser migration**

```powershell
git add src/pipeline/parsers.py tests/unit/test_office_parsers_v2.py
git commit -m "feat: add stable office document locators"
```

### Task 6: Parse Quality Reports and Candidate Preprocessing

**Files:**
- Create: `src/pipeline/quality.py`
- Modify: `scripts/preprocess.py:141-367`
- Modify: `scripts/audit_parser_baseline.py`
- Test: `tests/unit/test_parse_quality.py`
- Test: `tests/test_audit_parser_baseline.py`

**Interfaces:**
- Consumes: DocIR v2 from Tasks 1-5.
- Produces: `assess_document(doc: DocIR) -> ParseQualityReport`.
- Produces: `assess_snapshot(documents: Iterable[DocIR]) -> SnapshotQualityReport`.
- Produces: `QualityGateError(report)` when a hard failure exists.
- Produces: `process_document(file_path, output_dir, context, parsers) -> ProcessedDocument`; v2 callers must supply the catalog-issued `ParseContext`.

- [ ] **Step 1: Write hard-gate and warning tests**

```python
def test_duplicate_id_and_missing_locator_block_activation(self):
    doc = DocIR(
        doc_id="doc-1", source_path="rules.docx", source_format=".docx",
        document_version="dv-1", schema_version=2,
    )
    doc.add_page(DocPage(page_no=None, blocks=[
        DocBlock("same", "paragraph", "甲", None),
        DocBlock("same", "paragraph", "乙", None),
    ]))
    report = assess_document(doc)
    assert report.status == "failed"
    assert report.duplicate_block_id_count == 1
    assert report.missing_locator_count == 2

def test_uncertain_order_is_warning_not_hard_failure(self):
    doc = DocIR(
        doc_id="doc-1", source_path="rules.pdf", source_format=".pdf",
        document_version="dv-1", schema_version=2,
    )
    doc.add_page(DocPage(page_no=1, blocks=[DocBlock(
        "blk-1", "paragraph", "正文", 1,
        structural_locator="page/000001/text/000001",
        quality_flags=["uncertain_reading_order"],
    )]))
    report = assess_document(doc)
    assert report.status == "passed_with_warnings"
```

Add serialization round-trip, missing table parent/header/data, empty document, and silent truncation cases.

- [ ] **Step 2: Run quality tests and confirm no gate exists**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_parse_quality tests.test_audit_parser_baseline -v`

Expected: FAIL because quality report types and v2 audit fields are absent.

- [ ] **Step 3: Implement reports and fail-closed preprocessing**

```python
report = assess_document(doc_ir)
write_json_atomic(doc_dir / "parse_quality.json", report.to_dict())
if report.status == "failed":
    failures.append({"file": str(file_path), "error": "quality_gate", "reasons": report.failure_reasons})
    continue
```

Extract `process_document()` from the current CLI loop, require its v2 caller to inject `ParseContext`, save version-named immutable directories, write DocIR/report through temporary files, and include v2 aggregate counts in `parser_report.json`. The existing command without catalog context remains an explicitly labeled legacy-profile command and cannot create an activatable v2 snapshot; Tasks 9 and 13 call `process_document()` with catalog identities. Existing outputs stay untouched unless the new candidate completes.

- [ ] **Step 4: Run quality, audit, and phase-0 audit tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_parse_quality tests.test_audit_parser_baseline -v`

Expected: PASS; duplicate identities and failed serialization prevent candidate success.

- [ ] **Step 5: Commit the quality gate**

```powershell
git add src/pipeline/quality.py scripts/preprocess.py scripts/audit_parser_baseline.py tests/unit/test_parse_quality.py tests/test_audit_parser_baseline.py
git commit -m "feat: gate document parsing on quality reports"
```

### Task 7: File Intake Validation

**Files:**
- Create: `src/pipeline/file_validation.py`
- Modify: `app/core/config.py`
- Modify: `requirements.txt`
- Test: `tests/unit/test_file_validation.py`

**Interfaces:**
- Produces: `FileLimits(max_bytes, max_zip_entries, max_uncompressed_bytes, max_pages, max_table_rows)`.
- Produces: `validate_input_file(path: Path, limits: FileLimits) -> ValidatedFile`.
- `ValidatedFile` contains `source_sha256`, `size_bytes`, `detected_format`, and sanitized display name.

- [ ] **Step 1: Write mismatch, ZIP-bomb, and safe-file tests**

```python
def test_renamed_executable_is_rejected(self):
    path = self.tmp_path / "report.pdf"
    path.write_bytes(b"MZ" + b"x" * 100)
    with self.assertRaises(FileValidationError) as raised:
        validate_input_file(path, FileLimits())
    self.assertEqual(raised.exception.code, "signature_mismatch")

def test_ooxml_declared_size_limit_is_enforced(self):
    path = self.tmp_path / "huge.docx"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "x" * 200_000)
    with self.assertRaises(FileValidationError) as raised:
        validate_input_file(path, FileLimits(max_uncompressed_bytes=100_000))
    self.assertEqual(raised.exception.code, "decompressed_size_limit")
```

Also test PDF magic bytes, OOXML required members, macro member rejection, oversized file rejection before hashing, and display-name path stripping.

- [ ] **Step 2: Run validation tests and verify the module is missing**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_file_validation -v`

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement streaming hashes and bounded archive inspection**

```python
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
```

Check extension, signatures, OOXML content types, archive member count and declared expanded size without extraction; reject VBA/macro entries and return stable error codes without absolute paths.

Add `python-multipart==0.0.20` to `requirements.txt` because the admin document API in Task 9 accepts streamed `UploadFile` multipart bodies. Keep upload bytes in a bounded temporary file with a random internal name before validation.

- [ ] **Step 4: Run validation and parser tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_file_validation tests.unit.test_word_parser_v2 tests.unit.test_pdf_parser_v2 -v`

Expected: PASS; safe generated fixtures remain accepted.

- [ ] **Step 5: Commit file validation**

```powershell
git add src/pipeline/file_validation.py app/core/config.py requirements.txt tests/unit/test_file_validation.py
git commit -m "feat: validate document inputs before parsing"
```

### Task 8: Document Catalog Models and State Transitions

**Files:**
- Modify: `app/models/__init__.py`
- Create: `app/db/schema_upgrades.py`
- Modify: `app/db/init_db.py`
- Create: `app/services/document_catalog.py`
- Test: `tests/unit/test_document_catalog.py`

**Interfaces:**
- Produces models: `DocumentRecord`, `DocumentVersionRecord`, `KnowledgeSnapshotRecord`, `EvidenceRecord`.
- Extends: `RunRecord.knowledge_snapshot_id: str | None`.
- Produces: `DocumentCatalog.stage_version()`, `mark_ready()`, `activate()`, `delete_document()`, `rollback()`.
- Contract: all lifecycle methods take an existing `Session`; callers decide the outer transaction boundary.

- [ ] **Step 1: Write deduplication, transition, delete, rollback, and concurrency tests**

```python
def test_same_content_and_config_returns_existing_version(self):
    catalog = DocumentCatalog(self.session)
    first = catalog.stage_version(document_id="doc-1", source_sha256="a", parser_config_hash="c", chunker_version="2")
    second = catalog.stage_version(document_id="doc-1", source_sha256="a", parser_config_hash="c", chunker_version="2")
    assert first.id == second.id
    assert self.session.query(DocumentVersionRecord).count() == 1

def test_delete_hides_document_in_same_transaction(self):
    document = DocumentRecord(id="doc-1", display_name="rules.pdf", status="active")
    version = DocumentVersionRecord(
        id="dv-1", document_id="doc-1", source_sha256="a",
        parser_config_hash="c", chunker_version="2", status="active",
    )
    document.active_version_id = version.id
    self.session.add_all([document, version])
    self.session.commit()
    DocumentCatalog(self.session).delete_document(document.id)
    self.session.commit()
    assert DocumentCatalog(self.session).active_version_ids() == frozenset()
```

Add invalid transition (`staged -> active`), activation of failed quality, rollback to non-ready version, same-name/different-document, and uniqueness-race tests.

- [ ] **Step 2: Run catalog tests and confirm models are absent**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_document_catalog -v`

Expected: FAIL on missing models and catalog service.

- [ ] **Step 3: Implement portable SQLAlchemy records and guarded transitions**

```python
ALLOWED_VERSION_TRANSITIONS = {
    "staged": {"validating", "failed"},
    "validating": {"ready", "failed"},
    "ready": {"active", "deleted"},
    "active": {"superseded", "deleted"},
    "superseded": {"active", "deleted"},
    "failed": set(),
    "deleted": set(),
}
```

Use unique constraints from the spec, JSON text columns compatible with SQLite/PostgreSQL, and an additive schema upgrade that inspects `runs` before adding the nullable snapshot column. The upgrade must be idempotent on two consecutive calls.

- [ ] **Step 4: Run catalog, RunService, and API tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_document_catalog tests.unit.test_run_service tests.integration.test_runs_api -v`

Expected: PASS without changing phase-1 Run behavior while the snapshot field is null.

- [ ] **Step 5: Commit the catalog**

```powershell
git add app/models/__init__.py app/db/schema_upgrades.py app/db/init_db.py app/services/document_catalog.py tests/unit/test_document_catalog.py
git commit -m "feat: add versioned document catalog"
```

### Task 9: Immutable Snapshot Activation and Document Management API

**Files:**
- Create: `src/pipeline/snapshots.py`
- Create: `app/services/document_service.py`
- Create: `app/api/documents.py`
- Modify: `app/schemas/__init__.py`
- Modify: `app/main.py`
- Test: `tests/unit/test_snapshot_activation.py`
- Test: `tests/integration/test_documents_api.py`

**Interfaces:**
- Produces: `SnapshotManifest.build(version_artifacts, config) -> SnapshotManifest`.
- Produces: `VersionArtifact(document_id, document_version, docir_path, block_count, docir_sha256)` and `SnapshotConfig(docir_schema_version, chunker_version, parser_versions_hash)`.
- Produces: `write_manifest_atomic(manifest, path) -> Path` and `read_manifest(path) -> SnapshotManifest`.
- Produces: `DocumentService.ingest()`, `replace()`, `delete()`, `rollback()`.
- Endpoints: admin-only `POST /api/documents`, `POST /api/documents/{id}/versions`, `DELETE /api/documents/{id}`, `POST /api/documents/{id}/rollback/{version_id}`, `GET /api/documents/{id}`.

- [ ] **Step 1: Write canonical manifest and crash-boundary tests**

```python
def test_manifest_id_is_order_independent(self):
    a = VersionArtifact("doc-a", "dv-a", "a.docir.json", 1, "sha-a")
    b = VersionArtifact("doc-b", "dv-b", "b.docir.json", 1, "sha-b")
    config = SnapshotConfig(2, "chunk-v2", "parsers-v2")
    one = SnapshotManifest.build([b, a], config)
    two = SnapshotManifest.build([a, b], config)
    assert one.snapshot_id == two.snapshot_id

def test_activation_failure_keeps_previous_snapshot(self):
    document = DocumentRecord(id="doc-new", display_name="new.txt", status="active")
    version = DocumentVersionRecord(
        id="dv-new", document_id="doc-new", source_sha256="new",
        parser_config_hash="cfg", chunker_version="chunk-v2", status="ready",
        docir_path="new.docir.json", quality_report_path="quality.json",
    )
    old = KnowledgeSnapshotRecord(
        id="snap-old", manifest_path="old.json", status="active",
        document_versions_json="[]", parser_schema_version=2,
        chunker_version="chunk-v2", block_count=0, manifest_sha256="old",
    )
    self.session.add_all([document, version, old])
    self.session.commit()
    service = DocumentService(self.session, artifact_root=self.tmp_path)
    with patch("app.services.document_service.write_manifest_atomic", side_effect=OSError("disk full")):
        with self.assertRaises(DocumentServiceError) as raised:
            service.activate_ready_version("dv-new")
    self.assertEqual(raised.exception.code, "snapshot_write_failed")
    self.session.rollback()
    assert DocumentCatalog(self.session).active_snapshot().id == old.id
```

API tests must prove a normal user gets 403, an admin can create/status/delete, byte-identical replacement returns the existing version, and raw absolute paths never appear in an error response.

- [ ] **Step 2: Run snapshot and document API tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_snapshot_activation tests.integration.test_documents_api -v`

Expected: FAIL because manifests, orchestration, schemas, and router are absent.

- [ ] **Step 3: Implement atomic artifacts and transaction-first activation**

```python
candidate = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
candidate.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
os.replace(candidate, manifest_path)

with session.begin_nested():
    catalog.activate(version_id, manifest.snapshot_id, str(manifest_path))
```

The service sequence is validate → stage → parse → quality gate → write immutable DocIR/report/manifest → create candidate DB row → build/check index callback → activate in one DB transaction. Failed parse/index/commit preserves the old active snapshot and candidate files remain unreferenced for later cleanup.

- [ ] **Step 4: Run snapshot, API, validation, and catalog tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_snapshot_activation tests.integration.test_documents_api tests.unit.test_file_validation tests.unit.test_document_catalog -v`

Expected: PASS for create, update, delete, rollback, permission, retry, and crash boundaries.

- [ ] **Step 5: Commit snapshot lifecycle and API**

```powershell
git add src/pipeline/snapshots.py app/services/document_service.py app/api/documents.py app/schemas/__init__.py app/main.py tests/unit/test_snapshot_activation.py tests/integration/test_documents_api.py
git commit -m "feat: activate immutable knowledge snapshots"
```

### Task 10: Snapshot Retrieval, Cache Fingerprints, and Immediate Visibility

**Files:**
- Create: `src/system/index_fingerprint.py`
- Create: `app/services/document_visibility.py`
- Modify: `src/system/retrieval.py:124-318,381-430,837-912`
- Test: `tests/unit/test_snapshot_retrieval.py`

**Interfaces:**
- Produces: frozen `IndexFingerprint.cache_key() -> str` and `matches(metadata) -> bool`.
- Produces protocol: `VersionVisibility.allowed(version_id: str) -> bool`.
- Changes: `KnowledgeStore(manifest_path).load()` reads DocIR v2 blocks from a snapshot manifest.
- Changes: retriever `search(query, top_k, visibility=None)` over-fetches once, filters forbidden versions, and returns at most `top_k` allowed blocks.
- Produces: `filter_visible_candidates(candidates, top_k, visibility) -> list[dict]` so stale-index filtering is deterministic and unit-testable without loading an embedding model.

- [ ] **Step 1: Write table ingestion, fingerprint mismatch, and stale-index deletion tests**

```python
@dataclass(frozen=True)
class StaticVisibility:
    versions: frozenset[str]

    def allowed(self, version_id: str) -> bool:
        return version_id in self.versions

def write_table_docir(path, version, text):
    doc = DocIR(
        doc_id="doc-table", source_path="table.xlsx", source_format=".xlsx",
        document_version=version, schema_version=2,
    )
    doc.add_page(DocPage(page_no=None, blocks=[DocBlock(
        block_id="blk-table", block_type="table_rows", text=text, page=None,
        structural_locator="sheet/S1/rows/000001-000001",
    )]))
    path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")

def write_test_manifest(path, docir_path, version):
    artifact = VersionArtifact(
        "doc-table", version, str(docir_path), 1, sha256_file(docir_path)
    )
    manifest = SnapshotManifest.build(
        [artifact], SnapshotConfig(2, "chunk-v2", "parsers-v2")
    )
    write_manifest_atomic(manifest, path)

def test_table_row_text_is_loaded_from_snapshot(self):
    docir_path = self.tmp_path / "table.docir.json"
    write_table_docir(docir_path, version="dv-table", text="[列] 项目 | 金额\n[行 1] 现金 | 10")
    manifest_path = self.tmp_path / "snapshot.json"
    write_test_manifest(manifest_path, docir_path, version="dv-table")
    blocks = KnowledgeStore(manifest_path).load()
    assert blocks[0]["type"] == "table_rows"
    assert "现金" in blocks[0]["content"]

def test_deleted_version_is_filtered_even_when_index_contains_it(self):
    candidates = [
        {"document_version": "dv-deleted", "score": 0.9},
        {"document_version": "dv-active", "score": 0.8},
        {"document_version": "dv-active-2", "score": 0.7},
    ]
    visibility = StaticVisibility(frozenset({"dv-active", "dv-active-2"}))
    result = filter_visible_candidates(candidates, top_k=2, visibility=visibility)
    assert [item["document_version"] for item in result] == ["dv-active", "dv-active-2"]

def test_parser_version_change_invalidates_cache(self):
    base = dict(
        snapshot_id="snap-a", embedding_model="BAAI/bge-m3", embedding_revision="main",
        embedding_dimension=1024, normalization_method="l2", distance_metric="ip",
        docir_schema_version=2, chunker_version="chunk-v2", retrieval_channel="text",
        demo_subset_hash=None,
    )
    assert IndexFingerprint(parser_versions_hash="a", **base).cache_key() != IndexFingerprint(
        parser_versions_hash="b", **base
    ).cache_key()
```

Also test missing manifest files, manifest hash mismatch, block-ID digest mismatch, model revision, normalization, dimension, metric, chunker, and demo subset differences.

- [ ] **Step 2: Run retrieval tests against the legacy loader**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_snapshot_retrieval -v`

Expected: FAIL because the loader accepts only `knowledge_base.json` and cache metadata lacks a fingerprint.

- [ ] **Step 3: Implement v2 snapshot loading and fail-closed cache reuse**

```python
if not fingerprint.matches(saved_meta["fingerprint"]):
    return False
if saved_meta["block_ids_sha256"] != digest_block_ids(metadata):
    return False
```

Keep `LegacyKnowledgeStore` behind an explicit legacy profile, include all Evidence locator/version fields in metadata, write cache metadata atomically, filter after every retrieval channel, and over-fetch with `candidate_k=min(index_size, max(top_k * 4, top_k))` once.

- [ ] **Step 4: Run retrieval, baseline benchmark, and parser tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_snapshot_retrieval tests.test_benchmark_regression tests.unit.test_word_parser_v2 tests.unit.test_pdf_parser_v2 -v`

Expected: PASS; legacy comparison helpers continue to accept phase-1 reports.

- [ ] **Step 5: Commit snapshot-aware retrieval**

```powershell
git add src/system/index_fingerprint.py app/services/document_visibility.py src/system/retrieval.py tests/unit/test_snapshot_retrieval.py
git commit -m "feat: enforce snapshot visibility in retrieval"
```

### Task 11: Pin Knowledge Snapshots to Runs and Persist Evidence Identity

**Files:**
- Modify: `src/contracts.py:61-81,124-143`
- Modify: `app/services/run_service.py`
- Modify: `app/services/qa_runtime.py`
- Modify: `src/system/qa_engine.py`
- Modify: `app/schemas/__init__.py`
- Test: `tests/unit/test_run_knowledge_snapshot.py`
- Modify: `tests/unit/test_run_service.py`
- Modify: `tests/integration/test_runs_api.py`

**Interfaces:**
- Produces: `KnowledgeSnapshotRef(snapshot_id, manifest_path)`.
- Produces: `evidence_from_v2(raw: dict, snapshot_id: str) -> Evidence`.
- Changes: `QARuntime.answer(question, question_id, knowledge_snapshot) -> dict`.
- Extends: `RunResult.knowledge_snapshot_id` and `RunOut.knowledge_snapshot_id`.
- Contract: `RunRecord.knowledge_snapshot_id` is committed before `qa_runtime.answer()`.

- [ ] **Step 1: Write ordering, immutable evidence ID, and snapshot-switch tests**

```python
class FakeModelRegistry:
    def __init__(self, calls):
        self.calls = calls

    def snapshot(self):
        self.calls.append("model_snapshot")
        return ModelSnapshot(
            snapshot_id="model-a", profile_name="stable-v1", provider="fake"
        )

class FakeKnowledgeRegistry:
    def __init__(self, calls):
        self.calls = calls

    def active_ref(self, db):
        self.calls.append("knowledge_snapshot")
        return KnowledgeSnapshotRef("snap-a", "snapshot.json")

class RecordingQA:
    def __init__(self, session, calls):
        self.session = session
        self.calls = calls

    def answer(self, question, question_id, knowledge_snapshot):
        record = self.session.get(RunRecord, question_id)
        assert record.knowledge_snapshot_id == "snap-a"
        self.calls.append("answer:snap-a")
        return {"answer": "答案", "evidence": [], "confidence": 1.0}

def test_knowledge_snapshot_is_committed_before_inference(self):
    calls = []
    service = RunService(
        qa_runtime=RecordingQA(self.session, calls),
        registry=FakeModelRegistry(calls),
        knowledge_registry=FakeKnowledgeRegistry(calls),
    )
    result = service.create_and_execute(
        self.session, user_id=1, request=RunCreateRequest(question="问题")
    )
    assert calls == ["model_snapshot", "knowledge_snapshot", "answer:snap-a"]
    assert result.knowledge_snapshot_id == "snap-a"

def test_evidence_id_includes_snapshot_and_version(self):
    raw = {
        "document_id": "doc-a", "document_version": "dv-a", "block_id": "blk-a",
        "content": "依据", "source_name": "rules.pdf", "page": 1,
        "structural_locator": "page/000001/text/000001",
    }
    one = evidence_from_v2(raw, "snap-a")
    two = evidence_from_v2(raw, "snap-b")
    assert one.evidence_id != two.evidence_id
```

Add a test where the active catalog changes after Run creation: the in-flight Run must continue using its persisted snapshot while the next Run uses the new snapshot. Add a failed-final-commit assertion that no EvidenceRecord is published as completed.

- [ ] **Step 2: Run Run snapshot tests against the phase-1 signature**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_run_knowledge_snapshot tests.unit.test_run_service tests.integration.test_runs_api -v`

Expected: FAIL because Runs do not resolve or expose a knowledge snapshot.

- [ ] **Step 3: Implement snapshot-pinned runtime and EvidenceRecord persistence**

```python
knowledge = self.knowledge_registry.active_ref(db)
record = RunRecord(
    id=run_id,
    knowledge_snapshot_id=knowledge.snapshot_id,
    model_snapshot_json=model_snapshot.model_dump_json(),
    status=RunStatus.PENDING.value,
    question=question,
    user_id=user_id,
)
db.add(record)
db.commit()
raw = self.qa_runtime.answer(question, run_id, knowledge)
```

Make QARuntime rebuild only when the requested snapshot differs from its loaded snapshot, pass the manifest path to `FinancialQASystem`, create `ev_<sha256(snapshot|version|block)>`, persist EvidenceRecord rows in the same final transaction as completed status, and keep the legacy adapter only for the legacy profile.

- [ ] **Step 4: Run all Run, API, contract, and snapshot retrieval tests**

Run: `& '.\.conda\python.exe' -m unittest tests.unit.test_run_knowledge_snapshot tests.unit.test_run_service tests.integration.test_runs_api tests.unit.test_contracts tests.unit.test_snapshot_retrieval -v`

Expected: PASS; `/chat` keeps its shape and `/runs` adds the snapshot field compatibly.

- [ ] **Step 5: Commit Run snapshot pinning**

```powershell
git add src/contracts.py app/services/run_service.py app/services/qa_runtime.py src/system/qa_engine.py app/schemas/__init__.py tests/unit/test_run_knowledge_snapshot.py tests/unit/test_run_service.py tests/integration/test_runs_api.py
git commit -m "feat: pin knowledge snapshots to runs"
```

### Task 12: Authorized Evidence Reader and API

**Files:**
- Create: `app/services/evidence_reader.py`
- Create: `app/api/evidence.py`
- Modify: `app/schemas/__init__.py`
- Modify: `app/main.py`
- Test: `tests/integration/test_evidence_api.py`

**Interfaces:**
- Produces: `EvidenceReader.read(db, evidence_id, requesting_user) -> EvidenceDetail`.
- Endpoint: `GET /api/evidence/{evidence_id}`.
- Contract: Run owner can read current evidence; admin can read historical deleted-version evidence; other users receive 404.

- [ ] **Step 1: Write PDF, Word, ownership, and tamper tests**

```python
def test_owner_reads_word_structure_without_fake_page(self):
    response = self.client.get(
        f"/api/evidence/{self.word_evidence.id}", headers=self.owner_headers
    )
    assert response.status_code == 200
    assert response.json()["page_no"] is None
    assert response.json()["structure_path"] == "第一章/body/000003/paragraph"

def test_other_user_cannot_probe_evidence_id(self):
    response = self.client.get(
        f"/api/evidence/{self.word_evidence.id}", headers=self.other_headers
    )
    assert response.status_code == 404
```

Add PDF bbox/page, admin historical access after deletion, missing DocIR file, block not belonging to version, and DocIR block text tampering tests.

- [ ] **Step 2: Run evidence API tests**

Run: `& '.\.conda\python.exe' -m unittest tests.integration.test_evidence_api -v`

Expected: FAIL because no evidence reader or route exists.

- [ ] **Step 3: Implement exact version/block lookup and authorization**

```python
record = catalog.find_evidence_for_user(evidence_id, requesting_user.id, requesting_user.role)
doc = load_docir(record.document_version.docir_path)
block = doc.require_block(record.block_id)
if block.block_id != record.block_id or doc.document_version != record.document_version_id:
    raise EvidenceIntegrityError("evidence_identity_mismatch")
```

Return source name, exact text/table rows, page/bbox or section/element locator, and quality flags. Map missing/unauthorized to 404, integrity failure to a stable 409 response, and never return the storage path.

- [ ] **Step 4: Run evidence, document, and runs API suites**

Run: `& '.\.conda\python.exe' -m unittest tests.integration.test_evidence_api tests.integration.test_documents_api tests.integration.test_runs_api -v`

Expected: PASS for owner/admin access and all denial/integrity cases.

- [ ] **Step 5: Commit evidence reading**

```powershell
git add app/services/evidence_reader.py app/api/evidence.py app/schemas/__init__.py app/main.py tests/integration/test_evidence_api.py
git commit -m "feat: expose authorized evidence locations"
```

### Task 13: Side-by-Side Migration, Full Audit, and Phase Gate

**Files:**
- Create: `scripts/migrate_phase2_knowledge.py`
- Create: `docs/baseline/phase-2-runbook.md`
- Modify: `scripts/benchmark_regression.py`
- Test: `tests/test_phase2_migration.py`
- Test: `tests/test_benchmark_regression.py`

**Interfaces:**
- Produces CLI with required `--input`, `--candidate-dir`, and `--database-url` values plus the optional boolean `--activate` flag.
- Produces: `MigrationOptions(input_dir, candidate_dir, database_url, activate)` and `run_migration(options) -> MigrationReport` for testable CLI logic.
- Contract: the default command builds/audits a candidate and never activates; `--activate` is accepted only after all hard gates pass.
- Produces: `output/phase2/$phase2SnapshotId/migration_report.json` and `phase2_regression.json`, where the variable is read from the generated report before comparison.

- [ ] **Step 1: Write dry-run, resume, gate, and activation tests**

```python
def test_migration_default_never_switches_active_snapshot(self):
    input_dir = self.tmp_path / "corpus"
    input_dir.mkdir()
    (input_dir / "rules.txt").write_text("规则正文", encoding="utf-8")
    result = run_migration(MigrationOptions(
        input_dir=input_dir,
        candidate_dir=self.tmp_path / "candidate",
        database_url=self.database_url,
        activate=False,
    ))
    assert result.candidate_snapshot_id
    assert DocumentCatalog(self.session).active_snapshot().id == "snap-old"

def test_failed_quality_gate_rejects_activate(self):
    options = MigrationOptions(
        input_dir=self.tmp_path / "corpus",
        candidate_dir=self.tmp_path / "candidate",
        database_url=self.database_url,
        activate=True,
    )
    failed = SnapshotQualityReport(status="failed", hard_failure_count=1)
    with patch("scripts.migrate_phase2_knowledge.assess_snapshot", return_value=failed):
        with self.assertRaises(MigrationGateError) as raised:
            run_migration(options)
    self.assertEqual(raised.exception.code, "candidate_quality_failed")
```

Add resume-from-completed-version, stale temporary cleanup, unchanged old FAISS files, regression-report explanation field, and rollback-profile tests.

- [ ] **Step 2: Run migration and benchmark tests**

Run: `& '.\.conda\python.exe' -m unittest tests.test_phase2_migration tests.test_benchmark_regression -v`

Expected: FAIL because the migration CLI and phase-2 comparison fields do not exist.

- [ ] **Step 3: Implement resumable migration and explicit activation gate**

```python
if report.hard_failure_count:
    raise MigrationGateError("candidate_quality_failed")
if args.activate:
    document_service.activate_candidate(report.snapshot_id)
```

Write one immutable version directory per source, skip only when source hash plus parser/chunker configuration match, aggregate counts without loading all block IDs into RAM, compare the fixed phase-0 questions, and require a textual reason for each changed answer/evidence set before the report can pass.

- [ ] **Step 4: Run the complete automated suite and compilation check**

Run: `& '.\.conda\python.exe' -m unittest discover -s tests -v`

Expected: all tests PASS.

Run: `& '.\.conda\python.exe' -m compileall -q app src scripts tests`

Expected: exit code 0 with no syntax errors.

- [ ] **Step 5: Build and audit the real candidate without activation**

Run:

```powershell
& '.\.conda\python.exe' scripts/migrate_phase2_knowledge.py `
  --input 'data/documents' `
  --candidate-dir 'output/phase2' `
  --database-url 'sqlite:///C:/code/个人资料/code/Agentic_RAG/finance-qa-platform/data_app.db'
```

Expected: a candidate report with `duplicate_block_id_count=0`, `missing_locator_count=0`, `empty_block_count=0`, no active snapshot change, and old baseline files unchanged.

- [ ] **Step 6: Run the fixed phase-0/phase-1 regression against the candidate**

Run:

```powershell
$phase2Report = Get-ChildItem 'output/phase2' -Filter 'migration_report.json' -Recurse |
  Sort-Object LastWriteTime | Select-Object -Last 1
$phase2SnapshotId = Split-Path $phase2Report.DirectoryName -Leaf
& '.\.conda\python.exe' scripts/benchmark_regression.py `
  --baseline 'output/baseline/phase1-smoke.json' `
  --candidate "output/phase2/$phase2SnapshotId/qa_regression.json" `
  --output "output/phase2/$phase2SnapshotId/phase2_regression.json"
```

Expected: zero unexplained answer/evidence regressions; parser-correctness changes are listed with their document version and locator reason.

- [ ] **Step 7: Activate the audited snapshot and run deletion/rollback smoke tests**

Run:

```powershell
& '.\.conda\python.exe' scripts/migrate_phase2_knowledge.py `
  --input 'data/documents' `
  --candidate-dir 'output/phase2' `
  --database-url 'sqlite:///C:/code/个人资料/code/Agentic_RAG/finance-qa-platform/data_app.db' `
  --activate
```

Expected: the active snapshot changes atomically; a selected document is retrievable before deletion, absent from the next new Run after deletion, readable through its historical Evidence as admin, and recoverable by rollback.

- [ ] **Step 8: Document measured results and rollback commands**

Record exact snapshot IDs, document/block/table/OCR counts, quality warnings, regression changes, peak memory, index build time, deletion test, rollback test, and legacy profile command in `docs/baseline/phase-2-runbook.md`.

- [ ] **Step 9: Commit migration tooling and the verified runbook**

```powershell
git add scripts/migrate_phase2_knowledge.py scripts/benchmark_regression.py tests/test_phase2_migration.py tests/test_benchmark_regression.py docs/baseline/phase-2-runbook.md
git commit -m "docs: record phase two migration gate"
```

### Task 14: Final Compatibility and Branch Review

**Files:**
- Review: all files changed by Tasks 1-13
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: a reviewable phase-2 branch with no unexplained baseline regression and a tested rollback path.

- [ ] **Step 1: Verify repository cleanliness and intended scope**

Run: `git status --short`

Expected: only pre-existing unrelated untracked interview files/scripts remain; no generated corpus, database, cache, or secret is staged.

- [ ] **Step 2: Run the phase-2 focused suite**

Run:

```powershell
& '.\.conda\python.exe' -m unittest `
  tests.unit.test_docir_v2 `
  tests.unit.test_table_chunking `
  tests.unit.test_word_parser_v2 `
  tests.unit.test_pdf_parser_v2 `
  tests.unit.test_office_parsers_v2 `
  tests.unit.test_parse_quality `
  tests.unit.test_file_validation `
  tests.unit.test_document_catalog `
  tests.unit.test_snapshot_activation `
  tests.unit.test_snapshot_retrieval `
  tests.unit.test_run_knowledge_snapshot `
  tests.integration.test_documents_api `
  tests.integration.test_evidence_api -v
```

Expected: PASS.

- [ ] **Step 3: Run full finance and reference-Agent verification**

Run: `& '.\.conda\python.exe' -m unittest discover -s tests -v`

Run from the Agent reference project using its working environment: `python -m unittest discover -s graph/tests -v`

Expected: finance suite and the three offline Agent graph tests PASS.

- [ ] **Step 4: Inspect diff and history**

Run: `git diff main...HEAD --check`

Run: `git log --oneline main..HEAD`

Expected: no whitespace errors; one focused commit per completed task; no unrelated files.

- [ ] **Step 5: Perform whole-branch review and fix only verified findings**

Review these failure boundaries: identity determinism, parser ordering, quality fail-closed behavior, activation transaction, delete visibility, Run snapshot ordering, evidence authorization, cache compatibility, migration rollback. For each finding, add a failing regression test, make the minimum fix, rerun its focused suite, and commit with `fix: ...`.

- [ ] **Step 6: Present measured acceptance evidence before merge**

Report test counts, candidate snapshot ID, zero-duplicate/zero-missing-locator audit, fixed-question regression result, deletion latency semantics, historical Evidence lookup, and rollback result. Merge only after these concrete results are available.
