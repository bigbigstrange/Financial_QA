"""Prepare and validate a human-reviewed, snapshot-bound QA baseline.

The question workbook contains questions only. This tool deliberately never
generates reference answers or marks a case reviewed on behalf of a person.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = ROOT / "test" / "merged.xlsx"
DEFAULT_MANIFEST = ROOT / "output" / "kb_full" / "kb_v3_manifest.json"
DEFAULT_CHUNKS = ROOT / "output" / "kb_full" / "chunks.jsonl"
DEFAULT_REVIEW = ROOT / "output" / "quality" / "finance500-review.jsonl"
DEFAULT_SUGGESTIONS = ROOT / "output" / "quality" / "representative-evidence-candidates.jsonl"
REVIEWED = {"approved", "refuse", "out_of_scope"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def questions_from_workbook(path: Path) -> list[dict]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        if tuple(next(rows)[:2]) != ("question_id", "question"):
            raise ValueError("题库前两列必须为 question_id、question")
        result = []
        for sheet_row, values in enumerate(rows, start=2):
            if len(values) < 2 or values[1] is None or not str(values[1]).strip():
                continue
            result.append({
                "case_id": f"finance500-r{sheet_row:04d}",
                "sheet_row": sheet_row,
                "question_id": str(values[0]),
                "question": str(values[1]).strip(),
            })
        return result
    finally:
        workbook.close()


def prepare(workbook: Path, manifest: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"审核文件已存在，禁止覆盖人工标注：{output}")
    questions = questions_from_workbook(workbook)
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))["snapshot_id"]
    workbook_sha256 = _sha256(workbook)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for item in questions:
            record = {
                **item,
                "workbook_sha256": workbook_sha256,
                "source_snapshot": snapshot,
                "review_status": "pending_review",
                "reviewer": "",
                "institution": "",
                "product": "",
                "category": "",
                "expected_answer": "",
                "expected_facts": [],
                "expected_refusal": None,
                "gold_evidence_ids": [],
                "allowed_tools": ["knowledge"],
                "review_note": "",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"cases": len(questions), "snapshot": snapshot,
            "workbook_sha256": workbook_sha256, "review_file": str(output)}


def validate(workbook: Path, manifest: Path, chunks: Path, review: Path) -> dict:
    questions = questions_from_workbook(workbook)
    expected = {item["case_id"]: item for item in questions}
    workbook_sha256 = _sha256(workbook)
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))["snapshot_id"]
    evidence = {}
    with chunks.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("snapshot_id") != snapshot:
                continue
            evidence[str(chunk.get("chunk_id"))] = chunk

    errors: list[str] = []
    seen: set[str] = set()
    statuses: Counter[str] = Counter()
    with review.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {line_no} 行 JSON 无效：{exc.msg}")
                continue
            case_id = row.get("case_id")
            source = expected.get(case_id)
            if source is None or case_id in seen:
                errors.append(f"第 {line_no} 行题号不存在或重复：{case_id}")
                continue
            seen.add(case_id)
            if any(row.get(key) != source[key] for key in ("sheet_row", "question_id", "question")):
                errors.append(f"{case_id} 原题被修改")
            if row.get("workbook_sha256") != workbook_sha256 or row.get("source_snapshot") != snapshot:
                errors.append(f"{case_id} 题库或知识快照不匹配")
            status = str(row.get("review_status") or "")
            statuses[status] += 1
            if status not in REVIEWED | {"pending_review"}:
                errors.append(f"{case_id} 未知审核状态：{status}")
                continue
            if status not in REVIEWED:
                continue
            if not str(row.get("reviewer") or "").strip():
                errors.append(f"{case_id} 缺少人工审核人")
            refusal = row.get("expected_refusal")
            if type(refusal) is not bool:
                errors.append(f"{case_id} expected_refusal 必须明确为 true/false")
            if status == "approved" and refusal is True:
                errors.append(f"{case_id} approved 与拒答标签冲突")
            if status == "refuse" and refusal is False:
                errors.append(f"{case_id} refuse 与非拒答标签冲突")
            if status == "approved":
                if not str(row.get("expected_answer") or "").strip():
                    errors.append(f"{case_id} 缺少人工确认的参考答案")
                ids = row.get("gold_evidence_ids")
                if not isinstance(ids, list) or not ids:
                    errors.append(f"{case_id} 缺少标准证据 ID")
                else:
                    for evidence_id in ids:
                        chunk = evidence.get(evidence_id)
                        if chunk is None or not chunk.get("document_version"):
                            errors.append(f"{case_id} 标准证据不在活动快照：{evidence_id}")
            if status == "out_of_scope" and not str(row.get("review_note") or "").strip():
                errors.append(f"{case_id} 排除题目需注明原因")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"缺少 {len(missing)} 道题的审核记录")
    ready = (not errors and len(seen) == len(expected)
             and not statuses.get("pending_review")
             and not statuses.get("out_of_scope"))
    return {
        "workbook_cases": len(expected),
        "unique_questions": len({item["question"] for item in questions}),
        "reviewed_cases": sum(statuses[s] for s in REVIEWED),
        "pending_cases": statuses.get("pending_review", 0),
        "statuses": dict(statuses),
        "source_snapshot": snapshot,
        "quality_gate_ready": ready,
        "errors": errors,
        "note": "结构校验不能代替人工判断参考答案、银行/产品及原文是否正确。",
    }


def suggest(review: Path, chunks: Path, output: Path, *, count: int = 25,
            seed: int = 20260924) -> dict:
    """Offer lexical evidence candidates for human review, never gold labels."""
    if count < 1:
        raise ValueError("count 必须大于 0")
    if output.exists():
        raise FileExistsError(f"候选文件已存在，禁止覆盖：{output}")
    from scripts.build_gold_set import classify_question
    from src.system.sparse import SparseRetriever

    cases = [json.loads(line) for line in review.open(encoding="utf-8") if line.strip()]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        buckets[classify_question(case["question"])].append(case)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    # Include the known named-app failure mode in the first review batch.
    named_app = next((case for case in cases if "掌上银行" in case["question"]), None)
    if named_app is not None:
        selected.append(named_app)
    while len(selected) < min(count, len(cases)):
        advanced = False
        for category in sorted(buckets):
            while buckets[category] and buckets[category][-1] in selected:
                buckets[category].pop()
            if buckets[category] and len(selected) < count:
                selected.append(buckets[category].pop())
                advanced = True
        if not advanced:
            break
    snapshot = selected[0]["source_snapshot"] if selected else ""
    blocks = [json.loads(line) for line in chunks.open(encoding="utf-8") if line.strip()]
    blocks = [block for block in blocks if block.get("snapshot_id") == snapshot]
    retriever = SparseRetriever(blocks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for case in selected:
            hits = retriever.search(case["question"], top_k=8)
            row = {
                "case_id": case["case_id"], "question": case["question"],
                "source_snapshot": snapshot,
                "status": "machine_retrieved_not_gold",
                "warning": ("候选未出现掌上银行，需查明资料覆盖或标记拒答"
                            if "掌上银行" in case["question"] and not any(
                                "掌上银行" in str(hit.get("content") or "")
                                for hit in hits) else ""),
                "candidates": [{
                    "chunk_id": hit.get("chunk_id"),
                    "document_version": hit.get("document_version"),
                    "source_doc": hit.get("source_doc"),
                    "page_start": hit.get("page_start"),
                    "section_path": hit.get("section_path"),
                    "content_excerpt": str(hit.get("content") or "")[:1200],
                } for hit in hits],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"selected_cases": len(selected), "candidate_status": "not_gold",
            "snapshot": snapshot, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "suggest"))
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--suggestions", type=Path, default=DEFAULT_SUGGESTIONS)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.workbook, args.manifest, args.review)
    elif args.action == "suggest":
        result = suggest(args.review, args.chunks, args.suggestions,
                         count=args.count)
    else:
        result = validate(args.workbook, args.manifest, args.chunks, args.review)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "validate" and not result["quality_gate_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
