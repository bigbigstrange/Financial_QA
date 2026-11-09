"""Read-only inventory audit for question workbook and current text chunks.

This is not an answer-accuracy evaluation: the workbook has no gold answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import load_workbook


def audit(workbook_path: Path, chunks_path: Path) -> dict:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheet = workbook["Sheet1"]
        header = next(sheet.iter_rows(max_row=1, values_only=True))
        if tuple(header[:2]) != ("question_id", "question"):
            raise ValueError("Sheet1 must start with question_id, question")
        rows = [
            (number, str(values[0] or ""), str(values[1]).strip())
            for number, values in enumerate(
                sheet.iter_rows(min_row=2, values_only=True), start=2
            )
            if len(values) >= 2 and values[1] is not None and str(values[1]).strip()
        ]
    finally:
        workbook.close()

    app_hits = 0
    app_fund_hits = 0
    chunk_count = 0
    with chunks_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            chunk_count += 1
            content = str(json.loads(line).get("content") or "")
            if "掌上银行" in content:
                app_hits += 1
                if "基金" in content:
                    app_fund_hits += 1

    ids = {question_id for _, question_id, _ in rows}
    questions = {question for _, _, question in rows}
    return {
        "rows": len(rows),
        "unique_question_ids": len(ids),
        "unique_questions": len(questions),
        "duplicate_id_rows": len(rows) - len(ids),
        "duplicate_question_rows": len(rows) - len(questions),
        "question_rows": [
            {"sheet_row": number, "question_id": question_id, "question": question}
            for number, question_id, question in rows
        ],
        "text_chunks_scanned": chunk_count,
        "app_phrase_chunks": app_hits,
        "app_and_fund_chunks": app_fund_hits,
        "answer_accuracy": "not_evaluated",
        "note": "Workbook has no gold answers or gold evidence; source coverage is not answer accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.workbook, args.chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "question_rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
