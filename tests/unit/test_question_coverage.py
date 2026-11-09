"""The uploaded question sheet is an inventory, not an answer gold set."""

import json

from openpyxl import Workbook

from scripts.audit_question_coverage import audit


def test_duplicate_ids_and_questions_remain_distinct_sheet_rows(tmp_path):
    workbook_path = tmp_path / "questions.xlsx"
    chunks_path = tmp_path / "chunks.jsonl"
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append(["question_id", "question"])
    sheet.append(["q1", "掌上银行基金认购如何操作？"])
    sheet.append(["q1", "掌上银行基金认购如何操作？"])
    sheet.append(["q2", "基金风险是什么？"])
    sheet.append(["empty", None])
    book.save(workbook_path)
    chunks_path.write_text("\n".join([
        json.dumps({"content": "掌上银行客户数增长"}, ensure_ascii=False),
        json.dumps({"content": "交通银行基金认购流程"}, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")

    report = audit(workbook_path, chunks_path)

    assert report["rows"] == 3
    assert report["unique_question_ids"] == 2
    assert report["unique_questions"] == 2
    assert report["duplicate_id_rows"] == 1
    assert report["duplicate_question_rows"] == 1
    assert [item["sheet_row"] for item in report["question_rows"]] == [2, 3, 4]
    assert report["app_phrase_chunks"] == 1
    assert report["app_and_fund_chunks"] == 0
    assert report["answer_accuracy"] == "not_evaluated"
