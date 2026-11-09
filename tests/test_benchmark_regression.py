import unittest

from scripts.benchmark_regression import _FailedQA, compare_reports, run_cases


class FakeQA:
    def answer_question(self, question, question_id):
        if question == "boom":
            raise RuntimeError("offline failure")
        return {
            "answer": "基于证据的答案",
            "confidence": 0.8,
            "evidence": [
                {
                    "block_id": "doc-v1-b1",
                    "source_doc": "demo.pdf",
                    "page": 2,
                    "score": 0.7,
                    "text": "原始证据正文",
                }
            ],
        }


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class BenchmarkRegressionTests(unittest.TestCase):
    def test_compare_reports_counts_changes_without_accuracy_claims(self):
        before = {
            "cases": [
                {
                    "question_id": "1",
                    "answer": "相同",
                    "evidence": [{"block_id": "b1"}],
                    "status": "completed",
                    "latency_sec": 1.0,
                },
                {
                    "question_id": "2",
                    "answer": "旧答案",
                    "evidence": [{"block_id": "b2"}],
                    "status": "completed",
                    "latency_sec": 2.0,
                },
                {
                    "question_id": "3",
                    "answer": "相同",
                    "evidence": [{"block_id": "b3"}],
                    "status": "completed",
                    "latency_sec": 3.0,
                },
            ]
        }
        after = {
            "cases": [
                {
                    "question_id": "1",
                    "answer": "相同",
                    "evidence": [{"evidence_id": "b1"}],
                    "status": "completed",
                    "latency_sec": 1.2,
                },
                {
                    "question_id": "2",
                    "answer": "新答案",
                    "evidence": [{"evidence_id": "b2"}],
                    "status": "completed",
                    "latency_sec": 1.8,
                },
                {
                    "question_id": "3",
                    "answer": "相同",
                    "evidence": [{"evidence_id": "changed"}],
                    "status": "completed",
                    "latency_sec": 3.1,
                },
            ]
        }

        report = compare_reports(before, after)
        summary = report["summary"]

        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["answer_changed_count"], 1)
        self.assertEqual(summary["evidence_id_changed_count"], 1)
        self.assertEqual(summary["new_error_count"], 0)
        self.assertNotIn("accuracy", summary)

    def test_compare_reports_rejects_mismatched_question_ids(self):
        before = {"cases": [{"question_id": "1"}]}
        after = {"cases": [{"question_id": "2"}]}

        with self.assertRaisesRegex(ValueError, "question_id sets differ"):
            compare_reports(before, after)

    def test_compare_reports_prefers_shared_block_id_during_migration(self):
        before = {
            "cases": [
                {
                    "question_id": "1",
                    "evidence": [{"block_id": "stable-block"}],
                }
            ]
        }
        after = {
            "cases": [
                {
                    "question_id": "1",
                    "evidence": [
                        {
                            "evidence_id": "legacy:new-wrapper-id",
                            "block_id": "stable-block",
                        }
                    ],
                }
            ]
        }

        report = compare_reports(before, after)

        self.assertEqual(report["summary"]["evidence_id_changed_count"], 0)

    def test_empty_cases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test set is empty"):
            run_cases(FakeQA(), [], clock=SequenceClock([]), rss_bytes=SequenceClock([]))

    def test_required_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "question_id and question"):
            run_cases(
                FakeQA(),
                [{"question": "q"}],
                clock=SequenceClock([]),
                rss_bytes=SequenceClock([]),
            )

    def test_blank_or_nan_required_values_are_rejected(self):
        for row in (
            {"question_id": "", "question": "q"},
            {"question_id": "1", "question": float("nan")},
        ):
            with self.subTest(row=row), self.assertRaisesRegex(
                ValueError, "non-empty question_id and question"
            ):
                run_cases(
                    FakeQA(),
                    [row],
                    clock=SequenceClock([1.0, 1.1]),
                    rss_bytes=SequenceClock([100, 100]),
                )

    def test_report_preserves_answer_and_evidence_without_calling_it_accuracy(self):
        report = run_cases(
            FakeQA(),
            [{"question_id": "1", "question": "q"}],
            clock=SequenceClock([1.0, 1.25]),
            rss_bytes=SequenceClock([100, 150]),
        )
        self.assertEqual(report["metrics"]["evidence_response_rate"], 1.0)
        self.assertNotIn("success_rate", report["metrics"])
        self.assertEqual(report["cases"][0]["answer"], "基于证据的答案")
        self.assertEqual(report["cases"][0]["evidence"][0]["block_id"], "doc-v1-b1")
        self.assertEqual(report["metrics"]["max_observed_rss_bytes"], 150)

    def test_one_case_failure_is_recorded_and_run_continues(self):
        report = run_cases(
            FakeQA(),
            [
                {"question_id": "1", "question": "boom"},
                {"question_id": "2", "question": "q"},
            ],
            clock=SequenceClock([1.0, 1.1, 2.0, 2.2]),
            rss_bytes=SequenceClock([100, 120, 120, 180]),
        )
        self.assertEqual(report["metrics"]["error_count"], 1)
        self.assertEqual(report["cases"][0]["error_type"], "RuntimeError")
        self.assertEqual(report["cases"][1]["status"], "completed")

    def test_initialization_failure_is_recorded_for_each_case(self):
        report = run_cases(
            _FailedQA(OSError("model cache unavailable")),
            [{"question_id": "1", "question": "q"}],
            clock=SequenceClock([1.0, 1.1]),
            rss_bytes=SequenceClock([100, 100]),
        )

        self.assertEqual(report["metrics"]["error_count"], 1)
        self.assertEqual(report["cases"][0]["status"], "error")
        self.assertIn("QA initialization failed", report["cases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
