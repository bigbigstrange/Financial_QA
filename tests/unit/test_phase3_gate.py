import unittest

from scripts.phase3_gate import evaluate_gate


class Phase3GateTests(unittest.TestCase):
    def _report(self, **overrides):
        base = {
            "dense": {"recall_at_10": 0.1},
            "hybrid": {"recall_at_10": 0.5},
            "hybrid+rerank": {
                "recall_at_10": 0.6,
                "mrr": 0.5,
                "ndcg_at_10": 0.5,
            },
        }
        for strategy, values in overrides.items():
            base[strategy].update(values)
        return {"strategies": base}

    def test_gate_passes_when_hybrid_rerank_meets_thresholds(self):
        result = evaluate_gate(self._report())

        self.assertTrue(result["passed"])
        self.assertTrue(all(item["passed"] for item in result["checks"]))

    def test_gate_fails_on_low_recall(self):
        result = evaluate_gate(
            self._report(**{"hybrid+rerank": {"recall_at_10": 0.2}})
        )

        self.assertFalse(result["passed"])
        failed = [item for item in result["checks"] if not item["passed"]]
        self.assertTrue(any(item["metric"] == "recall_at_10" for item in failed))

    def test_missing_strategy_fails_closed(self):
        result = evaluate_gate({"strategies": {"dense": {}}})

        self.assertFalse(result["passed"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
