import unittest

from scripts.eval_retrieval_phase3 import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_at_k_counts_available_gold(self):
        ranked = ["a", "b", "c", "d", "e"]
        gold = {"b", "e"}

        self.assertEqual(recall_at_k(ranked, gold, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, gold, 5), 1.0)

    def test_mrr_uses_first_gold_position(self):
        ranked = ["a", "b", "c", "d"]
        self.assertEqual(mean_reciprocal_rank(ranked, {"c"}), 1.0 / 3.0)
        self.assertEqual(mean_reciprocal_rank(ranked, {"x"}), 0.0)

    def test_ndcg_rewards_early_hits(self):
        early = ["gold1", "other", "gold2", "other", "other", "other", "other", "other", "other", "other"]
        late = ["other", "other", "gold1", "other", "gold2", "other", "other", "other", "other", "other"]

        self.assertGreater(ndcg_at_k(early, {"gold1", "gold2"}, 10), ndcg_at_k(late, {"gold1", "gold2"}, 10))


if __name__ == "__main__":
    unittest.main()
