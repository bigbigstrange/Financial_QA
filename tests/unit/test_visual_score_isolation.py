import unittest
from src.system.retrieval import FusionRetriever, HybridRetriever


class StaticRetriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, *args, **kwargs):
        return [dict(row) for row in self.rows]


class VisualScoreIsolationTests(unittest.TestCase):
    def test_image_score_never_overwrites_text_similarity(self):
        text = {"block_id": "shared", "score": 0.2, "retrieval_channel": "text"}
        visual = {"block_id": "shared", "score": 0.9, "image_score": 0.9, "retrieval_channel": "image_chinese_clip",
                  "page_id": "vp_2", "image_sha256": "digest", "vision_page": 2}
        fusion = FusionRetriever(StaticRetriever([text]), StaticRetriever([visual]))
        row = HybridRetriever(fusion_retriever=fusion).search("query")[0]
        self.assertEqual(row["dense_score"], 0.2)
        self.assertEqual(row["image_score"], 0.9)
        self.assertEqual(row['page_id'], 'vp_2')
        self.assertEqual(row['image_sha256'], 'digest')
        self.assertIn("image_chinese_clip", row["retrieval_channels"])

    def test_image_only_hit_has_no_dense_score(self):
        visual = {"block_id": "image", "score": 0.9, "image_score": 0.9, "retrieval_channel": "image_chinese_clip"}
        fusion = FusionRetriever(StaticRetriever([]), StaticRetriever([visual]))
        row = HybridRetriever(fusion_retriever=fusion).search("query")[0]
        self.assertNotIn("dense_score", row)
