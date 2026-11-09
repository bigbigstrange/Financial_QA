import json
import tempfile
import unittest
from pathlib import Path

from app.services.model_registry import ModelRegistry


class ModelRegistrySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "registry.json"
        self.path.write_text(
            json.dumps(
                {
                    "active_profile": "stable-v1",
                    "profiles": {
                        "stable-v1": {
                            "provider": "deepseek",
                            "requested_model": "deepseek-chat",
                            "actual_model": "deepseek-chat",
                            "generation_parameters": {
                                "temperature": 0.2,
                                "max_tokens": 1024,
                            },
                            "embedding_model": "BAAI/bge-m3",
                            "reranker_model": None,
                            "prompt_versions": {"answer": "legacy-v1"},
                            "retrieval_config": {"top_k": 12},
                            "parser_config": {"docir": "legacy-v1"},
                        }
                    },
                    "gates": {},
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.registry = ModelRegistry(str(self.path))

    def test_snapshot_is_stable_and_explicit(self):
        snapshot = self.registry.snapshot("stable-v1", evaluation_run_id=7)

        self.assertEqual(snapshot.profile_name, "stable-v1")
        self.assertEqual(snapshot.embedding_model, "BAAI/bge-m3")
        self.assertEqual(snapshot.evaluation_run_id, 7)
        self.assertTrue(snapshot.snapshot_id.startswith("sha256:"))
        same = self.registry.snapshot("stable-v1", evaluation_run_id=7)
        self.assertEqual(snapshot.snapshot_id, same.snapshot_id)

    def test_returned_snapshot_does_not_change_with_registry(self):
        snapshot = self.registry.snapshot("stable-v1")
        data = self.registry.load()
        data["profiles"]["stable-v1"]["requested_model"] = "changed"
        self.registry.save(data)

        self.assertEqual(snapshot.requested_model, "deepseek-chat")
        self.assertNotEqual(
            snapshot.snapshot_id,
            self.registry.snapshot("stable-v1").snapshot_id,
        )

    def test_candidate_cannot_be_promoted_with_missing_eval_record(self):
        self.registry.upsert_profile("candidate", {"provider": "ark"})
        with self.assertRaises(ValueError):
            self.registry.promote("candidate", eval_run_id=999999999)
        self.assertEqual(self.registry.active_profile_name(), "stable-v1")


if __name__ == "__main__":
    unittest.main()
