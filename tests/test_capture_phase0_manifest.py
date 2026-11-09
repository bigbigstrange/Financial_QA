import json
import tempfile
import unittest
from pathlib import Path

from scripts.capture_phase0_manifest import build_manifest, sha256_file


class Phase0ManifestTests(unittest.TestCase):
    def test_hash_is_stable_and_missing_file_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.txt"
            payload.write_text("baseline", encoding="utf-8")

            self.assertEqual(sha256_file(payload), sha256_file(payload))
            manifest = build_manifest(
                finance_root=root,
                agent_root=root / "missing-agent",
                kb_path=payload,
                test_set=root / "missing.xlsx",
                environ={},
            )

            self.assertEqual(manifest["artifacts"]["knowledge_base"]["status"], "present")
            self.assertEqual(manifest["artifacts"]["test_set"]["status"], "missing")

    def test_secret_values_never_appear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "sk-do-not-write-this"
            manifest = build_manifest(
                finance_root=root,
                agent_root=root,
                kb_path=root / "kb.json",
                test_set=root / "cases.xlsx",
                environ={"DEEPSEEK_API_KEY": secret, "FINANCE_WEB_DEMO": "1"},
            )

            serialized = json.dumps(manifest, ensure_ascii=False)
            self.assertNotIn(secret, serialized)
            self.assertTrue(manifest["environment_flags"]["DEEPSEEK_API_KEY_configured"])

    def test_dirty_baseline_code_is_bound_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_root = root / "agent"
            (root / "scripts").mkdir()
            (agent_root / "graph" / "nodes").mkdir(parents=True)
            (root / "scripts" / "benchmark_regression.py").write_text("benchmark", encoding="utf-8")
            (agent_root / "graph" / "nodes" / "web_search.py").write_text("search", encoding="utf-8")

            manifest = build_manifest(
                finance_root=root,
                agent_root=agent_root,
                kb_path=root / "kb.json",
                test_set=root / "cases.xlsx",
                environ={},
            )

            self.assertEqual(manifest["artifacts"]["benchmark_script"]["status"], "present")
            self.assertEqual(manifest["artifacts"]["agent_web_search"]["status"], "present")
            self.assertIn("sha256", manifest["artifacts"]["benchmark_script"])
            self.assertIn("sha256", manifest["artifacts"]["agent_web_search"])


if __name__ == "__main__":
    unittest.main()
