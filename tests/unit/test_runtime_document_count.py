"""The health count includes documents that can only be found by page image."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.services.qa_runtime import QARuntime
from src.pipeline.snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    write_manifest_atomic,
)


class RuntimeDocumentCountTests(unittest.TestCase):
    def test_health_reports_web_search_configured_without_key(self):
        with patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test-secret"}):
            status = QARuntime().status()
        self.assertTrue(status["web_search_configured"])
        self.assertNotIn("tvly-test-secret", str(status))
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            self.assertFalse(QARuntime().status()["web_search_configured"])

    def test_health_reports_stopped_paid_budget(self):
        from src.system.pilot_budget import PilotBudget
        with TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            budget = PilotBudget(path)
            budget.stop("missing-request", "request_uncertain")
            with patch.dict("os.environ", {"FINANCE_LLM_PROVIDER": "ark",
                                        "FINANCE_PILOT_BUDGET_PATH": str(path)}):
                status = QARuntime().status()
            self.assertEqual(status["pilot_budget_stop_reason"], "request_uncertain")

    def test_health_counts_text_and_image_only_documents_in_active_snapshot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "kb_v3_manifest.json"
            manifest = SnapshotManifest.build(
                [
                    VersionArtifact("doc-text", "dv-text", "text.json", 1, "sha-text"),
                    VersionArtifact("doc-image", "dv-image", "image.json", 0, "sha-image"),
                ],
                SnapshotConfig(2, "chunk-v2", "parsers-v2"),
            )
            write_manifest_atomic(manifest, path)
            runtime = QARuntime()
            runtime._manifest_path = str(path)
            runtime._qa = SimpleNamespace(
                knowledge_chunks=[{"document_id": "doc-text"}],
                image_retriever=SimpleNamespace(
                    stats={"indexed_pages": 1},
                    pages=[{"document_id": "doc-image"}],
                ),
            )

            status = runtime.status()

            self.assertEqual(status["documents"], 2)
            self.assertEqual(status["text_documents"], 1)
            self.assertEqual(status["image_only_documents"], 1)


if __name__ == "__main__":
    unittest.main()
