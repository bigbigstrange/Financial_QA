import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.document_visibility import filter_visible_candidates
from src.pipeline.docir import DocBlock, DocIR, DocPage
from src.pipeline.snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    sha256_file,
    write_manifest_atomic,
)
from src.system.index_fingerprint import IndexFingerprint
from src.system.retrieval import KnowledgeStore


@dataclass(frozen=True)
class StaticVisibility:
    versions: frozenset[str]

    def allowed(self, version_id: str) -> bool:
        return version_id in self.versions


def write_table_docir(path: Path, version: str, text: str) -> None:
    doc = DocIR(
        doc_id="doc-table",
        source_path="table.xlsx",
        source_format=".xlsx",
        document_version=version,
        schema_version=2,
    )
    doc.add_page(
        DocPage(
            page_no=None,
            blocks=[
                DocBlock(
                    block_id="blk-table",
                    block_type="table_rows",
                    text=text,
                    page=None,
                    structural_locator="sheet/S1/rows/000001-000001",
                )
            ],
        )
    )
    path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")


def write_test_manifest(path: Path, docir_path: Path, version: str) -> None:
    artifact = VersionArtifact(
        "doc-table",
        version,
        str(docir_path),
        1,
        sha256_file(docir_path),
    )
    manifest = SnapshotManifest.build(
        [artifact],
        SnapshotConfig(2, "chunk-v2", "parsers-v2"),
    )
    write_manifest_atomic(manifest, path)


class SnapshotRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_table_row_text_is_loaded_from_snapshot(self):
        docir_path = self.tmp_path / "table.docir.json"
        write_table_docir(
            docir_path,
            version="dv-table",
            text="[列] 项目 | 金额\n[行 1] 现金 | 10",
        )
        manifest_path = self.tmp_path / "snapshot.json"
        write_test_manifest(manifest_path, docir_path, version="dv-table")

        blocks = KnowledgeStore(manifest_path).load()

        self.assertEqual(blocks[0]["type"], "table_rows")
        self.assertIn("现金", blocks[0]["content"])
        self.assertEqual(blocks[0]["document_version"], "dv-table")

    def test_missing_snapshot_does_not_silently_load_empty(self):
        with self.assertRaises(FileNotFoundError):
            KnowledgeStore(self.tmp_path / "missing.json").load()

    def test_relative_docir_is_relative_to_manifest(self):
        path = self.tmp_path / "table.docir.json"
        write_table_docir(path, "dv-table", "正文")
        manifest = SnapshotManifest.build([VersionArtifact("doc-table", "dv-table", path.name, 1, sha256_file(path))],
            SnapshotConfig(2, "chunk-v2", "parsers-v2"))
        manifest_path = self.tmp_path / "manifest.json"
        write_manifest_atomic(manifest, manifest_path)
        self.assertEqual(KnowledgeStore(manifest_path).load()[0]["content"], "正文")

    def test_deleted_version_is_filtered_even_when_index_contains_it(self):
        candidates = [
            {"document_version": "dv-deleted", "score": 0.9},
            {"document_version": "dv-active", "score": 0.8},
            {"document_version": "dv-active-2", "score": 0.7},
        ]
        visibility = StaticVisibility(frozenset({"dv-active", "dv-active-2"}))

        result = filter_visible_candidates(candidates, top_k=2, visibility=visibility)

        self.assertEqual(
            [item["document_version"] for item in result],
            ["dv-active", "dv-active-2"],
        )

    def test_parser_version_change_invalidates_cache(self):
        base = dict(
            snapshot_id="snap-a",
            embedding_model="BAAI/bge-m3",
            embedding_revision="main",
            embedding_dimension=1024,
            normalization_method="l2",
            distance_metric="ip",
            docir_schema_version=2,
            chunker_version="chunk-v2",
            retrieval_channel="text",
            demo_subset_hash=None,
        )
        one = IndexFingerprint(parser_versions_hash="a", **base)
        two = IndexFingerprint(parser_versions_hash="b", **base)

        self.assertNotEqual(one.cache_key(), two.cache_key())
        self.assertFalse(one.matches(two.to_dict()))

    def test_missing_manifest_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            KnowledgeStore(self.tmp_path / "missing.json").load()

    def test_missing_docir_referenced_by_manifest_fails(self):
        manifest_path = self.tmp_path / "broken.json"
        docir_path = self.tmp_path / "missing.docir.json"
        write_table_docir(docir_path, version="dv-broken", text="正文")
        write_test_manifest(
            manifest_path,
            docir_path,
            version="dv-broken",
        )
        docir_path.unlink()

        with self.assertRaises(FileNotFoundError):
            KnowledgeStore(manifest_path).load()


if __name__ == "__main__":
    unittest.main()
