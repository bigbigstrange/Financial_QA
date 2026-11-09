import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.system.index_fingerprint import (
    IndexFingerprint,
    index_meta_matches,
    load_index_fingerprint_meta,
    write_index_fingerprint_meta,
)
from src.system.retrieval import _digest_block_ids
from src.system.retrieval import TextRetriever, VisionPageRetriever


class IndexFingerprintIOTests(unittest.TestCase):
    def test_text_fingerprint_binds_actual_snapshot_config_and_sequence_limit(self):
        retriever = object.__new__(TextRetriever)
        retriever.snapshot_id = "snap-v3"
        retriever.snapshot_config = {"docir_schema_version": 3, "chunker_version": "chunker-v3", "parser_versions_hash": "parsers-v3"}
        retriever.model_name = "bge-m3"
        retriever.model = type("Model", (), {"get_sentence_embedding_dimension": lambda _: 1024})()
        retriever.embedding_revision = "fixed-commit:max_seq=512"
        first = retriever._build_fingerprint()
        self.assertEqual(first.docir_schema_version, 3)
        self.assertEqual(first.chunker_version, "chunker-v3")
        self.assertEqual(first.parser_versions_hash, "parsers-v3")
        retriever.embedding_revision = "fixed-commit:max_seq=1024"
        self.assertNotEqual(first.cache_key(), retriever._build_fingerprint().cache_key())

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _fingerprint(self, parser_hash="parsers-v2") -> IndexFingerprint:
        return IndexFingerprint(
            snapshot_id="snap-a",
            embedding_model="BAAI/bge-m3",
            embedding_revision="main",
            embedding_dimension=1024,
            normalization_method="l2",
            distance_metric="ip",
            docir_schema_version=2,
            chunker_version="chunk-v2",
            parser_versions_hash=parser_hash,
            retrieval_channel="text",
        )

    def test_metadata_round_trip_and_block_digest(self):
        index_path = self.tmp_path / "index.faiss"
        fingerprint = self._fingerprint()
        blocks = [
            {"block_id": "blk-2", "content": "乙"},
            {"block_id": "blk-1", "content": "甲"},
        ]
        digest = _digest_block_ids(blocks)
        write_index_fingerprint_meta(
            index_path,
            fingerprint,
            block_ids_sha256=digest,
        )

        meta = load_index_fingerprint_meta(index_path)

        self.assertTrue(
            index_meta_matches(
                fingerprint,
                meta,
                block_ids_sha256=digest,
            )
        )
        self.assertFalse(
            index_meta_matches(
                fingerprint,
                meta,
                block_ids_sha256="tampered",
            )
        )

    def test_parser_or_snapshot_change_rejects_cache(self):
        index_path = self.tmp_path / "index.faiss"
        old = self._fingerprint("parsers-v1")
        write_index_fingerprint_meta(index_path, old)

        meta = load_index_fingerprint_meta(index_path)

        self.assertFalse(
            index_meta_matches(self._fingerprint("parsers-v2"), meta)
        )
        self.assertFalse(
            index_meta_matches(
                IndexFingerprint(
                    snapshot_id="snap-b",
                    embedding_model="BAAI/bge-m3",
                    embedding_revision="main",
                    embedding_dimension=1024,
                    normalization_method="l2",
                    distance_metric="ip",
                    docir_schema_version=2,
                    chunker_version="chunk-v2",
                    parser_versions_hash="parsers-v1",
                    retrieval_channel="text",
                ),
                meta,
            )
        )

    def test_vision_cache_path_uses_fingerprint(self):
        retriever = object.__new__(VisionPageRetriever)
        retriever.snapshot_id = "snap-a"
        retriever.kb_hash = "legacy-hash"
        retriever.backend = "visrag"
        retriever.model_name = "openai/clip-vit-base-patch32"
        retriever.cache_dir = self.tmp_path

        index_path, _ = retriever._cache_paths()

        fingerprint = retriever._path_fingerprint()
        self.assertTrue(index_path.name.startswith(f"{fingerprint.cache_key()}_image"))
        self.assertNotIn("legacy-hash", index_path.name)


if __name__ == "__main__":
    unittest.main()
