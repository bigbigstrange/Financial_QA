import tempfile
import unittest
import pickle
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.system.vision import ChineseVisionRetriever
from src.system.retrieval import VisionPageRetriever, _faiss_write_index, _digest_block_ids
from src.system.index_fingerprint import write_index_fingerprint_meta


class ChineseVisionTests(unittest.TestCase):
    def test_inventory_build_preserves_chunkless_page_and_invalidates_image_digest(self):
        import torch
        from PIL import Image
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as cache:
            path = Path(cache)/'page.png'
            Image.new('RGB', (8, 8), 'red').save(path)
            page = dict(page_id='vp_x', snapshot_id='s', document_id='d',
                document_version='v', source_doc='x', page=1, image_ref=str(path),
                image_sha256='first', chunks=[], quality_flags=[])
            r = ChineseVisionRetriever(cache)
            r.processor = lambda **kwargs: {}
            r.model = SimpleNamespace(get_image_features=lambda **kwargs: torch.tensor([[3., 4.]]))
            r.build([], {}, visual_pages=[page])
            np.testing.assert_allclose(r.vectors, [[0.6, 0.8]])
            self.assertEqual(r.pages[0]['page_id'], 'vp_x')
            r.model = SimpleNamespace(get_image_features=lambda **kwargs: torch.tensor([[4., 3.]]))
            r.build([], {}, visual_pages=[{**page, 'image_sha256': 'changed'}])
            np.testing.assert_allclose(r.vectors, [[0.8, 0.6]])
            self.assertEqual(len(list(Path(cache).glob('vision_cn_*.npz'))), 2)

    def test_unlinked_page_is_empty_text_evidence_with_preserved_metadata(self):
        from src.contracts import evidence_from_v2
        with tempfile.TemporaryDirectory() as cache:
            r = ChineseVisionRetriever(cache)
            r.pages = [dict(page_id='vp_test', snapshot_id='snap', document_id='doc',
                document_version='v', source_doc='same.pdf', page=2, image_ref='two.png',
                image_sha256='abc', chunks=[], quality_flags=['no_text_chunks'])]
            hits = r.resolve_hits(np.array([0.9]), top_k=1)
            self.assertEqual(len(hits), 1)
            hit = hits[0]
            self.assertEqual(hit['block_id'], 'vp_test')
            self.assertIsNone(hit['chunk_id'])
            self.assertEqual(hit['block_ids'], [])
            self.assertEqual(hit['content'], '')
            self.assertEqual(hit['vision_page'], 2)
            ev = evidence_from_v2(hit, 'snap')
            self.assertEqual(ev.metadata['evidence_kind'], 'page_image')
            self.assertEqual(ev.metadata['page_id'], 'vp_test')
            self.assertEqual(ev.metadata['image_sha256'], 'abc')
            self.assertEqual(ev.text, '')
            class Hidden:
                def allowed(self, version):
                    return version != 'v'
            self.assertEqual(r.resolve_hits(np.array([0.9]), visibility=Hidden()), [])

    def test_zero_limit_returns_no_linked_results(self):
        with tempfile.TemporaryDirectory() as cache:
            r = ChineseVisionRetriever(cache)
            r.pages = [{'page': 1, 'image_ref': 'one.png', 'chunks': [{'block_id': 'ck'}]}]
            self.assertEqual(r.resolve_hits(np.array([0.9]), top_k=0), [])

    def test_old_name_grouped_cache_is_not_reused_for_version_grouped_pages(self):
        blocks = [{"source_doc": "same.pdf", "document_version": version,
                   "block_id": f"ck_{version}", "page": 1, "content": text}
                  for version, text in [("v_a", "甲"), ("v_b", "乙")]]
        for snapshot in (None, "snap_test"):
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as cache:
                # Only the heavyweight model encoder is replaced; cache IO/build are real.
                retriever = VisionPageRetriever.__new__(VisionPageRetriever)
                retriever.cache_dir = Path(cache)
                retriever.kb_hash = "old_kb"
                retriever.snapshot_id = snapshot
                retriever.backend = "visrag"
                retriever.model_name = "test_model"
                old_prefix = (f"{retriever._path_fingerprint().cache_key()}_image"
                              if snapshot else "old_kb_visrag_image")
                old_index = Path(cache) / f"{old_prefix}.faiss"
                index = faiss.IndexFlatIP(2)
                index.add(np.array([[1., 0.]], dtype=np.float32))
                _faiss_write_index(index, old_index)
                old_index.with_suffix(".pkl").write_bytes(pickle.dumps([
                    {"content": "甲\n乙", "image_ref": "b.png"}]))
                if snapshot:
                    write_index_fingerprint_meta(old_index, retriever._path_fingerprint(dim=2),
                                                 block_ids_sha256=_digest_block_ids(blocks))
                with patch.object(retriever, "_encode_page_items", return_value=np.eye(2, dtype=np.float32)):
                    retriever.build(blocks, {("v_a", 1): "a.png", ("v_b", 1): "b.png"})
                self.assertEqual([(p["image_ref"], p["content"]) for p in retriever.metadata],
                                 [("a.png", "甲"), ("b.png", "乙")])
                self.assertEqual(retriever.index.ntotal, 2)

    def test_legacy_visual_backend_keeps_same_named_versions_separate(self):
        blocks = [{"source_doc": "same.pdf", "document_version": version,
                   "page": 1, "content": text}
                  for version, text in [("v_a", "甲"), ("v_b", "乙")]]
        pages = VisionPageRetriever._build_page_items(blocks, {
            ("v_a", 1): "a.png", ("v_b", 1): "b.png"})
        self.assertEqual([(p["image_ref"], p["content"]) for p in pages],
                         [("a.png", "甲"), ("b.png", "乙")])
        self.assertEqual(len({p["block_id"] for p in pages}), 2)

    def test_same_named_documents_keep_separate_images_and_evidence(self):
        blocks = [
            {"source_doc": "same.pdf", "document_id": "doc_a", "document_version": "v_a",
             "chunk_id": "ck_a", "block_id": "ck_a", "page": 1},
            {"source_doc": "same.pdf", "document_id": "doc_b", "document_version": "v_b",
             "chunk_id": "ck_b", "block_id": "ck_b", "page": 1},
        ]
        pages = ChineseVisionRetriever.page_items(blocks, {
            ("v_a", 1): "a.png", ("v_b", 1): "b.png"})
        self.assertEqual(len(pages), 2)
        with tempfile.TemporaryDirectory() as cache:
            retriever = ChineseVisionRetriever(cache)
            retriever.pages = pages
            hits = retriever.resolve_hits(np.array([0.9, 0.8]))
        self.assertEqual([(h["chunk_id"], h["image_ref"]) for h in hits],
                         [("ck_a", "a.png"), ("ck_b", "b.png")])
        self.assertEqual([h["source_doc"] for h in hits], ["same.pdf", "same.pdf"])

    def test_multi_page_chunk_maps_both_pages_and_keeps_identity(self):
        block = {"chunk_id": "ck_1", "block_id": "ck_1", "source_doc": "A",
                 "page_start": 2, "page_end": 3, "document_version": "v1"}
        pages = ChineseVisionRetriever.page_items([block], {("A", 2): "two.png", ("A", 3): "three.png"})
        self.assertEqual([p["page"] for p in pages], [2, 3])
        with tempfile.TemporaryDirectory() as cache:
            r = ChineseVisionRetriever(cache)
            r.pages = pages
            hits = r.resolve_hits(np.array([0.2, 0.8]))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_id"], "ck_1")
        self.assertEqual(hits[0]["vision_page"], 3)
        self.assertEqual(hits[0]["document_version"], "v1")

    def test_visibility_filters_image_evidence(self):
        class Hidden:
            def allowed(self, version):
                return False
        with tempfile.TemporaryDirectory() as cache:
            r = ChineseVisionRetriever(cache)
            r.pages = [{"page": 1, "image_ref": "one.png", "chunks": [{"block_id": "ck1", "document_version": "deleted"}]}]
            self.assertEqual(r.resolve_hits(np.array([0.9]), visibility=Hidden()), [])

    def test_no_synthetic_page_for_unpaginated_document(self):
        self.assertEqual(ChineseVisionRetriever.page_items(
            [{"source_doc": "text", "page": None}], {}), [])
