import tempfile
import unittest
from pathlib import Path

from PIL import Image
from src.pipeline.docir import DocIR, DocPage
from src.pipeline.snapshots import (SnapshotConfig, SnapshotManifest,
    VersionArtifact, sha256_file, write_manifest_atomic)


class VisualPagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docs = [DocIR(doc_id='d'+n, document_version='v'+n,
            source_path=n+'/same.pdf', source_name='same.pdf', source_format='pdf',
            pages=[DocPage(page_no=1, meta={'image_ref': n+'.png'})]) for n in ('a', 'b')]
        self.docs.append(DocIR(doc_id='txt', document_version='vt', source_path='x.txt',
            source_format='txt', pages=[DocPage()]))
        for n in ('a', 'b'):
            Image.new('RGB', (8, 8), 'red').save(self.root / (n+'.png'))
        self.write_manifest()
        self.blocks = [{'block_id': 'chunk_a', 'document_id': 'da',
            'document_version': 'va', 'snapshot_id': self.manifest.snapshot_id,
            'page_start': 1, 'page_end': 1, 'content': 'real text'}]

    def write_manifest(self):
        artifacts = []
        for doc in self.docs:
            path = doc.save(self.root)['json']
            artifacts.append(VersionArtifact(doc.document_id, doc.document_version,
                path.name, len(doc.all_blocks()), sha256_file(path)))
        self.manifest = SnapshotManifest.build(artifacts, SnapshotConfig(2, 'test', 'test'))
        self.path = write_manifest_atomic(self.manifest, self.root/'kb_v3_manifest.json')

    def pages(self, **kwargs):
        from src.system.visual_pages import build_visual_pages
        return build_visual_pages(self.path, self.blocks, allowed_roots=(self.root,), **kwargs)

    def test_identity_separates_snapshot_version_and_page(self):
        from src.system.visual_pages import make_visual_page_id as ident
        value = ident('s', 'd', 'v', 1)
        self.assertTrue(value.startswith('vp_'))
        self.assertEqual(value, ident('s', 'd', 'v', 1))
        for args in [('s2','d','v',1), ('s','d2','v',1), ('s','d','v2',1), ('s','d','v',2)]:
            self.assertNotEqual(value, ident(*args))
        for page in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ident('s','d','v',page)

    def test_enumerates_unlinked_pages_without_fabricating_txt_page(self):
        pages = self.pages()
        self.assertEqual([p['document_version'] for p in pages], ['va', 'vb'])
        self.assertEqual(pages[0]['chunks'], self.blocks)
        self.assertEqual(pages[1]['chunks'], [])
        self.assertEqual(pages[1]['image_sha256'], sha256_file(self.root/'b.png'))
        self.assertIn('no_parsed_blocks', pages[1]['quality_flags'])

    def test_subset_does_not_expand(self):
        self.assertEqual([p['document_version'] for p in self.pages(selected_versions={'vb'})], ['vb'])
        self.assertEqual(self.pages(selected_versions=set()), [])

    def test_windows_manifest_separators_resolve_without_changing_identity(self):
        from src.system.visual_pages import resolve_artifact_path
        path = resolve_artifact_path('nested\\page.png', self.root, (self.root,))
        self.assertEqual(path, (self.root / 'nested' / 'page.png').resolve())
        with self.assertRaises(ValueError):
            resolve_artifact_path('..\\outside.png', self.root, (self.root,))

    def test_missing_image_is_reported_not_dropped(self):
        (self.root/'b.png').unlink()
        page = self.pages()[1]
        self.assertIn('missing_page_image', page['quality_flags'])
        self.assertIsNone(page['image_sha256'])

    def test_missing_reference_is_reported(self):
        self.docs[1].pages[0].meta = {}
        self.write_manifest()
        self.blocks = []
        self.assertIn('missing_page_image', self.pages()[1]['quality_flags'])

    def test_tampered_docir_rejected(self):
        (self.root/'da.docir.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'docir'):
            self.pages()

    def test_duplicate_pages_rejected(self):
        self.docs[1].pages *= 2
        self.write_manifest()
        self.blocks = []
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.pages()

    def test_mismatched_chunk_identity_rejected(self):
        self.blocks[0]['document_id'] = 'db'
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.pages()

    def test_outside_path_rejected_even_when_missing(self):
        from src.system.visual_pages import resolve_artifact_path
        with self.assertRaisesRegex(ValueError, 'outside'):
            resolve_artifact_path('../secret.png', self.root, (self.root,))

    def test_image_only_stack_works_without_text_model(self):
        from unittest.mock import patch
        import numpy as np
        from src.system.retrieval import build_retrieval_stack
        from src.system.vision import ChineseVisionRetriever
        (self.root/'chunks.jsonl').write_text('', encoding='utf-8')
        def encode(retriever, blocks, page_images, force_rebuild=False, *, visual_pages=None):
            retriever.pages = visual_pages
            retriever.vectors = np.eye(len(visual_pages), dtype=np.float32)
        with patch.dict('os.environ', {'FINANCE_VISION_BACKEND': 'chinese_clip',
                'FINANCE_MAX_BLOCKS': '', 'FINANCE_DEMO_KEYWORDS': '', 'FINANCE_DISABLE_PAID_API': '1'}), \
                patch.object(ChineseVisionRetriever, 'build', encode), \
                patch('src.system.retrieval.TextRetriever', side_effect=AssertionError('no text model needed')):
            store, text, image, hybrid = build_retrieval_stack(str(self.path), str(self.root/'cache'),
                'unused', 'unused', skip_image=False)
        self.assertEqual(text.search('问题'), [])
        self.assertEqual(store.blocks, [])
        hits = image.resolve_hits(np.array([0.9, 0.8]))
        self.assertEqual([h['document_version'] for h in hits], ['va', 'vb'])

    def test_unmatched_visual_filter_never_loads_unselected_documents(self):
        import json
        from unittest.mock import patch
        from src.system.retrieval import build_retrieval_stack
        (self.root/'chunks.jsonl').write_text(json.dumps({**self.blocks[0], 'source_doc': 'unselected.pdf'}), encoding='utf-8')
        with patch.dict('os.environ', {'FINANCE_VISION_BACKEND': 'chinese_clip', 'FINANCE_MAX_BLOCKS': ''}), \
                patch('src.system.retrieval.TextRetriever', side_effect=AssertionError('unselected encoder invoked')):
            with self.assertRaisesRegex(ValueError, '0'):
                build_retrieval_stack(str(self.path), str(self.root/'cache'), 'unused', 'unused',
                                     skip_image=False, doc_keywords=['selected-but-missing.pdf'])

    def test_explicit_filter_does_not_fall_back_to_unrelated_text(self):
        import json
        from unittest.mock import patch
        from src.system.retrieval import build_retrieval_stack
        (self.root/'chunks.jsonl').write_text(json.dumps({**self.blocks[0], 'source_doc': 'unrelated.pdf'}), encoding='utf-8')
        with patch.dict('os.environ', {'FINANCE_VISION_BACKEND': 'visrag', 'FINANCE_MAX_BLOCKS': ''}), \
                patch('src.system.retrieval.TextRetriever', side_effect=AssertionError('unrelated text encoded')):
            with self.assertRaises(ValueError):
                build_retrieval_stack(str(self.path), str(self.root/'cache'), 'unused', 'unused',
                                      skip_image=False, doc_keywords=['selected-image.pdf'])

    def test_image_only_selected_document_survives_selected_text_subset(self):
        import json
        from unittest.mock import patch
        from src.system.retrieval import build_retrieval_stack
        from src.system.vision import ChineseVisionRetriever
        self.docs[0].source_name = 'selected-image.pdf'
        self.docs[1].source_name = 'selected-text.pdf'
        self.docs.append(DocIR(doc_id='dc', document_version='vc',
            source_path='c/unrelated.pdf', source_name='unrelated.pdf', source_format='pdf',
            pages=[DocPage(page_no=1, meta={'image_ref': 'c.png'})]))
        Image.new('RGB', (8, 8), 'blue').save(self.root/'c.png')
        self.write_manifest()
        selected_text = {**self.blocks[0], 'block_id': 'chunk_b', 'document_id': 'db',
            'document_version': 'vb', 'snapshot_id': self.manifest.snapshot_id,
            'source_doc': 'selected-text.pdf'}
        (self.root/'chunks.jsonl').write_text(json.dumps(selected_text), encoding='utf-8')

        def capture(retriever, blocks, page_images, force_rebuild=False, *, visual_pages=None):
            retriever.pages = visual_pages

        with patch.dict('os.environ', {'FINANCE_VISION_BACKEND': 'chinese_clip',
                'FINANCE_MAX_BLOCKS': '', 'FINANCE_DEMO_KEYWORDS': ''}), \
                patch.object(ChineseVisionRetriever, 'build', capture), \
                patch('src.system.retrieval.TextRetriever'):
            _, _, image, _ = build_retrieval_stack(str(self.path), str(self.root/'cache'),
                'unused', 'unused', skip_image=False, doc_keywords=['selected'])
        self.assertEqual([page['document_version'] for page in image.pages], ['va', 'vb'])


if __name__ == '__main__':
    unittest.main()
