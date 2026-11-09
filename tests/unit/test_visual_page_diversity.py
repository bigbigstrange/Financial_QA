"""A visual page should not occupy the whole evidence window."""

import numpy as np

from src.system.vision import ChineseVisionRetriever


def _page(page_id, version, number, score_chunks):
    return {
        "page_id": page_id,
        "snapshot_id": "snap",
        "document_id": version,
        "document_version": version,
        "source_doc": f"{version}.pdf",
        "page": number,
        "image_ref": f"{page_id}.png",
        "image_sha256": f"sha-{page_id}",
        "quality_flags": [],
        "chunks": [
            {
                "block_id": chunk_id,
                "chunk_id": chunk_id,
                "content": content,
                "document_version": version,
                "source_doc": f"{version}.pdf",
                "page": number,
            }
            for chunk_id, content in score_chunks
        ],
    }


def test_one_page_cannot_fill_two_visual_slots(tmp_path):
    retriever = ChineseVisionRetriever(tmp_path)
    retriever.pages = [
        _page("page_a", "bank_a", 1, [("a_generic", "网点介绍"), ("a_fund", "基金认购操作")]),
        _page("page_b", "bank_b", 1, [("b_fund", "基金认购申请")]),
    ]

    hits = retriever.resolve_hits(np.array([0.9, 0.8]), top_k=2, query="基金认购")

    assert [hit["page_id"] for hit in hits] == ["page_a", "page_b"]
    assert [hit["chunk_id"] for hit in hits] == ["a_fund", "b_fund"]


def test_cross_page_chunk_only_uses_highest_scoring_page(tmp_path):
    retriever = ChineseVisionRetriever(tmp_path)
    retriever.pages = [
        _page("page_3", "bank_a", 3, [("shared", "基金认购")]),
        _page("page_2", "bank_a", 2, [("shared", "基金认购")]),
        _page("page_4", "bank_a", 4, [("unique", "基金认购")]),
    ]

    hits = retriever.resolve_hits(np.array([0.95, 0.9, 0.8]), top_k=2)

    assert [hit["page_id"] for hit in hits] == ["page_3", "page_4"]
    assert [hit["chunk_id"] for hit in hits] == ["shared", "unique"]


def test_image_only_page_keeps_provenance_and_visibility(tmp_path):
    retriever = ChineseVisionRetriever(tmp_path)
    retriever.pages = [
        _page("hidden", "private", 1, [("private_chunk", "基金")]),
        _page("image_only", "public", 2, []),
    ]

    class PublicOnly:
        def allowed(self, version):
            return version == "public"

    hits = retriever.resolve_hits(np.array([0.95, 0.9]), top_k=2, visibility=PublicOnly())

    assert len(hits) == 1
    assert hits[0]["page_id"] == "image_only"
    assert hits[0]["document_version"] == "public"
    assert hits[0]["evidence_kind"] == "page_image"
    assert hits[0]["chunk_id"] is None


def test_legacy_sentinel_does_not_merge_different_documents(tmp_path):
    retriever = ChineseVisionRetriever(tmp_path)
    first = _page("first", "legacy-v1", 1, [("first_chunk", "甲银行基金")])
    second = _page("second", "legacy-v1", 1, [("second_chunk", "乙银行基金")])
    first.update(document_id="doc_a", source_doc="a.pdf")
    second.update(document_id="doc_b", source_doc="b.pdf")
    retriever.pages = [first, second]

    hits = retriever.resolve_hits(np.array([0.9, 0.8]), top_k=2)

    assert [hit["block_id"] for hit in hits] == ["first_chunk", "second_chunk"]


def test_name_keyed_legacy_images_keep_versions_separate():
    blocks = [
        {"source_doc": "same.pdf", "document_id": "doc_a", "document_version": "v_a",
         "block_id": "a", "chunk_id": "a", "page": 1},
        {"source_doc": "same.pdf", "document_id": "doc_b", "document_version": "v_b",
         "block_id": "b", "chunk_id": "b", "page": 1},
    ]

    pages = ChineseVisionRetriever.page_items(blocks, {("same.pdf", 1): "shared.png"})

    assert len(pages) == 2
    assert [page["document_version"] for page in pages] == ["v_a", "v_b"]
    assert [[chunk["block_id"] for chunk in page["chunks"]] for page in pages] == [["a"], ["b"]]
