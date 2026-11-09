"""KnowledgeStore 对 KB v3 `chunks.jsonl` 的加载契约测试。

阶段 A 花了大力气把行级碎片聚合成 400+ 字的语义单元；如果检索层仍按 block
加载，这些成果就完全用不上。因此这里的测试锁定三件事：

  1. `chunks.jsonl` 优先级最高，且字段被完整保留（章节/页码/overlap/溯源）；
  2. 关键词与 max_blocks 截断按 chunk 计数（不会把 chunk 切一半）；
  3. 旧格式（legacy knowledge_base.json）在补齐后也带上了 section_path 与表格字段。
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.system.retrieval import KnowledgeStore
from src.pipeline.docir import DocIR, DocPage
from src.pipeline.snapshots import SnapshotManifest, SnapshotConfig, VersionArtifact, sha256_file, write_manifest_atomic


def _chunk(chunk_id: str, content: str, **kw) -> dict:
    base = {
        "block_id": chunk_id,
        "chunk_id": chunk_id,
        "block_ids": [f"blk_{chunk_id}"],
        "content": content,
        "source_doc": "某银行2023年报.pdf",
        "document_id": "doc_1",
        "document_version": "dv_1",
        "snapshot_id": "snap_1",
        "page": 3,
        "page_start": 3,
        "page_end": 4,
        "bbox": [10.0, 20.0, 300.0, 400.0],
        "section_path": "经营概况/资产质量",
        "type": "paragraph",
        "char_len": len(content),
        "token_estimate": len(content),
        "overlap_prev_chars": 96,
        "quality_flags": [],
    }
    base.update(kw)
    return base


class ChunkLoadingTests(unittest.TestCase):
    def test_manifest_keeps_page_images_for_same_named_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = []
            for suffix in ("a", "b"):
                image = root / f"{suffix}.png"
                image.touch()
                doc = DocIR(doc_id=f"doc_{suffix}", source_path=f"{suffix}/same.pdf",
                            source_format="pdf", source_name="same.pdf",
                            document_version=f"v_{suffix}",
                            pages=[DocPage(page_no=1, meta={"image_ref": image.name})])
                path = doc.save(root)["json"]
                artifacts.append(VersionArtifact(doc.document_id, doc.document_version,
                                                 path.name, 0, sha256_file(path)))
            manifest = SnapshotManifest.build(artifacts, SnapshotConfig(2, "test", "test"))
            write_manifest_atomic(manifest, root / "kb_v3_manifest.json")
            kb = self._write_kb(root, [_chunk("ck_a", "甲", document_version="v_a"),
                                     _chunk("ck_b", "乙", document_version="v_b")])
            store = KnowledgeStore(str(kb))
            store.load()
            self.assertEqual(len(store.page_images), 2)
            self.assertEqual(store.page_images[("v_a", 1)], str(root / "a.png"))
            self.assertEqual(store.page_images[("v_b", 1)], str(root / "b.png"))

    def _write_kb(self, root: Path, items: list[dict]) -> Path:
        (root / "chunks.jsonl").write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
            encoding="utf-8",
        )
        (root / "kb_v3.json").write_text(
            json.dumps({"schema_version": 3, "documents": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        return root / "kb_v3.json"

    def test_loads_chunks_with_full_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = self._write_kb(root, [_chunk("ck_1", "营业收入1000亿元，同比增长5%。")])
            store = KnowledgeStore(str(kb_path))
            blocks = store.load()
            self.assertEqual(len(blocks), 1)
            item = blocks[0]
            self.assertEqual(item["block_id"], "ck_1")
            self.assertEqual(item["chunk_id"], "ck_1")
            self.assertEqual(item["section_path"], "经营概况/资产质量")
            self.assertEqual(item["page_start"], 3)
            self.assertEqual(item["page_end"], 4)
            self.assertEqual(item["overlap_prev_chars"], 96)
            self.assertEqual(item["document_version"], "dv_1")
            self.assertEqual(item["snapshot_id"], "snap_1")
            self.assertEqual(item["type"], "paragraph")

    def test_kb_path_can_point_at_chunks_file_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_kb(root, [_chunk("ck_1", "内容甲。")])
            store = KnowledgeStore(str(root / "chunks.jsonl"))
            self.assertEqual(len(store.load()), 1)

    def test_empty_content_chunks_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = self._write_kb(
                root,
                [_chunk("ck_1", "有内容。"), _chunk("ck_2", "   "),
                 {"chunk_id": "ck_3", "content": "缺 id？不，有 id。"}],
            )
            store = KnowledgeStore(str(kb_path))
            blocks = store.load()
            self.assertEqual([b["block_id"] for b in blocks], ["ck_1", "ck_3"])

    def test_max_blocks_counts_chunks_not_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = self._write_kb(
                root, [_chunk(f"ck_{i}", f"第{i}段内容。") for i in range(10)]
            )
            store = KnowledgeStore(str(kb_path))
            blocks = store.load(max_blocks=4)
            self.assertEqual(len(blocks), 4)
            # 每个 chunk 都是完整单元，不是被截断的块
            self.assertTrue(all(b["content"].endswith("。") for b in blocks))

    def test_keyword_filter_matches_section_and_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = self._write_kb(
                root,
                [
                    _chunk("ck_1", "内容甲。", section_path="流动性风险"),
                    _chunk("ck_2", "内容乙。", source_doc="另一份.pdf"),
                ],
            )
            store = KnowledgeStore(str(kb_path))
            blocks = store.load(doc_keywords=["流动性风险"])
            self.assertEqual([b["block_id"] for b in blocks], ["ck_1"])

    def test_malformed_line_is_skipped_without_aborting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "chunks.jsonl").write_text(
                json.dumps(_chunk("ck_1", "内容甲。"), ensure_ascii=False)
                + "\n{ not json\n"
                + json.dumps(_chunk("ck_2", "内容乙。"), ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            store = KnowledgeStore(str(root / "chunks.jsonl"))
            blocks = store.load()
            self.assertEqual([b["block_id"] for b in blocks], ["ck_1", "ck_2"])

    def test_falls_back_when_chunks_missing(self):
        """没有 chunks.jsonl 时不能报错，必须退回原有加载路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = {
                "documents": [
                    {
                        "doc_id": "d1",
                        "pages": [
                            {
                                "page_no": 1,
                                "blocks": [
                                    {"block_id": "b1", "block_type": "paragraph",
                                     "text": "旧格式内容。",
                                     "section_path": "章节A"}
                                ],
                            }
                        ],
                    }
                ]
            }
            kb_path = root / "knowledge_base.json"
            kb_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
            store = KnowledgeStore(str(kb_path))
            blocks = store.load()
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["content"], "旧格式内容。")
            # 旧格式补齐后 section_path 必须保留（结构化通道依赖它）
            self.assertEqual(blocks[0]["section_path"], "章节A")


if __name__ == "__main__":
    unittest.main()
