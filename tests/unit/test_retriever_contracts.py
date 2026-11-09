"""检索器接口契约测试。

## 为什么需要这组测试

`FusionRetriever.search()` 会**无条件**把 `visibility=` 传给图像检索器：

    image_results = self.image_retriever.search(query, top_k=..., visibility=visibility)

而 `FINANCE_SKIP_IMAGE=1` / 演示模式下用的是 `_EmptyImageRetriever`，它的 `search`
当时没有 `visibility` 参数 → 抛 `TypeError` → **整条融合检索返回 0 条证据**。

这是个**静默失败**：不报错到用户、不写日志、只是永远答「知识库中无依据」。
低内存演示模式因此一直是坏的，却没人发现。

本测试把「所有会被传入 visibility 的检索器都必须接受它」写成契约，
今后再加参数时会立刻失败，而不是等到线上查不出问题。
"""
import inspect
import unittest

from src.system.retrieval import (
    FusionRetriever,
    HybridRetriever,
    SparseRetriever,
    StructuredRetriever,
    TextRetriever,
    VisionPageRetriever,
    _EmptyImageRetriever,
)
from src.system.sparse import SparseRetriever as SparseRetrieverDirect
from src.system.structured import StructuredRetriever as StructuredRetrieverDirect


def _accepts_visibility(cls: type) -> bool:
    """`search` 是否接受 visibility（显式声明或用 **kwargs 兜底）。"""
    signature = inspect.signature(cls.search)
    if "visibility" in signature.parameters:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


class RetrieverSignatureContractTests(unittest.TestCase):
    """`HybridRetriever` / `FusionRetriever` 会传 visibility 的检索器全集。"""

    def test_all_visibility_consumers_accept_the_kwarg(self):
        consumers = [
            ("TextRetriever", TextRetriever),
            ("VisionPageRetriever", VisionPageRetriever),
            ("_EmptyImageRetriever", _EmptyImageRetriever),
            ("SparseRetriever(retrieval)", SparseRetriever),
            ("StructuredRetriever(retrieval)", StructuredRetriever),
            ("SparseRetriever(sparse)", SparseRetrieverDirect),
            ("StructuredRetriever(structured)", StructuredRetrieverDirect),
            ("FusionRetriever", FusionRetriever),
            ("HybridRetriever", HybridRetriever),
        ]
        failures = [name for name, cls in consumers if not _accepts_visibility(cls)]
        self.assertEqual(
            failures, [],
            f"这些检索器不接受 visibility，会让融合检索静默返回 0 条: {failures}",
        )

    def test_empty_image_retriever_returns_empty_list(self):
        retriever = _EmptyImageRetriever()
        self.assertEqual(retriever.search("任意问题"), [])
        self.assertEqual(retriever.search("任意问题", top_k=8, visibility=None), [])
        # 即使多传了未知参数也不应抛错（图像塔后端差异会带来额外参数）
        self.assertEqual(
            retriever.search("任意问题", top_k=8, visibility=None, extra=1), []
        )


class _FakeTextRetriever:
    """模拟 TextRetriever：必须设置 retrieval_channel，否则融合层拿不到通道信息。"""

    def __init__(self, items):
        self.items = items

    def search(self, query, top_k=10, visibility=None):
        out = []
        for it in self.items[:top_k]:
            item = dict(it)
            item.setdefault("retrieval_channel", "text")
            out.append(item)
        return out


class FusionWithEmptyImageTests(unittest.TestCase):
    """核心回归：skip_image 时融合检索必须仍能返回文本通道结果。"""

    def _docs(self):
        return [
            {"block_id": "ck_1", "content": "2023年营业收入1000亿元。",
             "source_doc": "某银行年报.pdf", "page": 3},
            {"block_id": "ck_2", "content": "同比增长5%。",
             "source_doc": "某银行年报.pdf", "page": 3},
        ]

    def test_fusion_returns_text_results_when_image_tower_is_skipped(self):
        fusion = FusionRetriever(
            text_retriever=_FakeTextRetriever(self._docs()),
            image_retriever=_EmptyImageRetriever(),
        )
        results = fusion.search("营业收入是多少？", top_k=5, visibility=None)
        self.assertEqual([r["block_id"] for r in results], ["ck_1", "ck_2"])
        self.assertGreater(results[0].get("fusion_score", 0), 0)
        self.assertIn("text", results[0].get("retrieval_channels", []))

    def test_fusion_works_without_visibility_argument(self):
        fusion = FusionRetriever(
            text_retriever=_FakeTextRetriever(self._docs()),
            image_retriever=_EmptyImageRetriever(),
        )
        self.assertEqual(len(fusion.search("营业收入", top_k=5)), 2)

    def test_demo_mode_hybrid_stack_returns_evidence(self):
        """端到端：HybridRetriever 在 skip_image 配置下必须能返回证据。

        这条覆盖的正是历史上"演示模式永远答无依据"的故障场景。
        """
        docs = self._docs()
        hybrid = HybridRetriever(
            fusion_retriever=FusionRetriever(
                text_retriever=_FakeTextRetriever(docs),
                image_retriever=_EmptyImageRetriever(),
            ),
            sparse_retriever=SparseRetrieverDirect(docs),
            structured_retriever=StructuredRetrieverDirect(docs),
        )
        results = hybrid.search("某银行营业收入", top_k=5)
        self.assertTrue(results, "skip_image 模式下混合检索返回了空结果")
        self.assertTrue(all(r.get("block_id") for r in results))


if __name__ == "__main__":
    unittest.main()
