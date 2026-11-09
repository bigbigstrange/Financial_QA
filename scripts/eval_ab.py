"""阶段 A 验收用 A/B 评测：旧链路（行级 block） vs 新链路（chunker-v3 chunk）。

与旧脚本（`eval_retrieval_phase3.py`）的三个关键差异，正是阶段 3 门禁失败的三个原因：

  1. **标准答案是人工审核过的**（gold_set.json），不再从冒烟测试自动抓；
  2. **两侧语料限定在同一批文档上**，比较的是「表示与切块」而不是「谁覆盖的文档多」；
  3. **门禁要求新链路显著优于旧链路**，而不只是过绝对阈值 —— 否则"修完还是原样"也能通过。

用法：
    # 同一批问题，两条链路对照
    python scripts/eval_ab.py --gold output/eval/gold_set.json

    # 只看新链路（不跑旧链路）
    python scripts/eval_ab.py --gold output/eval/gold_set.json --skip-legacy

    # 门禁（失败 exit 2，可用于 CI）
    python scripts/eval_ab.py --gold output/eval/gold_set.json --gate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "output" / "eval"
LEGACY_KB = ROOT / "output" / "knowledge_base.json"
V3_DIR = ROOT / "output" / "kb_v3"

# 门禁阈值：绝对下限 + 相对旧链路的提升倍数
GATE = {
    "v3_hybrid_rerank.recall_at_10": 0.50,
    "v3_hybrid_rerank.mrr": 0.40,
    "v3_hybrid_rerank.ndcg_at_10": 0.40,
    "recall_at_10_improvement_vs_legacy": 1.30,   # v3 ≥ legacy × 1.30
}


# ---------------------------------------------------------------------------
# 指标（与 eval_retrieval_phase3.py 保持一致，避免口径漂移）
# ---------------------------------------------------------------------------

def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for bid in ranked[:k] if bid in gold)
    return hits / min(len(gold), k)


def mean_reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for rank, bid in enumerate(ranked, start=1):
        if bid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    ranked = ranked[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, bid in enumerate(ranked) if bid in gold)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(ranked_cases: list[dict]) -> dict[str, Any]:
    recalls = {k: [] for k in (5, 10, 20)}
    mrrs: list[float] = []
    ndcgs: list[float] = []
    for case in ranked_cases:
        ranked = case["ranked_ids"]
        gold = set(case["gold_ids"])
        for k in (5, 10, 20):
            recalls[k].append(recall_at_k(ranked, gold, k))
        mrrs.append(mean_reciprocal_rank(ranked, gold))
        ndcgs.append(ndcg_at_k(ranked, gold, 10))
    n = max(1, len(mrrs))
    return {
        "questions": n,
        "recall_at_5": round(sum(recalls[5]) / n, 4),
        "recall_at_10": round(sum(recalls[10]) / n, 4),
        "recall_at_20": round(sum(recalls[20]) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "ndcg_at_10": round(sum(ndcgs) / n, 4),
    }


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------

def load_v3_corpus() -> list[dict]:
    path = V3_DIR / "chunks.jsonl"
    if not path.is_file():
        raise SystemExit(f"缺少 KB v3 语料: {path}（先跑 scripts/build_kb_v3.py）")
    items: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append({
                "chunk_id": raw.get("chunk_id") or raw.get("block_id"),
                "block_id": raw.get("block_id") or raw.get("chunk_id"),
                "content": raw.get("content") or "",
                "source_doc": raw.get("source_doc"),
                "page": raw.get("page_start", raw.get("page")),
                "section_path": raw.get("section_path"),
                "type": raw.get("type"),
            })
    return [x for x in items if x["content"].strip()]


def _stem(name: str | None) -> str:
    return Path(str(name or "")).stem


def load_legacy_corpus(allowed_stems: set[str]) -> list[dict]:
    """旧链路的行级 block，**限定在与 KB v3 相同的文档集合上**。

    不限定文档集合的话，比较就变成"谁覆盖的文档多"（旧库缺 38% 文档），
    而不是阶段 A 真正改的"表示与切块"。这里对齐文档，让差异只来自切块。
    """
    import ijson

    if not LEGACY_KB.is_file():
        raise SystemExit(f"缺少旧知识库: {LEGACY_KB}")
    items: list[dict] = []
    with LEGACY_KB.open("rb") as fh:
        for doc in ijson.items(fh, "documents.item", use_float=True):
            doc_id = str(doc.get("doc_id") or "")
            if _stem(doc_id) not in allowed_stems:
                continue
            for page in doc.get("pages") or []:
                pno = page.get("page_no")
                for block in page.get("blocks") or []:
                    text = (block.get("text") or "").strip()
                    if not text:
                        continue
                    bid = block.get("block_id") or f"{doc_id}_p{pno}"
                    items.append({
                        "chunk_id": bid,          # 旧链路的检索单元是 block
                        "block_id": bid,
                        "content": text,
                        "source_doc": doc_id,
                        "page": pno,
                        "section_path": block.get("section_path"),
                        "type": block.get("block_type") or "paragraph",
                    })
    return items


# ---------------------------------------------------------------------------
# 检索策略
# ---------------------------------------------------------------------------

class DenseIndex:
    def __init__(self, items: list[dict], model_name: str, batch_size: int = 32):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.items = items
        self.np = np
        device = (os.environ.get("FINANCE_EMBED_DEVICE") or "").strip() or None
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        # 与 src/system/retrieval.py 保持一致：限制序列长度以控制显存
        max_seq = (os.environ.get("FINANCE_EMBED_MAX_SEQ_LEN") or "").strip()
        if max_seq.isdigit() and int(max_seq) > 0:
            self.model.max_seq_length = int(max_seq)
        else:
            self.model.max_seq_length = 512
        vectors = []
        texts = [it["content"] for it in items]
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            emb = self.model.encode(batch, batch_size=len(batch),
                                    show_progress_bar=False, convert_to_numpy=True)
            emb = np.asarray(emb, dtype=np.float32)
            faiss.normalize_L2(emb)
            vectors.append(emb)
        self.index = faiss.IndexFlatIP(int(self.model.get_sentence_embedding_dimension()))
        if vectors:
            self.index.add(np.vstack(vectors))

    def search(self, query: str, top_k: int) -> list[str]:
        import faiss

        if not self.items:
            return []
        q = self.model.encode([query], convert_to_numpy=True).astype(self.np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return [self.items[i]["chunk_id"] for i in idxs[0] if 0 <= i < len(self.items)]


def _rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def run_strategies(
    items: list[dict],
    cases: list[dict],
    *,
    embed_model: str,
    top_k: int,
    skip_dense: bool,
) -> dict[str, Any]:
    from src.system.rerank import EvidenceReranker
    from src.system.sparse import SparseRetriever
    from src.system.structured import StructuredRetriever

    by_id = {it["chunk_id"]: it for it in items}
    sparse = SparseRetriever(items)
    structured = StructuredRetriever(items)
    dense = None if skip_dense else DenseIndex(items, embed_model)
    reranker = EvidenceReranker()

    buckets: dict[str, list[dict]] = {
        name: [] for name in ("dense", "sparse", "structured", "hybrid", "hybrid+rerank")
    }
    for case in cases:
        question = case["question"]
        gold = [cid for cid in case["gold_chunk_ids"] if cid in by_id]
        dense_ids = dense.search(question, top_k) if dense else []
        sparse_hits = sparse.search(question, top_k=top_k)
        sparse_ids = [h["chunk_id"] for h in sparse_hits]
        struct_hits = structured.search(question, top_k=top_k)
        struct_ids = [h["chunk_id"] for h in struct_hits]

        fused = _rrf([lst for lst in (dense_ids, sparse_ids, struct_ids) if lst])
        hybrid_ids = sorted(fused, key=lambda x: -fused[x])[:top_k]

        pool = []
        for cid in hybrid_ids:
            it = by_id.get(cid)
            if not it:
                continue
            pool.append({
                "chunk_id": cid,
                "block_id": cid,
                "content": it["content"],
                "source_doc": it["source_doc"],
                "page": it["page"],
                "type": it.get("type"),
                "fusion_score": fused.get(cid, 0.0),
            })
        reranked = reranker.rerank(question, pool, top_k=top_k) if pool else []
        rerank_ids = [x["chunk_id"] for x in reranked]

        for name, ranked in (
            ("dense", dense_ids),
            ("sparse", sparse_ids),
            ("structured", struct_ids),
            ("hybrid", hybrid_ids),
            ("hybrid+rerank", rerank_ids),
        ):
            buckets[name].append({
                "question_id": case.get("question_id"),
                "gold_ids": gold,
                "ranked_ids": ranked,
            })
    return {name: evaluate(rows) for name, rows in buckets.items()}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 A 验收 A/B 评测")
    ap.add_argument("--gold", default=str(EVAL_DIR / "gold_set.json"))
    ap.add_argument("--output", default=str(EVAL_DIR / "ab_report.json"))
    ap.add_argument("--embed-model", default=os.environ.get("FINANCE_EMBED_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--skip-dense", action="store_true", help="只跑 BM25/结构化，不加载嵌入模型")
    ap.add_argument("--gate", action="store_true", help="跑门禁，失败 exit 2")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.is_file():
        raise SystemExit(
            f"缺少金标准: {gold_path}\n"
            "请先跑 scripts/build_gold_set.py 的 select/retrieve/draft，"
            "人工审核 gold_review.csv 后再 finalize。"
        )
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = [c for c in gold.get("cases") or [] if c.get("gold_chunk_ids")]
    if not cases:
        raise SystemExit("金标准里没有带 gold_chunk_ids 的题目（可能还没人工审核通过）")
    print(f"金标准题目: {len(cases)}")

    v3_items = load_v3_corpus()
    v3_ids = {it["chunk_id"] for it in v3_items}
    covered = sum(1 for c in cases if any(g in v3_ids for g in c["gold_chunk_ids"]))
    print(f"KB v3 语料: {len(v3_items)} 个检索单元；gold 可命中题目 {covered}/{len(cases)}")
    if covered < len(cases):
        print(f"  ⚠️ {len(cases) - covered} 题的 gold 不在 KB v3 中 —— "
              "这些题在当前知识库里无解，指标会被拉低（需补全文档或修正 gold）")

    report: dict[str, Any] = {
        "gold_path": str(gold_path),
        "question_count": len(cases),
        "gold_covered": covered,
        "v3_corpus_units": len(v3_items),
        "v3": run_strategies(v3_items, cases, embed_model=args.embed_model,
                             top_k=args.top_k, skip_dense=args.skip_dense),
    }

    if not args.skip_legacy:
        allowed = {_stem(it.get("source_doc")) for it in v3_items}
        legacy_items = load_legacy_corpus(allowed)
        report["legacy_corpus_units"] = len(legacy_items)
        report["legacy_docs"] = len(allowed)
        print(f"旧链路语料（限定同批文档）: {len(legacy_items)} 个行级 block")
        report["legacy"] = run_strategies(
            legacy_items, cases, embed_model=args.embed_model,
            top_k=args.top_k, skip_dense=args.skip_dense,
        )

    # 对照
    if "legacy" in report:
        v3 = report["v3"]["hybrid+rerank"]
        lg = report["legacy"]["hybrid+rerank"]
        report["comparison"] = {
            "recall_at_10": {"legacy": lg["recall_at_10"], "v3": v3["recall_at_10"],
                             "delta": round(v3["recall_at_10"] - lg["recall_at_10"], 4)},
            "ndcg_at_10": {"legacy": lg["ndcg_at_10"], "v3": v3["ndcg_at_10"],
                           "delta": round(v3["ndcg_at_10"] - lg["ndcg_at_10"], 4)},
            "mrr": {"legacy": lg["mrr"], "v3": v3["mrr"],
                    "delta": round(v3["mrr"] - lg["mrr"], 4)},
        }

    # 门禁
    checks: list[dict[str, Any]] = []
    v3_best = report["v3"].get("hybrid+rerank") or {}
    for key, threshold in GATE.items():
        if key == "recall_at_10_improvement_vs_legacy":
            if "legacy" not in report:
                continue
            base = report["legacy"]["hybrid+rerank"]["recall_at_10"]
            value = v3_best["recall_at_10"] / base if base > 0 else 0.0
            checks.append({"metric": key, "threshold": threshold,
                           "value": round(value, 4), "passed": value >= threshold})
            continue
        _, metric = key.split(".", 1)
        value = float(v3_best.get(metric) or 0.0)
        checks.append({"metric": key, "threshold": threshold,
                       "value": value, "passed": value >= threshold})
    report["gate"] = {"passed": all(c["passed"] for c in checks), "checks": checks}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("策略         " + "  ".join(f"{k:>10}" for k in
                                 ("R@5", "R@10", "R@20", "MRR", "nDCG@10")))
    print("=" * 72)
    for label in ("legacy", "v3"):
        if label not in report:
            continue
        for name, metric in report[label].items():
            print(f"{label:6} {name:12} " + "  ".join(
                f"{metric[k]:>10.4f}" for k in
                ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10")))
    print()
    if "comparison" in report:
        print("新旧对照（hybrid+rerank）：")
        for key, item in report["comparison"].items():
            print(f"  {key:14} 旧 {item['legacy']:.4f} → 新 {item['v3']:.4f} "
                  f"（{item['delta']:+.4f}）")
    print()
    print("门禁:")
    for check in checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['metric']:44} "
              f"值={check['value']} 阈值={check['threshold']}")
    print(f"\n报告已写入: {out}")

    if args.gate and not report["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
