#!/usr/bin/env python3
"""Phase 3 retrieval A/B: dense vs sparse vs structured vs hybrid vs rerank."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faiss
import numpy as np

from src.system.rerank import EvidenceReranker
from src.system.retrieval import FusionRetriever, HybridRetriever
from src.system.sparse import SparseRetriever
from src.system.structured import StructuredRetriever


class DenseIndex:
    def __init__(self, blocks: List[dict], model_name: str, batch_size: int = 8):
        from sentence_transformers import SentenceTransformer

        self.blocks = blocks
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            trust_remote_code=True,
        )
        self.batch_size = max(1, int(batch_size))
        embeddings = []
        texts = [block["content"] for block in blocks]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            emb = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            emb = np.asarray(emb, dtype=np.float32)
            faiss.normalize_L2(emb)
            embeddings.append(emb)
        self.index = faiss.IndexFlatIP(
            int(self.model.get_sentence_embedding_dimension())
        )
        self.index.add(np.vstack(embeddings))

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        q = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.blocks):
                item = dict(self.blocks[idx])
                item["score"] = float(score)
                item["retrieval_channel"] = "text"
                out.append(item)
        return out


class DenseAdapter:
    def __init__(self, dense: DenseIndex):
        self.dense = dense

    def search(self, query, top_k=10, visibility=None):
        results = self.dense.search(query, top_k=top_k)
        if visibility is not None:
            results = [
                item
                for item in results
                if visibility.allowed(item.get("document_version") or "")
            ]
        return results


@dataclass
class MetricResult:
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float


def recall_at_k(ranked: List[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for bid in ranked[:k] if bid in gold)
    return hits / min(len(gold), k)


def mean_reciprocal_rank(ranked: List[str], gold: set[str]) -> float:
    for rank, bid in enumerate(ranked, start=1):
        if bid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: List[str], gold: set[str], k: int = 10) -> float:
    ranked = ranked[:k]
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, bid in enumerate(ranked)
        if bid in gold
    )
    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    strategy: str,
    ranked_cases: List[Dict[str, Any]],
) -> tuple[MetricResult, List[dict]]:
    recalls = {k: [] for k in (5, 10, 20)}
    mrrs = []
    ndcgs = []
    details = []
    for case in ranked_cases:
        ranked = case["ranked_ids"]
        gold = set(case["gold_block_ids"])
        for k in (5, 10, 20):
            recalls[k].append(recall_at_k(ranked, gold, k))
        mrrs.append(mean_reciprocal_rank(ranked, gold))
        ndcgs.append(ndcg_at_k(ranked, gold, 10))
        details.append(
            {
                "question_id": case["question_id"],
                "ranked_ids": ranked[:20],
                "gold_hits": [bid for bid in ranked if bid in gold],
            }
        )
    n = max(1, len(mrrs))
    result = MetricResult(
        recall_at_5=round(sum(recalls[5]) / n, 4),
        recall_at_10=round(sum(recalls[10]) / n, 4),
        recall_at_20=round(sum(recalls[20]) / n, 4),
        mrr=round(sum(mrrs) / n, 4),
        ndcg_at_10=round(sum(ndcgs) / n, 4),
    )
    return result, details


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 3 检索 A/B 评测")
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "output" / "phase3" / "eval_corpus.jsonl"),
    )
    parser.add_argument(
        "--gold",
        default=str(ROOT / "output" / "phase3" / "golden_retrieval.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "phase3" / "retrieval_report.json"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("FINANCE_EMBED_MODEL", "BAAI/bge-m3"),
    )
    parser.add_argument("--max-k", type=int, default=20)
    args = parser.parse_args()

    corpus = [
        json.loads(line)
        for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus_ids = {item["block_id"] for item in corpus}
    golden = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    cases = golden["cases"]

    print(f"加载 dense 模型: {args.embedding_model} …", flush=True)
    dense = DenseIndex(corpus, args.embedding_model)
    sparse = SparseRetriever(corpus)
    structured = StructuredRetriever(corpus)
    hybrid = HybridRetriever(
        fusion_retriever=DenseAdapter(dense),
        sparse_retriever=sparse,
        structured_retriever=structured,
    )
    reranker = EvidenceReranker()

    strategies: dict[str, List[Dict[str, Any]]] = {
        "dense": [],
        "sparse": [],
        "structured": [],
        "hybrid": [],
        "hybrid+rerank": [],
    }
    for case in cases:
        question = str(case["question"])
        gold_ids = [
            bid for bid in case["gold_block_ids"] if bid in corpus_ids
        ]
        dense_candidates = dense.search(question, top_k=args.max_k)
        sparse_candidates = sparse.search(question, top_k=args.max_k)
        structured_candidates = structured.search(question, top_k=args.max_k)
        hybrid_candidates = hybrid.search(question, top_k=args.max_k)
        rerank_pool = hybrid.search(question, top_k=max(30, args.max_k * 2))
        rerank_candidates = reranker.rerank(
            question,
            rerank_pool,
            top_k=args.max_k,
        )
        strategies["dense"].append(
            {
                "question_id": case["question_id"],
                "gold_block_ids": gold_ids,
                "gold_unavailable": [
                    bid for bid in case["gold_block_ids"] if bid not in corpus_ids
                ],
                "ranked_ids": [item["block_id"] for item in dense_candidates],
            }
        )
        strategies["sparse"].append(
            {
                "question_id": case["question_id"],
                "gold_block_ids": gold_ids,
                "gold_unavailable": [
                    bid for bid in case["gold_block_ids"] if bid not in corpus_ids
                ],
                "ranked_ids": [item["block_id"] for item in sparse_candidates],
            }
        )
        strategies["structured"].append(
            {
                "question_id": case["question_id"],
                "gold_block_ids": gold_ids,
                "gold_unavailable": [
                    bid for bid in case["gold_block_ids"] if bid not in corpus_ids
                ],
                "ranked_ids": [item["block_id"] for item in structured_candidates],
            }
        )
        strategies["hybrid"].append(
            {
                "question_id": case["question_id"],
                "gold_block_ids": gold_ids,
                "gold_unavailable": [
                    bid for bid in case["gold_block_ids"] if bid not in corpus_ids
                ],
                "ranked_ids": [item["block_id"] for item in hybrid_candidates],
            }
        )
        strategies["hybrid+rerank"].append(
            {
                "question_id": case["question_id"],
                "gold_block_ids": gold_ids,
                "gold_unavailable": [
                    bid for bid in case["gold_block_ids"] if bid not in corpus_ids
                ],
                "ranked_ids": [item["block_id"] for item in rerank_candidates],
            }
        )

    report_section = {}
    for name, ranked_cases in strategies.items():
        metrics, details = evaluate(name, ranked_cases)
        report_section[name] = {
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "recall_at_20": metrics.recall_at_20,
            "mrr": metrics.mrr,
            "ndcg_at_10": metrics.ndcg_at_10,
            "details": details,
        }

    report = {
        "schema_version": 1,
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "gold_source": golden.get("source"),
        "corpus_blocks": len(corpus),
        "case_count": len(cases),
        "gold_blocks_in_corpus": len({bid for case in cases for bid in case["gold_block_ids"]} & corpus_ids),
        "embedding_model": args.embedding_model,
        "strategies": report_section,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report_section, ensure_ascii=False, indent=2))
    print(f"报告已写入: {output}")


if __name__ == "__main__":
    main()
