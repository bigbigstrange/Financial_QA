"""Offline retrieval check against *unreviewed* Financial-50 evidence mappings.

This is a diagnostic, not a gold-set accuracy or answer-quality evaluation.
No generation, web request, or paid API call is made.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.system.retrieval import build_retrieval_stack
from src.system.rerank import EvidenceReranker


def _metrics(rows: list[dict], k: int) -> dict:
    if not rows:
        return {"case_count": 0, "any_hit_at_k": None, "all_groups_hit_at_k": None,
                "mean_reciprocal_rank": None}
    any_hits = 0
    all_hits = 0
    reciprocal = 0.0
    for row in rows:
        groups = [set(group) for group in row["candidate_groups"]]
        ranked = row["ranked_ids"][:k]
        hits = [next((index + 1 for index, item in enumerate(ranked) if item in group), None)
                for group in groups]
        any_hits += any(rank is not None for rank in hits)
        all_hits += all(rank is not None for rank in hits)
        first = min((rank for rank in hits if rank is not None), default=None)
        reciprocal += 1.0 / first if first else 0.0
    count = len(rows)
    return {"case_count": count, "any_hit_at_k": round(any_hits / count, 4),
            "all_groups_hit_at_k": round(all_hits / count, 4),
            "mean_reciprocal_rank": round(reciprocal / count, 4)}


def evaluate(candidate_file: Path, manifest: Path, cache_dir: Path,
             model_dir: Path, output: Path, top_k: int = 20) -> dict:
    if output.exists():
        raise FileExistsError("评测报告已存在，禁止覆盖")
    if os.getenv("FINANCE_DISABLE_PAID_API") != "1":
        raise RuntimeError("必须设置 FINANCE_DISABLE_PAID_API=1")
    cases = [json.loads(line) for line in candidate_file.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not cases or any(row.get("review_status") != "pending_snapshot_review" for row in cases):
        raise ValueError("仅接受尚未审核的候选映射，不得伪装成人工 gold")
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))["snapshot_id"]
    if any(row.get("source_snapshot") != snapshot for row in cases):
        raise ValueError("候选映射与当前快照不符")

    start = time.monotonic()
    store, text, _image, hybrid = build_retrieval_stack(
        kb_path=str(manifest), cache_dir=str(cache_dir), text_model=str(model_dir),
        image_proxy_model=str(model_dir), skip_image=True, enable_graph=False,
    )
    if len(store.blocks) < 30000 or len(text.metadata) != len(store.blocks):
        raise RuntimeError("检索索引并非当前完整文本快照")
    reranker = EvidenceReranker()
    strategies = {name: [] for name in ("dense", "sparse", "structured", "hybrid", "hybrid_rerank")}
    excluded = {"web": 0, "refuse": 0, "unmapped": 0}
    for number, case in enumerate(cases, 1):
        if case["expected_refusal"]:
            excluded["refuse"] += 1
            continue
        if case["expected_source"] == "web":
            excluded["web"] += 1
            continue
        mappings = case["evidence_mappings"]
        if not mappings or any(not item["candidates"] for item in mappings):
            excluded["unmapped"] += 1
            continue
        groups = [[item["chunk_id"] for item in mapping["candidates"]]
                  for mapping in mappings]
        query = case["question"]
        hybrid_hits = hybrid.search(query, top_k=top_k, allow_remote_graph=False)
        results = {
            "dense": text.search(query, top_k=top_k),
            "sparse": hybrid.sparse_retriever.search(query, top_k=top_k),
            "structured": hybrid.structured_retriever.search(query, top_k=top_k),
            "hybrid": hybrid_hits,
        }
        results["hybrid_rerank"] = reranker.rerank(
            query, hybrid_hits, top_k=top_k,
        )
        for name, hits in results.items():
            strategies[name].append({"case_id": case["case_id"],
                                     "candidate_groups": groups,
                                     "ranked_ids": [hit["block_id"] for hit in hits]})
        if number % 10 == 0:
            print(f"候选题进度 {number}/{len(cases)}", flush=True)

    report = {
        "status": "provisional_machine_mapped_evidence_not_human_gold",
        "snapshot_id": snapshot,
        "candidate_file": str(candidate_file),
        "corpus_chunks": len(store.blocks),
        "model": str(model_dir),
        "visual_channel": "off_for_this_diagnostic",
        "remote_graph": "off_for_this_diagnostic",
        "paid_model_calls": 0,
        "web_calls": 0,
        "excluded": excluded,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "strategies": {
            name: {"at_5": _metrics(rows, 5), "at_10": _metrics(rows, 10),
                   "at_20": _metrics(rows, 20), "details": rows}
            for name, rows in strategies.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "output" / "quality" /
                        "financial50-snapshot-candidates.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "output" / "kb_full" /
                        "kb_v3_manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "output" / "kb_full" / "vector_cache")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "quality" /
                        "financial50-retrieval-provisional.json")
    args = parser.parse_args()
    result = evaluate(args.candidates, args.manifest, args.cache_dir, args.model_dir, args.output)
    print(json.dumps({"status": result["status"], "excluded": result["excluded"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "strategies": {name: {k: value[k] for k in ("at_5", "at_10", "at_20")}
                                     for name, value in result["strategies"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
