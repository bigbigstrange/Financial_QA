"""用最新的候选证据精确计算 KG 抽取范围与成本。

输出一张「成本阶梯」：核心文档 / 逐步扩大 / 全量，让范围决策有据可依。

用法：
    python scripts/plan_kg_scope.py
    python scripts/plan_kg_scope.py --candidates output/eval/gold_evidence_candidates.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KB = ROOT / "output" / "kb_v3"
EVAL = ROOT / "output" / "eval"

# DeepSeek 定价（CNY / 百万 tokens）
PRICE_IN = 2.0
PRICE_OUT = 8.0
TOKENS_PER_CHAR = 0.75      # 中文约 1 字 ≈ 0.75 token
CHUNK_TOKENS = 1100         # LightRAG 默认 chunk 1200，含重叠后有效约 1100
PROMPT_TOKENS = 3000        # 抽取提示词（含 few-shot 示例）量级
OUT_PER_CHUNK = 600         # 每个 chunk 抽出的实体/关系 JSON
MERGE_RATIO = 0.4           # 实体/关系合并摘要相对抽取的开销


def cost_of(chars: int) -> dict:
    content_tokens = chars * TOKENS_PER_CHAR
    chunks = max(1, int(content_tokens / CHUNK_TOKENS))
    extract_in = chunks * (PROMPT_TOKENS + CHUNK_TOKENS)
    extract_out = chunks * OUT_PER_CHUNK
    total_in = (extract_in * (1 + MERGE_RATIO)) / 1_000_000
    total_out = (extract_out * (1 + MERGE_RATIO)) / 1_000_000
    return {
        "chars": chars,
        "chunks": chunks,
        "in_m": round(total_in, 2),
        "out_m": round(total_out, 2),
        "cost": round(total_in * PRICE_IN + total_out * PRICE_OUT, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="规划 KG 抽取范围")
    ap.add_argument("--candidates", default=str(EVAL / "gold_evidence_candidates.json"))
    ap.add_argument("--old", default=str(EVAL / "gold_evidence_candidates.old40.json"))
    ap.add_argument("--top", type=int, default=3, help="输出前 N 份文档明细")
    args = ap.parse_args()

    # 每份文档的字符数（来自 KB v3）
    doc_chars: dict[str, int] = collections.Counter()
    with (KB / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_chars[str(item.get("source_doc"))] += len(item.get("content") or "")
    total_chars = sum(doc_chars.values())

    path = Path(args.candidates)
    if not path.is_file():
        raise SystemExit(f"缺少候选证据: {path}（先跑 build_gold_set.py --stage retrieve）")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []

    refs: collections.Counter = collections.Counter()
    for q in questions:
        for c in q.get("candidates") or []:
            doc = c.get("source_doc")
            if doc:
                refs[doc] += 1

    print("=" * 78)
    print("金标准题目引用的文档（决定 KG 抽取范围）")
    print("=" * 78)
    print(f"  题目数        : {len(questions)}")
    print(f"  覆盖文档数    : {len(refs)} / {len(doc_chars)}")
    covered_chars = sum(doc_chars.get(d, 0) for d in refs)
    print(f"  覆盖字符数    : {covered_chars:,} / {total_chars:,} "
          f"({covered_chars / max(1, total_chars):.1%})")
    print()
    print(f"  引用最多的 {args.top * 4} 份文档：")
    print(f"    {'引用':>4}  {'字符数':>10}  {'KG成本':>8}  文档")
    for name, n in refs.most_common(args.top * 4):
        chars = doc_chars.get(name, 0)
        print(f"    {n:>4}  {chars:>10,}  {cost_of(chars)['cost']:>7.1f}  {name[:52]}")

    print()
    print("=" * 78)
    print("成本阶梯（按引用次数降序累加）")
    print("=" * 78)
    print(f"  {'范围':<28} {'文档':>5} {'字符数':>12} {'chunk':>7} {'成本(CNY)':>10}")
    ladder = []
    ranked = [d for d, _ in refs.most_common()]
    for step in (10, 20, 38, 50, len(ranked)):
        if step > len(ranked):
            continue
        picked = ranked[:step]
        chars = sum(doc_chars.get(d, 0) for d in picked)
        info = cost_of(chars)
        label = f"引用最多的 {step} 份"
        marks = []
        if step == 10:
            marks.append("最小可行")
        if step >= len(ranked):
            marks.append("= 全部被引用文档")
        suffix = f"  ← {'、'.join(marks)}" if marks else ""
        print(f"  {label:<28} {step:>5} {chars:>12,} {info['chunks']:>7,} "
              f"{info['cost']:>10.1f}{suffix}")
        ladder.append({"step": step, "docs": step, "chars": chars, **info})
    full = cost_of(total_chars)
    print(f"  {'全量 KB v3':<28} {len(doc_chars):>5} {total_chars:>12,} "
          f"{full['chunks']:>7,} {full['cost']:>10.1f}  ← 完整覆盖")

    # 与旧候选对照
    old_path = Path(args.old)
    if old_path.is_file():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_refs = set()
        for q in old.get("questions") or []:
            for c in q.get("candidates") or []:
                if c.get("source_doc"):
                    old_refs.add(c["source_doc"])
        new_refs = set(refs)
        print()
        print("=" * 78)
        print("与旧候选（40 份 KB 时检索得到）的对照")
        print("=" * 78)
        print(f"  旧候选文档数 : {len(old_refs)}")
        print(f"  新候选文档数 : {len(new_refs)}")
        added = sorted(new_refs - old_refs)
        removed = sorted(old_refs - new_refs)
        print(f"  新增命中     : {len(added)} 份")
        for name in added[:8]:
            print(f"      + {name[:64]}")
        print(f"  不再命中     : {len(removed)} 份")
        for name in removed[:5]:
            print(f"      - {name[:64]}")

    out = EVAL / "kg_scope_plan.json"
    out.write_text(
        json.dumps(
            {
                "candidates": str(path),
                "questions": len(questions),
                "referenced_docs": len(refs),
                "covered_chars": covered_chars,
                "total_chars": total_chars,
                "prices": {"in_per_m": PRICE_IN, "out_per_m": PRICE_OUT},
                "ladder": ladder,
                "full_kb": {"docs": len(doc_chars), **full},
                "top_referenced": [
                    {"doc": d, "refs": n, "chars": doc_chars.get(d, 0),
                     "cost": cost_of(doc_chars.get(d, 0))["cost"]}
                    for d, n in refs.most_common(30)
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n方案已写入: {out}")


if __name__ == "__main__":
    main()
