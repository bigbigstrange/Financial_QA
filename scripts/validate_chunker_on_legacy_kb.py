"""只读验证：把旧知识库的真实块喂进 chunker-v3，对比切块前后的质量指标。

这是「切块层」单独的 A/B，不依赖 MinerU，可直接量化块粒度修复带来的收益。

用法：
    python scripts/validate_chunker_on_legacy_kb.py --max-docs 30
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.chunker import (  # noqa: E402
    ChunkConfig,
    _ends_with_sentence_punct,
    build_chunks,
)
from src.pipeline.docir import DocBlock  # noqa: E402

SENT_END = "。！？；!?;"


def legacy_block_to_docblock(raw: dict, doc_id: str, page_no: int) -> DocBlock | None:
    """把旧知识库的块转成 DocBlock，尽量保留可用字段。"""
    text = (raw.get("text") or "").strip()
    btype = str(raw.get("block_type") or "paragraph")
    if not text and btype not in {"table", "table_rows"}:
        return None
    bid = raw.get("block_id") or f"{doc_id}_p{page_no}_b"
    return DocBlock(
        block_id=str(bid),
        block_type=btype,
        text=text,
        page=page_no,
        bbox=raw.get("bbox") if isinstance(raw.get("bbox"), list) else None,
        section_path=raw.get("section_path"),
        parent_id=raw.get("parent_id"),
        caption=raw.get("caption"),
        unit=raw.get("unit"),
        table_headers=list(raw.get("table_headers") or []),
        table_rows=list(raw.get("table_rows") or []),
        table_md=raw.get("table_md"),
        formula_latex=raw.get("formula_latex"),
        structural_locator=raw.get("structural_locator") or f"legacy/{bid}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="切块层 A/B 验证（旧知识库 → chunker-v3）")
    ap.add_argument("--kb", default=str(ROOT / "output" / "knowledge_base.json"))
    ap.add_argument("--max-docs", type=int, default=30)
    ap.add_argument("--report", default=str(ROOT / "output" / "kb_v3" / "chunker_ab_report.json"))
    args = ap.parse_args()

    import ijson

    config = ChunkConfig.from_env()
    before_lens: list[int] = []
    after_lens: list[int] = []
    before_end_ok = 0
    before_total = 0
    after_end_ok = 0
    after_total = 0
    before_types: collections.Counter = collections.Counter()
    after_types: collections.Counter = collections.Counter()
    docs_done = 0
    flagged = collections.Counter()
    samples: list[dict] = []

    with open(args.kb, "rb") as fh:
        for doc in ijson.items(fh, "documents.item", use_float=True):
            if docs_done >= args.max_docs:
                break
            doc_id = str(doc.get("doc_id") or "")
            blocks: list[DocBlock] = []
            for page in doc.get("pages") or []:
                page_no = page.get("page_no")
                for raw in page.get("blocks") or []:
                    text = (raw.get("text") or "").strip()
                    if not text:
                        continue
                    before_total += 1
                    before_lens.append(len(text))
                    if text[-1] in SENT_END:
                        before_end_ok += 1
                    before_types[str(raw.get("block_type") or "paragraph")] += 1
                    db = legacy_block_to_docblock(raw, doc_id, int(page_no or 1))
                    if db is not None:
                        blocks.append(db)
            if not blocks:
                continue

            chunks = build_chunks(blocks, document_version=f"legacy-{doc_id}", config=config)
            docs_done += 1
            for c in chunks:
                after_total += 1
                after_lens.append(c.char_len)
                if _ends_with_sentence_punct(c.text):
                    after_end_ok += 1
                after_types[c.block_type] += 1
                for fl in c.quality_flags:
                    flagged[fl] += 1

            if len(samples) < 3 and len(chunks) >= 2:
                samples.append(
                    {
                        "doc_id": doc_id,
                        "before_blocks": len(blocks),
                        "after_chunks": len(chunks),
                        "first_chunk_head": chunks[0].text[:160],
                        "second_chunk_head": chunks[1].text[:160],
                        "second_overlap_chars": chunks[1].overlap_prev_chars,
                    }
                )

    def pct(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    def dist(lens: list[int]) -> dict:
        if not lens:
            return {}
        s = sorted(lens)
        return {
            "count": len(s),
            "mean": round(statistics.fmean(s), 1),
            "p50": s[len(s) // 2],
            "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
            "max": s[-1],
            "under_50": pct(sum(1 for x in s if x < 50), len(s)),
            "under_100": pct(sum(1 for x in s if x < 100), len(s)),
            "over_1000": pct(sum(1 for x in s if x > 1000), len(s)),
        }

    report = {
        "config": {
            "target_chars": config.target_chars,
            "max_chars": config.max_chars,
            "min_chars": config.min_chars,
            "overlap_chars": config.overlap_chars,
            "chunker_version": config.chunker_version,
        },
        "documents_compared": docs_done,
        "before_legacy_blocks": {"unit": "block（一个文本对象一块）", **dist(before_lens)},
        "after_chunks": {"unit": "chunk（段落聚合 + overlap）", **dist(after_lens)},
        "sentence_end_ratio": {
            "before": pct(before_end_ok, before_total),
            "after": pct(after_end_ok, after_total),
        },
        "unit_type_counts": {
            "before": dict(before_types.most_common(10)),
            "after": dict(after_types.most_common(10)),
        },
        "quality_flags_after": dict(flagged.most_common()),
        "samples": samples,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "documents_compared": docs_done,
            "before_legacy_blocks": report["before_legacy_blocks"],
            "after_chunks": report["after_chunks"],
            "sentence_end_ratio": report["sentence_end_ratio"],
            "quality_flags_after": report["quality_flags_after"],
        },
        ensure_ascii=False, indent=2,
    ))
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
