"""阶段 A 验收：用真实 MinerU 产出验证「解析 + 切块」链路的量化指标。

对照基线（旧链路 PyMuPDF 行级碎片，实测 30 份文档）：
    块长中位数 24 字 · <50 字占 78.2% · 以句末标点结尾仅 46.4%
    全库表格 236 个 · OCR 块 0 个 · section_path 覆盖 0%

本脚本对已完成 MinerU 解析的文档跑完整链路（middle_json → DocIR v3 → chunker-v3），
输出可直接对照验收标准的指标表。

用法：
    python scripts/validate_mineru_pipeline.py
    python scripts/validate_mineru_pipeline.py --max-docs 20
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
    chunk_stats,
)
from src.pipeline.docir import DocIR  # noqa: E402
from src.pipeline.identity import ParseContext  # noqa: E402
from src.pipeline.mineru_adapter import middle_json_to_docir  # noqa: E402

SENT_END = "。！？；"
LEGACY_BASELINE = {
    "block_len_p50": 24,
    "block_under_50_ratio": 0.782,
    "block_sentence_end_ratio": 0.464,
    "kept_blocks_per_doc_mean": 235.9,   # 7076 / 30
}


def pct(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _aggregate_chunk_stats(chunks: list) -> dict:
    """全语料层面的分类型统计（表格/图注本来就不以句号结尾，必须分层看）。"""
    if not chunks:
        return {}
    stats = chunk_stats(chunks)
    return {
        "sentence_end_ratio_by_type": stats.get("sentence_end_ratio_by_type", {}),
        "chunk_len_p50_by_type": stats.get("chunk_len_p50_by_type", {}),
        "chunk_len_mean_all": stats.get("chunk_len_mean"),
        "chunk_len_p50_all": stats.get("chunk_len_p50"),
        "prose_chunk_count": stats.get("prose_chunk_count"),
        "prose_sentence_end_ratio": stats.get("prose_sentence_end_ratio"),
        "trailing_prose_checked": stats.get("trailing_prose_checked"),
        "trailing_prose_cut_count": stats.get("trailing_prose_cut_count"),
        "trailing_prose_end_ratio": stats.get("trailing_prose_end_ratio"),
    }


def dist(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "mean": round(statistics.fmean(s), 1),
        "p50": s[n // 2],
        "p90": s[min(n - 1, int(n * 0.9))],
        "max": s[-1],
        "under_50": pct(sum(1 for x in s if x < 50), n),
        "under_100": pct(sum(1 for x in s if x < 100), n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed-root", default=str(ROOT / "output" / "kb_v3" / "parsed"))
    ap.add_argument("--report", default=str(ROOT / "output" / "kb_v3" / "phaseA_validation.json"))
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    config = ChunkConfig.from_env()

    mids = sorted(p for p in parsed_root.rglob("extracted/middle_json.json"))
    if args.max_docs:
        mids = mids[: args.max_docs]
    print(f"发现 {len(mids)} 份 MinerU 产出，开始链路验证…", flush=True)

    block_lens: list[int] = []
    chunk_lens: list[int] = []
    block_types: collections.Counter = collections.Counter()
    chunk_types: collections.Counter = collections.Counter()
    noise: collections.Counter = collections.Counter()
    block_end_ok = 0
    chunk_end_ok = 0
    prose_end_ok = 0
    prose_total = 0
    section_ok = 0
    section_total = 0
    table_rows_total = 0
    figure_total = 0
    table_md_ok = 0
    overlap_pairs = 0
    flags: collections.Counter = collections.Counter()
    per_doc: list[dict] = []
    docs_ok = 0
    docs_failed = 0
    parse_modes: collections.Counter = collections.Counter()
    all_chunks: list = []

    for mid in mids:
        try:
            middle = json.loads(mid.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {mid}: {exc}")
            docs_failed += 1
            continue
        parsed_dir = mid.parent
        stem = mid.parents[1].name
        ctx = ParseContext(
            document_id=f"doc-{stem}",
            source_name=f"{stem}.pdf",
            source_sha256="0" * 64,
            source_size_bytes=0,
            parser_config_hash="parser-v3",
            chunker_version=config.chunker_version,
        )
        try:
            doc_ir = middle_json_to_docir(
                middle,
                file_path=Path(f"{stem}.pdf"),
                parsed_dir=parsed_dir,
                context=ctx,
                tier="standard",
                render_page_images=False,   # 验证指标不需要页图，省时间
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {stem}: {type(exc).__name__}: {exc}")
            docs_failed += 1
            continue

        docs_ok += 1
        parse_modes[str(doc_ir.meta.get("mineru_parse_mode"))] += 1
        for k, v in (doc_ir.meta.get("mineru_noise_counts") or {}).items():
            noise[k] += v

        blocks = doc_ir.all_blocks()
        for b in blocks:
            text = (b.text or "").strip()
            block_types[b.block_type] += 1
            if b.block_type in {"table", "table_rows"}:
                if b.block_type == "table_rows":
                    table_rows_total += 1
                    if (b.table_md or "").strip():
                        table_md_ok += 1
                continue
            if b.block_type == "figure":
                figure_total += 1
            if not text:
                continue
            block_lens.append(len(text))
            section_total += 1
            if b.section_path:
                section_ok += 1
            if text[-1] in SENT_END:
                block_end_ok += 1
            if b.block_type == "paragraph" and len(text) >= 100:
                prose_total += 1
                if text[-1] in SENT_END:
                    prose_end_ok += 1

        chunks = build_chunks(blocks, document_version=doc_ir.document_version,
                              config=config)
        all_chunks.extend(chunks)
        for c in chunks:
            chunk_lens.append(c.char_len)
            chunk_types[c.block_type] += 1
            if _ends_with_sentence_punct(c.text):
                chunk_end_ok += 1
            if c.overlap_prev_chars > 0:
                overlap_pairs += 1
            for f in c.quality_flags:
                flags[f] += 1

        cs = chunk_stats(chunks)
        per_doc.append({
            "doc": stem[:60],
            "parse_mode": doc_ir.meta.get("mineru_parse_mode"),
            "pages": doc_ir.total_pages,
            "blocks": len(blocks),
            "chunks": len(chunks),
            "tables": doc_ir.meta.get("mineru_table_count"),
            "figures": doc_ir.meta.get("mineru_figure_count"),
            "chunk_p50": cs.get("chunk_len_p50"),
            "chunk_sentence_end_ratio": cs.get("chunk_sentence_end_ratio"),
        })

    report = {
        "documents_converted": docs_ok,
        "documents_failed": docs_failed,
        "mineru_parse_modes": dict(parse_modes),
        "chunker_config": {
            "target_chars": config.target_chars,
            "max_chars": config.max_chars,
            "min_chars": config.min_chars,
            "overlap_chars": config.overlap_chars,
        },
        "legacy_baseline": LEGACY_BASELINE,
        "BLOCK_LEVEL": {
            "total_indexable_blocks": len(block_lens),
            "block_len": dist(block_lens),
            "sentence_end_ratio": pct(block_end_ok, len(block_lens)),
            "paragraph_over_100_sentence_end_ratio": pct(prose_end_ok, prose_total),
            "section_path_coverage": pct(section_ok, section_total),
            "block_type_counts": dict(block_types.most_common(20)),
        },
        "CHUNK_LEVEL": {
            "total_chunks": len(chunk_lens),
            "chunk_len": dist(chunk_lens),
            "sentence_end_ratio": pct(chunk_end_ok, len(chunk_lens)),
            "overlap_pair_count": overlap_pairs,
            "chunk_type_counts": dict(chunk_types.most_common(10)),
            "quality_flags": dict(flags.most_common()),
            **_aggregate_chunk_stats(all_chunks),
        },
        "COVERAGE": {
            "table_row_groups": table_rows_total,
            "table_row_groups_with_markdown": table_md_ok,
            "figures": figure_total,
            "noise_blocks_removed": dict(noise.most_common()),
        },
        "per_document": per_doc,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 74)
    print(f"链路验证：成功 {docs_ok} 份，失败 {docs_failed} 份")
    print("=" * 74)
    bl = report["BLOCK_LEVEL"]
    cl = report["CHUNK_LEVEL"]
    print(f"[块级]  可检索块 {bl['total_indexable_blocks']:,}")
    print(f"        长度中位数 {bl['block_len'].get('p50')} 字 "
          f"(旧链路 {LEGACY_BASELINE['block_len_p50']})")
    print(f"        <50 字占比 {bl['block_len'].get('under_50'):.1%} "
          f"(旧链路 {LEGACY_BASELINE['block_under_50_ratio']:.1%})")
    print(f"        句末完整率 {bl['sentence_end_ratio']:.1%} "
          f"(旧链路 {LEGACY_BASELINE['block_sentence_end_ratio']:.1%})")
    print(f"        正文(>=100字)句末完整率 {bl['paragraph_over_100_sentence_end_ratio']:.1%}")
    print(f"        section_path 覆盖率 {bl['section_path_coverage']:.1%}（旧链路 0%）")
    print(f"        块类型 {bl['block_type_counts']}")
    print()
    print(f"[切块级] chunk 总数 {cl['total_chunks']:,}")
    print(f"        长度中位数 {cl['chunk_len'].get('p50')}，均值 {cl['chunk_len'].get('mean')}")
    print(f"        ★ 正文收尾率 {cl.get('trailing_prose_end_ratio'):.1%} "
          f"(被切断 {cl.get('trailing_prose_cut_count')}/{cl.get('trailing_prose_checked')}) "
          f" ← 核心验收指标")
    print(f"        散文块句末完整率 {cl.get('prose_sentence_end_ratio'):.1%} "
          f"(n={cl.get('prose_chunk_count')})")
    print(f"        全部 chunk 句末完整率 {cl['sentence_end_ratio']:.1%}（含元数据/表格/图注）")
    print(f"        分类型句末完整率: {cl.get('sentence_end_ratio_by_type')}")
    print(f"        分类型长度中位数: {cl.get('chunk_len_p50_by_type')}")
    print(f"        带 overlap 的 chunk 数 {cl['overlap_pair_count']:,}")
    print(f"        质量标记 {cl['quality_flags']}")
    print()
    cov = report["COVERAGE"]
    print(f"[覆盖]  表格行组 {cov['table_row_groups']:,}（旧链路全库表格块仅 236）")
    print(f"        图表 {cov['figures']:,}（旧链路 0）")
    print(f"        已剔除噪声块 {cov['noise_blocks_removed']}")
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
