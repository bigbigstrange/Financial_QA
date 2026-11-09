#!/usr/bin/env python3
"""一键构建 KB v3：MinerU 产物 → DocIR v3 + 检索单元（chunks）。

用法：
    python scripts/build_kb_v3.py
    python scripts/build_kb_v3.py --parsed-root output/kb_v3/parsed --subset output/kb_v3/subset_100.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.chunker import ChunkConfig  # noqa: E402
from src.pipeline.kb_builder import build_kb_v3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 KB v3")
    ap.add_argument("--parsed-root", default=str(ROOT / "output" / "kb_v3" / "parsed"))
    ap.add_argument("--output-dir", default=str(ROOT / "output" / "kb_v3"))
    ap.add_argument("--docs-root", action="append", default=None,
                    help="源文件根目录，可重复传入（原生文档目录 + Office 转换产物目录）")
    ap.add_argument("--subset", action="append", default=None,
                    help="子集清单，可重复传入（原生 PDF 与 Office 转换后的子集要一起给）")
    ap.add_argument("--render-page-images", action="store_true", default=True,
                    help="转换时渲染整页图，供视觉检索通道使用（默认开启，约 50ms/页）")
    ap.add_argument("--no-page-images", dest="render_page_images",
                    action="store_false", help="关闭整页图渲染")
    ap.add_argument("--target-chars", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=0)
    ap.add_argument("--overlap-chars", type=int, default=0)
    args = ap.parse_args()

    config = ChunkConfig.from_env()
    overrides = {}
    if args.target_chars:
        overrides["target_chars"] = args.target_chars
    if args.max_chars:
        overrides["max_chars"] = args.max_chars
    if args.overlap_chars:
        overrides["overlap_chars"] = args.overlap_chars
    if overrides:
        config = ChunkConfig(**{**config.__dict__, **overrides})

    stats = build_kb_v3(
        parsed_root=Path(args.parsed_root),
        output_dir=Path(args.output_dir),
        docs_root=(
            [Path(p) for p in args.docs_root] if args.docs_root
            else [
                ROOT / "data" / "documents",
                ROOT / "output" / "kb_v3" / "converted",
            ]
        ),
        subset_path=(
            [Path(p) for p in args.subset] if args.subset
            else [
                ROOT / "output" / "kb_v3" / "subset_100.json",
                ROOT / "output" / "kb_v3" / "converted" / "subset_converted.json",
            ]
        ),
        config=config,
        render_page_images=args.render_page_images,
    )

    cs = stats.get("chunk_stats") or {}
    cov = stats.get("coverage") or {}
    print(json.dumps(
        {
            "snapshot_id": stats["snapshot_id"],
            "documents": stats["document_count"],
            "blocks": stats["block_count"],
            "chunks": stats["chunk_count"],
            "duplicate_documents_removed": stats["duplicate_documents_removed"],
            "chunk_len_p50": cs.get("chunk_len_p50"),
            "trailing_prose_end_ratio": cs.get("trailing_prose_end_ratio"),
            "coverage": cov,
            "skipped": len(stats.get("skipped") or []),
        },
        ensure_ascii=False, indent=2,
    ))
    if stats.get("skipped"):
        print("\n跳过的文档:")
        for item in stats["skipped"][:20]:
            print(f"  {item['doc_key'][:50]:50} {item['reason'][:80]}")
    print(f"\n产物目录: {args.output_dir}")


if __name__ == "__main__":
    main()
