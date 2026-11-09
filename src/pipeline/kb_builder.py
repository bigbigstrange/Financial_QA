"""KB v3 构建：MinerU 解析产物 → 可检索的知识库（DocIR v3 + chunks）。

产物（全部落在 `output/kb_v3/`）：

    docir/<doc_key>.docir.json     每份文档的 DocIR v3（不可变、可溯源）
    chunks.jsonl                   检索单元（一行一个 chunk，供 FAISS/BM25/LightRAG 消费）
    kb_v3_manifest.json            知识快照（复用 src/pipeline/snapshots.py）
    parse_quality_v3.json          聚合质量报告（文档级 + chunk 级）
    kb_v3_report.md                人工复核报告

与旧链路的三个关键差异：
    1. **去重**：按源文件 SHA-256 去重，同一内容只入库一次
       （旧知识库 1146 个文档 = 572 唯一 × 2，整库重复两份）；
    2. **检索单元是 chunk 而非 block**：块长中位数从 24 字提升到 400+ 字，
       且带句对齐 overlap；
    3. **质量硬门禁**：`validate_v2()` 失败的文档不入库，并记录失败原因。

用法：
    python scripts/build_kb_v3.py
    python scripts/build_kb_v3.py --parsed-root output/kb_v3/parsed --output-dir output/kb_v3
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .chunker import ChunkConfig, build_chunks, chunk_stats
from .docir import DocIR
from .identity import ParseContext, artifact_key
from .mineru_adapter import MineruUnavailable, middle_json_to_docir
from .quality import assess_document
from .snapshots import (
    SnapshotConfig,
    SnapshotManifest,
    VersionArtifact,
    sha256_file,
    write_manifest_atomic,
)

KB_SCHEMA_VERSION = 3


def _load_subset_map(
    subset_path: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    """doc_key → 子集记录（用于把解析产物映射回原始文件）。

    支持传入单个路径或路径列表：最终知识库同时包含「原生 PDF」与
    「Office 转换后的 PDF」两个子集，必须合并成一张映射表才能一次构建。
    多个子集之间若有同名 doc_key，后加载的覆盖先加载的（后者通常是转换产物，
    带有 origin_* 溯源信息，优先保留）。
    """
    if not subset_path:
        return {}
    paths: List[Path] = []
    if isinstance(subset_path, (str, Path)):
        paths = [Path(subset_path)]
    else:
        paths = [Path(p) for p in subset_path]

    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("files", []):
            merged[artifact_key(rec["rel_path"])] = rec
    return merged


def _rebase_refs(doc_ir: DocIR, *, parse_dir: Path, kb_root: Path) -> None:
    """把 `image_ref` / `figure_ref` 统一成相对 KB 根目录的 POSIX 路径。

    解析产物里的引用是相对 `extracted/` 的（如 `images/page_0_full.png`），
    而知识库构建后 DocIR 位于 `docir/` 子目录 —— 必须换基，否则检索层
    永远解析不到页图（旧知识库正是因为页图路径失效，视觉通道 100% 死掉）。
    """

    def _rebase(ref: Optional[str]) -> Optional[str]:
        if not ref:
            return ref
        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = parse_dir / ref
        try:
            return candidate.resolve().relative_to(kb_root.resolve()).as_posix()
        except ValueError:
            return str(candidate.resolve())

    for page in doc_ir.pages:
        if page.meta:
            ref = page.meta.get("image_ref")
            if ref:
                page.meta["image_ref"] = _rebase(ref)
    for block in doc_ir.all_blocks():
        meta = block.meta or {}
        for key in ("figure_ref", "mineru_image_ref", "image_ref"):
            if meta.get(key):
                meta[key] = _rebase(meta[key])


def build_kb_v3(
    parsed_root: Path,
    output_dir: Path,
    *,
    docs_root: Optional[Any] = None,
    subset_path: Optional[Any] = None,
    config: Optional[ChunkConfig] = None,
    render_page_images: bool = False,
) -> Dict[str, Any]:
    """遍历 MinerU 产物构建 KB v3。返回汇总统计。

    `docs_root` 支持传多个根目录：原生文档在 `data/documents/`，
    Office 转换产物在 `output/kb_v3/converted/`，两者都要能找到源文件。
    """
    parsed_root = Path(parsed_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docir_dir = output_dir / "docir"
    docir_dir.mkdir(parents=True, exist_ok=True)

    config = config or ChunkConfig.from_env()
    subset = _load_subset_map(subset_path)

    roots: List[Path] = []
    if docs_root:
        if isinstance(docs_root, (str, Path)):
            roots = [Path(docs_root)]
        else:
            roots = [Path(r) for r in docs_root]
    if not roots:
        roots = [Path(".")]

    def _resolve_source(rel_path: str) -> Optional[Path]:
        for root in roots:
            candidate = root / rel_path
            if candidate.is_file():
                return candidate
        return None

    mids = sorted(
        p for p in parsed_root.rglob("extracted/middle_json.json")
        # 分片解析会留下 shard_xxxxx/extracted/middle_json.json，
        # 那是同一份文档的片段（已由 merge_shard_middles 合并到上级目录），必须排除
        if not any(part.startswith("shard_") for part in p.parts)
    )
    documents: List[Dict[str, Any]] = []
    all_chunks: List[Dict[str, Any]] = []
    quality_docs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    seen_sha: Dict[str, str] = {}
    seen_version: Dict[str, str] = {}

    jobs = [(mid.parents[1].name, mid) for mid in mids]
    # Plain text needs no OCR/layout service, but must still share the same
    # identity, quality gate, chunker and manifest path as parsed documents.
    jobs.extend((key, None) for key, rec in subset.items()
                if str(rec.get("ext", "")).lower() == ".txt")

    chunks_fh = (output_dir / "chunks.jsonl").open("w", encoding="utf-8")
    try:
        for doc_key, mid in jobs:
            rec = subset.get(doc_key)
            if rec is None:
                skipped.append({"doc_key": doc_key, "reason": "not_in_subset"})
                continue

            rel_path = rec["rel_path"]
            src = _resolve_source(rel_path)
            if src is None:
                skipped.append({"doc_key": doc_key, "reason": "source_missing"})
                continue

            context = ParseContext.for_file(
                src,
                document_id=f"doc_{doc_key}",
                parser_config_hash="parser-v3",
                chunker_version=config.chunker_version,
            )
            # 去重：同一内容只入库一次（旧知识库整库重复两份）
            if context.source_sha256 in seen_sha:
                skipped.append(
                    {"doc_key": doc_key, "reason": f"duplicate_of:{seen_sha[context.source_sha256]}"}
                )
                continue

            try:
                if mid is None:
                    from .parsers import PlainTextParser
                    doc_ir = PlainTextParser().parse(src, context=context)
                else:
                    middle = json.loads(mid.read_text(encoding="utf-8"))
                    doc_ir = middle_json_to_docir(
                        middle,
                        file_path=src,
                        parsed_dir=mid.parent,
                        context=context,
                        tier=str(rec.get("tier") or "standard"),
                        render_page_images=render_page_images,
                    )
            except (MineruUnavailable, KeyError, ValueError) as exc:
                skipped.append({"doc_key": doc_key, "reason": f"convert_failed:{exc}"[:200]})
                continue

            # 转换来的文档：把文件名/格式还原成原始件，避免溯源丢失
            origin_name = rec.get("origin_name")
            if origin_name:
                doc_ir.source_name = origin_name
                doc_ir.meta["origin_name"] = origin_name
                doc_ir.meta["origin_rel_path"] = rec.get("origin_rel_path")
                doc_ir.meta["origin_ext"] = rec.get("origin_ext")
                doc_ir.meta["converted_from_office"] = True

            # 把页图/图表引用重写成「相对 KB 根目录」的路径。
            # 解析阶段产出的是相对 extracted/ 的路径，而 DocIR 落在 docir/ 下，
            # 直接拼接会解析不到；统一成相对 KB 根目录后整棵 KB 目录可搬迁。
            _rebase_refs(doc_ir, parse_dir=mid.parent if mid else src.parent, kb_root=output_dir)

            report = assess_document(doc_ir)
            quality_docs.append({"doc_key": doc_key, **report.to_dict()})
            if report.status == "failed":
                skipped.append(
                    {"doc_key": doc_key, "reason": "quality_gate:" + ",".join(report.failure_reasons)}
                )
                continue
            if doc_ir.document_version in seen_version:
                skipped.append({"doc_key": doc_key, "reason": "duplicate_version"})
                continue

            seen_sha[context.source_sha256] = doc_key
            seen_version[doc_ir.document_version] = doc_key

            # 落 DocIR（不可变，可溯源）
            docir_path = docir_dir / f"{doc_key}.docir.json"
            docir_path.write_text(
                json.dumps(doc_ir.to_dict(), ensure_ascii=False), encoding="utf-8"
            )

            # 切块（检索单元）
            chunks = build_chunks(
                doc_ir.all_blocks(),
                document_version=doc_ir.document_version,
                config=config,
            )
            for chunk in chunks:
                item = chunk.to_retrieval_item(
                    source_doc=doc_ir.source_name or src.name,
                    document_id=doc_ir.document_id,
                    document_version=doc_ir.document_version,
                    snapshot_id=None,
                )
                item["source_format"] = doc_ir.source_format
                item["docir_path"] = str(docir_path.relative_to(output_dir))
                item["mineru_parse_mode"] = doc_ir.meta.get("mineru_parse_mode")
                all_chunks.append(item)
                chunks_fh.write(json.dumps(item, ensure_ascii=False) + "\n")

            documents.append(
                {
                    "doc_key": doc_key,
                    "document_id": doc_ir.document_id,
                    "document_version": doc_ir.document_version,
                    "source_name": doc_ir.source_name or src.name,
                    "source_format": doc_ir.source_format,
                    "source_sha256": context.source_sha256,
                    "source_size_bytes": context.source_size_bytes,
                    "parser_name": doc_ir.parser_name,
                    "parser_version": doc_ir.parser_version,
                    "mineru_tier": doc_ir.meta.get("mineru_tier"),
                    "mineru_parse_mode": doc_ir.meta.get("mineru_parse_mode"),
                    "page_count": doc_ir.total_pages,
                    "block_count": len(doc_ir.all_blocks()),
                    "chunk_count": len(chunks),
                    "table_count": doc_ir.meta.get("mineru_table_count", 0),
                    "figure_count": doc_ir.meta.get("mineru_figure_count", 0),
                    "docir_path": str(docir_path.relative_to(output_dir)),
                    "tag": rec.get("topic"),
                    "org": rec.get("org"),
                    "origin_name": doc_ir.meta.get("origin_name"),
                    "origin_rel_path": doc_ir.meta.get("origin_rel_path"),
                    "origin_ext": doc_ir.meta.get("origin_ext"),
                    "converted_from_office": bool(doc_ir.meta.get("converted_from_office")),
                }
            )
    finally:
        chunks_fh.close()

    # ---- 快照 manifest（复用阶段 2 契约）----
    snapshot_id: Optional[str] = None
    if documents:
        artifacts = [
            VersionArtifact(
                document_id=doc["document_id"],
                document_version=doc["document_version"],
                docir_path=doc["docir_path"],
                block_count=int(doc["block_count"]),
                docir_sha256=sha256_file(output_dir / doc["docir_path"]),
            )
            for doc in documents
        ]
        manifest = SnapshotManifest.build(
            artifacts,
            SnapshotConfig(
                docir_schema_version=3,
                chunker_version=config.chunker_version,
                parser_versions_hash="parsers-v3",
            ),
        )
        write_manifest_atomic(manifest, output_dir / "kb_v3_manifest.json")
        snapshot_id = manifest.snapshot_id
        for item in all_chunks:
            item["snapshot_id"] = snapshot_id
        # 回填 snapshot_id 到 chunks.jsonl
        (output_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks) + "\n",
            encoding="utf-8",
        )

    stats = {
        "kb_schema_version": KB_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "chunker_config": asdict(config),
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "block_count": sum(d["block_count"] for d in documents),
        "duplicate_documents_removed": sum(
            1 for s in skipped if s["reason"].startswith("duplicate_of")
        ),
        "skipped": skipped,
        "chunk_stats": chunk_stats_from_items(all_chunks),
        "coverage": {
            "documents_with_tables": sum(1 for d in documents if d["table_count"]),
            "documents_with_figures": sum(1 for d in documents if d["figure_count"]),
            "ocr_documents": sum(
                1 for d in documents if d["mineru_parse_mode"] == "ocr"
            ),
            "converted_from_office": sum(
                1 for d in documents if d.get("converted_from_office")
            ),
            "sections_present": sum(
                1 for c in all_chunks if c.get("section_path")
            ),
            "chunks_with_overlap": sum(
                1 for c in all_chunks if (c.get("overlap_prev_chars") or 0) > 0
            ),
        },
    }

    (output_dir / "kb_v3.json").write_text(
        json.dumps(
            {
                "schema_version": KB_SCHEMA_VERSION,
                "snapshot_id": snapshot_id,
                "config": asdict(config),
                "documents": documents,
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "parse_quality_v3.json").write_text(
        json.dumps(
            {
                "documents": quality_docs,
                "aggregate": {
                    "document_count": len(quality_docs),
                    "failed": sum(1 for q in quality_docs if q["status"] == "failed"),
                    "passed": sum(1 for q in quality_docs if q["status"] != "failed"),
                    "block_type_counts": _sum_counts(
                        q.get("block_type_counts") or {} for q in quality_docs
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "kb_v3_report.md").write_text(
        _render_report(stats, documents), encoding="utf-8"
    )
    return stats


def _sum_counts(counters: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for counter in counters:
        for key, value in counter.items():
            total[key] = total.get(key, 0) + int(value)
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))


def chunk_stats_from_items(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """对已落盘（dict 形式）的 chunk 计算质量指标。"""
    from .chunker import Chunk

    rebuilt = [
        Chunk(
            chunk_id=it["chunk_id"],
            text=it.get("content") or "",
            block_ids=list(it.get("block_ids") or []),
            page_start=it.get("page_start"),
            page_end=it.get("page_end"),
            bbox_union=it.get("bbox"),
            section_path=it.get("section_path"),
            block_type=it.get("type") or "paragraph",
            parent_block_id=it.get("parent_block_id"),
            char_len=int(it.get("char_len") or 0),
            token_estimate=int(it.get("token_estimate") or 0),
            overlap_prev_chars=int(it.get("overlap_prev_chars") or 0),
            quality_flags=list(it.get("quality_flags") or []),
            trailing_prose_end=it.get("trailing_prose_end"),
        )
        for it in items
    ]
    return chunk_stats(rebuilt)


def _render_report(stats: Dict[str, Any], documents: Sequence[Dict[str, Any]]) -> str:
    cs = stats.get("chunk_stats") or {}
    cov = stats.get("coverage") or {}
    lines = [
        "# KB v3 构建报告",
        "",
        f"- snapshot_id: `{stats.get('snapshot_id')}`",
        f"- 文档数: {stats['document_count']}",
        f"- 可检索块: {stats['block_count']:,}",
        f"- **检索单元（chunk）: {stats['chunk_count']:,}**",
        f"- 去重移除: {stats['duplicate_documents_removed']} 份",
        "",
        "## 切块质量",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| chunk 长度中位数 | {cs.get('chunk_len_p50')} 字 |",
        f"| chunk 长度均值 | {cs.get('chunk_len_mean')} 字 |",
        f"| chunk 长度 p90 | {cs.get('chunk_len_p90')} 字 |",
        f"| **正文收尾率** | **{cs.get('trailing_prose_end_ratio')}** |",
        f"| 散文块句末完整率 | {cs.get('prose_sentence_end_ratio')} |",
        f"| 过短 chunk | {cs.get('chunk_under_min_count')} |",
        f"| 硬切 chunk | {cs.get('chunk_hard_split_count')} |",
        f"| 带 overlap 的 chunk | {cs.get('chunk_overlap_pair_count')} |",
        "",
        "## chunk 类型分布",
        "",
        "| 类型 | 数量 | 长度中位数 | 句末完整率 |",
        "|---|---:|---:|---:|",
    ]
    types = cs.get("chunk_type_counts") or {}
    p50 = cs.get("chunk_len_p50_by_type") or {}
    endr = cs.get("sentence_end_ratio_by_type") or {}
    for t, n in sorted(types.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {n} | {p50.get(t)} | {endr.get(t)} |")
    lines += [
        "",
        "## 覆盖情况",
        "",
        f"- 含表格的文档: {cov.get('documents_with_tables')}/{stats['document_count']}",
        f"- 含图表的文档: {cov.get('documents_with_figures')}/{stats['document_count']}",
        f"- 走 OCR 的文档: {cov.get('ocr_documents')}/{stats['document_count']}",
        f"- 由 Office 转换而来: {cov.get('converted_from_office')}/{stats['document_count']}",
        f"- 有 section_path 的 chunk: {cov.get('sections_present')}/{stats['chunk_count']}",
        f"- 有 overlap 的 chunk: {cov.get('chunks_with_overlap')}/{stats['chunk_count']}",
        "",
    ]
    if stats.get("skipped"):
        lines += ["## 跳过/失败的文档", ""]
        for item in stats["skipped"][:50]:
            lines.append(f"- `{item['doc_key']}` — {item['reason']}")
        lines.append("")
    lines += [
        "## 文档清单",
        "",
        "| 文档 | 页 | 块 | chunk | 表 | 图 | 模式 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for doc in documents:
        lines.append(
            f"| {doc['source_name'][:48]} | {doc['page_count']} | {doc['block_count']} | "
            f"{doc['chunk_count']} | {doc['table_count']} | {doc['figure_count']} | "
            f"{doc.get('mineru_parse_mode')} |"
        )
    return "\n".join(lines) + "\n"
