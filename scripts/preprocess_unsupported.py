#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量解析此前未支持的格式（.doc/.ppt/.pps/.xls/.xlsx/.pptx/.rtf），
并合并进已有 output/knowledge_base.json（流式写出，避免整库载入内存）。

用法:
  python scripts/preprocess_unsupported.py
  python scripts/preprocess_unsupported.py --limit 20   # 先小批量试跑
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_paths import strip_roaming_user_site_from_sys_path

strip_roaming_user_site_from_sys_path()

import importlib.util

_preprocess_path = Path(__file__).resolve().parent / "preprocess.py"
_spec = importlib.util.spec_from_file_location("finance_preprocess", _preprocess_path)
_preprocess = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_preprocess)
_materialize_page_images = _preprocess._materialize_page_images
_parser_name = _preprocess._parser_name
_resolve_input_path = _preprocess._resolve_input_path

from src.pipeline import build_default_parsers

UNSUPPORTED_EXTS = {".doc", ".ppt", ".pps", ".xls", ".xlsx", ".pptx", ".rtf"}


def _doc_dir_name(file_path: Path) -> str:
    # 避免与同名 .docx/.pdf 目录冲突
    return f"{file_path.stem}__{file_path.suffix.lower().lstrip('.')}"


def _collect_files(input_path: Path) -> list[Path]:
    files: list[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() in UNSUPPORTED_EXTS:
            files.append(input_path)
        return files
    for ext in sorted(UNSUPPORTED_EXTS):
        files.extend(input_path.rglob(f"*{ext}"))
    # Windows 大小写不敏感时可能重复
    uniq = []
    seen = set()
    for p in files:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return sorted(uniq, key=lambda x: str(x).lower())


def _existing_source_paths(kb_path: Path) -> set[str]:
    if not kb_path.is_file():
        return set()
    try:
        import ijson
    except ImportError:
        return set()
    paths: set[str] = set()
    with kb_path.open("rb") as f:
        for doc in ijson.items(f, "documents.item"):
            src = (doc.get("source_path") or "").strip()
            if src:
                paths.add(os.path.normcase(os.path.abspath(src)))
            # 兼容仅有 doc_id 的情况
            did = (doc.get("doc_id") or "").strip()
            if did:
                paths.add(did.lower())
    return paths


def _json_default(obj):
    # ijson 数字默认为 Decimal；Excel 可能带 datetime
    try:
        from decimal import Decimal

        if isinstance(obj, Decimal):
            return float(obj)
    except Exception:
        pass
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def _stream_merge_kb(
    old_kb: Path,
    new_docs: list[dict],
    out_kb: Path,
    stats: dict,
) -> None:
    import ijson

    tmp = out_kb.with_suffix(".json.tmp")
    created = datetime.now().isoformat()
    with tmp.open("w", encoding="utf-8") as fout:
        fout.write('{\n  "version": "2.0",\n')
        fout.write(f'  "created_time": {json.dumps(created, ensure_ascii=False)},\n')
        fout.write('  "documents": [\n')
        first = True
        old_count = 0
        if old_kb.is_file() and old_kb.stat().st_size > 0:
            with old_kb.open("rb") as fin:
                try:
                    doc_iter = ijson.items(fin, "documents.item", use_float=True)
                except TypeError:
                    fin.seek(0)
                    doc_iter = ijson.items(fin, "documents.item")
                for doc in doc_iter:
                    if not first:
                        fout.write(",\n")
                    fout.write(json.dumps(doc, ensure_ascii=False, default=_json_default))
                    first = False
                    old_count += 1
        for doc in new_docs:
            if not first:
                fout.write(",\n")
            fout.write(json.dumps(doc, ensure_ascii=False, default=_json_default))
            first = False
        fout.write("\n  ],\n")
        merged_stats = {
            "total_files": old_count + stats.get("success_files", 0) + stats.get("failed_files", 0),
            "success_files": old_count + stats.get("success_files", 0),
            "failed_files": stats.get("failed_files", 0),
            "skipped_files": stats.get("skipped_files", 0),
            "incremental_added": stats.get("success_files", 0),
            "previous_docs": old_count,
        }
        fout.write(
            f'  "stats": {json.dumps(merged_stats, ensure_ascii=False, default=_json_default)},\n'
        )
        fout.write(
            f'  "failures": {json.dumps(stats.get("failures", []), ensure_ascii=False, default=_json_default)}\n'
        )
        fout.write("}\n")
    tmp.replace(out_kb)


def main() -> int:
    code_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="增量解析未支持格式并合并知识库")
    parser.add_argument(
        "--input",
        "-i",
        default=str(code_root / "data" / "documents" / "dataset" / "数据集"),
        help="输入目录",
    )
    parser.add_argument("--output", "-o", default=str(code_root / "output"), help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个（调试用，0=全部）")
    parser.add_argument(
        "--tesseract-cmd",
        default=r"F:\Tools\tesseract\tesseract.exe",
        help="Tesseract 路径（本脚本基本不用 OCR，保留兼容）",
    )
    parser.add_argument("--temp-dir", default=r"F:\temp", help="临时目录")
    args = parser.parse_args()

    if args.tesseract_cmd and Path(args.tesseract_cmd).exists():
        os.environ["TESSERACT_CMD"] = args.tesseract_cmd
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = args.temp_dir
        os.environ["TMP"] = args.temp_dir

    input_path = _resolve_input_path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    kb_path = output_path / "knowledge_base.json"

    print("=" * 56)
    print("增量解析未支持格式")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print("=" * 56)

    files = _collect_files(input_path)
    print(f"待解析候选: {len(files)}")
    existing = _existing_source_paths(kb_path)
    print(f"已有知识库路径/ID 索引: {len(existing)}")

    todo = []
    skipped = 0
    for fp in files:
        abs_n = os.path.normcase(os.path.abspath(str(fp)))
        # 仅按绝对路径去重，避免同名 .docx 挡住新增 .doc
        if abs_n in existing:
            skipped += 1
            continue
        todo.append(fp)

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"本次实际处理: {len(todo)}（跳过已存在路径 {skipped}）")

    parsers = build_default_parsers(legacy_output_dir=output_path)
    # 只保留能处理目标扩展名的解析器
    parsers = [p for p in parsers if UNSUPPORTED_EXTS & set(getattr(p, "SUPPORTED_EXTS", set()))]

    new_docs: list[dict] = []
    md_parts: list[str] = []
    failures: list[dict] = []
    parser_stats: dict = {}
    for p in parsers:
        parser_stats[_parser_name(p)] = {"attempts": 0, "success": 0, "failed": 0}

    for i, file_path in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {file_path.name}")
        doc_dir = output_path / _doc_dir_name(file_path)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)

        candidates = [p for p in parsers if p.supports(file_path)]
        if not candidates:
            failures.append({"file": str(file_path), "error": "no_parser"})
            print("  ❌ 无解析器")
            continue

        doc_ir = None
        attempts = []
        for p in candidates:
            name = _parser_name(p)
            parser_stats[name]["attempts"] += 1
            try:
                doc_ir = p.parse(file_path)
                # 覆盖 doc_id，避免与同名 pdf/docx 冲突
                doc_ir.doc_id = _doc_dir_name(file_path)
                parser_stats[name]["success"] += 1
                attempts.append({"parser": name, "status": "success"})
                break
            except Exception as e:
                parser_stats[name]["failed"] += 1
                attempts.append({"parser": name, "status": "failed", "error": str(e)})
                print(f"  ⚠️ {name} 失败: {e}")

        if doc_ir is None:
            failures.append(
                {
                    "file": str(file_path),
                    "error": "all_parsers_failed",
                    "attempts": attempts,
                }
            )
            print("  ❌ 全部解析器失败")
            continue

        try:
            _materialize_page_images(doc_ir, file_path, doc_dir)
            saved = doc_ir.save(doc_dir)
            new_docs.append(doc_ir.to_dict())
            md_parts.append(doc_ir.to_markdown())
            print(f"  ✅ 成功 pages={doc_ir.total_pages} -> {saved['markdown']}")
        except Exception as e:
            failures.append({"file": str(file_path), "error": str(e), "attempts": attempts})
            print(f"  ❌ 保存失败: {e}")

    stats = {
        "success_files": len(new_docs),
        "failed_files": len(failures),
        "skipped_files": skipped,
        "failures": failures,
    }

    print("\n合并 knowledge_base.json …")
    backup = None
    if kb_path.is_file():
        backup = kb_path.with_suffix(".json.bak_before_unsupported")
        if not backup.exists():
            shutil.copy2(kb_path, backup)
            print(f"已备份: {backup}")
    _stream_merge_kb(kb_path if kb_path.is_file() else Path(""), new_docs, kb_path, stats)

    # 追加合并 markdown
    md_path = output_path / "combined_knowledge_base.md"
    if md_parts:
        header = (
            f"\n\n---\n\n# 增量补充（未支持格式）{datetime.now().isoformat()}\n"
            f"新增文档: {len(md_parts)}\n\n"
        )
        with md_path.open("a", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(md_parts))

    report = {
        "created_time": datetime.now().isoformat(),
        "mode": "incremental_unsupported",
        "input_dir": str(input_path),
        "processed": len(todo),
        "success": len(new_docs),
        "failed": len(failures),
        "skipped_existing": skipped,
        "parser_stats": parser_stats,
        "failures": failures[:100],
    }
    report_path = output_path / "unsupported_parse_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 56)
    print(f"成功: {len(new_docs)}  失败: {len(failures)}  跳过: {skipped}")
    print(f"知识库: {kb_path}")
    print(f"报告: {report_path}")
    print("注意: 索引需重建后问答才能用到新文档：")
    print("  python scripts/build_index.py --force-rebuild")
    print("=" * 56)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
