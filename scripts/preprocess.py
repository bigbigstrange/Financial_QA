#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
用于将多格式文档转换为统一的Markdown格式
"""

import sys
import argparse
import json
import shutil
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import build_default_parsers


def _parser_name(parser) -> str:
    return parser.__class__.__name__


def _resolve_input_path(input_dir: str) -> Path:
    raw = Path(input_dir)
    if raw.exists():
        return raw

    code_root = Path(__file__).resolve().parent.parent
    candidates = []
    if not raw.is_absolute():
        candidates.append((code_root / raw).resolve())
    candidates.extend(
        [
            (code_root / "data" / "documents").resolve(),
            (code_root / "data" / "documents" / "dataset").resolve(),
            (code_root / "data" / "documents" / "dataset" / "数据集").resolve(),
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    raise ValueError(
        "输入路径不存在: {}。可尝试: {}".format(
            input_dir,
            ", ".join(str(p) for p in candidates),
        )
    )


def _materialize_page_images(doc_ir, source_file: Path, doc_dir: Path):
    """
    为 mRAG 生成真实页图文件，并写回 page/block 的 image_ref。
    """
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_image_paths = {}

    suffix = source_file.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
            from PIL import Image
        except Exception:
            return page_image_paths

        pdf = fitz.open(source_file)
        try:
            for idx, page in enumerate(pdf, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                out_path = pages_dir / f"page_{idx:04d}.jpg"
                img.save(out_path, format="JPEG", quality=92)
                page_image_paths[idx] = str(out_path)
        finally:
            pdf.close()
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        out_path = pages_dir / "page_0001.jpg"
        try:
            from PIL import Image

            with Image.open(source_file) as img:
                img = img.convert("RGB")
                img.save(out_path, format="JPEG", quality=92)
        except Exception:
            shutil.copyfile(source_file, out_path)
        page_image_paths[1] = str(out_path)

    if page_image_paths:
        doc_ir.meta["page_images"] = page_image_paths
        for page in doc_ir.pages:
            image_ref = page_image_paths.get(page.page_no)
            if image_ref:
                page.meta["image_ref"] = image_ref
                for block in page.blocks:
                    block.meta["image_ref"] = image_ref
    return page_image_paths


def preprocess_documents(
    input_dir,
    output_dir,
    force_rebuild=True,
    tesseract_cmd=None,
    temp_dir=None,
):
    """
    预处理文档
    
    Args:
        input_dir: 输入文档目录
        output_dir: 输出目录
        force_rebuild: 是否强制重新处理
    """
    print(f"=========================================")

    if tesseract_cmd:
        t_path = Path(tesseract_cmd)
        if not t_path.exists():
            raise ValueError(f"Tesseract 路径不存在: {tesseract_cmd}")
        os.environ["TESSERACT_CMD"] = str(t_path)
        print(f"Tesseract 路径: {t_path}")

    if temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(temp_path)
        os.environ["TMP"] = str(temp_path)
        print(f"运行时临时目录: {temp_path}")
    print(f"开始预处理文档...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"=========================================")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    parsers = build_default_parsers(legacy_output_dir=output_path)
    
    # 获取所有支持的文件
    input_path = _resolve_input_path(input_dir)
    print(f"实际输入路径: {input_path}")
    supported_extensions = {
        ext
        for parser in parsers
        for ext in getattr(parser, "SUPPORTED_EXTS", set())
    }
    
    if input_path.is_file():
        # 处理单个文件
        file_list = [input_path] if input_path.suffix.lower() in supported_extensions else []
    elif input_path.is_dir():
        # 递归处理目录下的所有支持文件
        file_list = []
        for ext in sorted(supported_extensions):
            file_list.extend(input_path.rglob(f"*{ext}"))
            file_list.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        raise ValueError(f"输入路径不存在: {input_path}")

    print(f"找到 {len(file_list)} 个支持的文件")
    
    if not file_list:
        print("未找到任何支持的文件！")
        return
    
    # 处理所有文件
    success_count = 0
    skipped_count = 0
    documents = []
    combined_markdown_parts = []
    failures = []
    parser_stats = {}
    fallback_chains = []

    for p in parsers:
        parser_stats[_parser_name(p)] = {
            "attempts": 0,
            "success": 0,
            "failed": 0,
        }

    for file_path in file_list:
        try:
            print(f"\n处理: {file_path.name}")
            doc_dir = output_path / file_path.stem
            if doc_dir.exists():
                if force_rebuild:
                    shutil.rmtree(doc_dir, ignore_errors=True)
                else:
                    skipped_count += 1
                    print(f"⏭️ 跳过（已存在，使用 --force-rebuild 可重建）: {file_path.name}")
                    continue

            candidate_parsers = [p for p in parsers if p.supports(file_path)]
            if not candidate_parsers:
                print(f"❌ 无可用解析器: {file_path.name}")
                failures.append({"file": str(file_path), "error": "no_parser"})
                continue

            doc_ir = None
            parse_attempts = []
            for parser in candidate_parsers:
                parser_name = _parser_name(parser)
                parser_stats[parser_name]["attempts"] += 1
                try:
                    doc_ir = parser.parse(file_path)
                    parser_stats[parser_name]["success"] += 1
                    parse_attempts.append({"parser": parser_name, "status": "success"})
                    break
                except Exception as parser_error:
                    parser_stats[parser_name]["failed"] += 1
                    parse_attempts.append(
                        {
                            "parser": parser_name,
                            "status": "failed",
                            "error": str(parser_error),
                        }
                    )

            if doc_ir is None:
                reason = "unknown"
                for attempt in parse_attempts:
                    if attempt.get("status") == "failed" and attempt.get("error"):
                        reason = attempt["error"]
                        break
                failures.append(
                    {
                        "file": str(file_path),
                        "error": "all_parsers_failed",
                        "reason": reason,
                        "attempts": parse_attempts,
                    }
                )
                fallback_chains.append({"file": str(file_path), "attempts": parse_attempts})
                print(f"❌ 处理失败（所有解析器失败）: {file_path.name}")
                print(f"   - 失败原因: {reason}")
                continue

            fallback_chains.append({"file": str(file_path), "attempts": parse_attempts})
            _materialize_page_images(doc_ir, file_path, doc_dir)
            saved = doc_ir.save(doc_dir)

            documents.append(doc_ir.to_dict())
            combined_markdown_parts.append(doc_ir.to_markdown())
            if doc_ir.total_pages > 0:
                success_count += 1
                print(f"✅ 成功: {file_path.name}")
                print(f"   - markdown: {saved['markdown']}")
                print(f"   - docir: {saved['json']}")
            else:
                print(f"❌ 失败: {file_path.name}")
                failures.append({"file": str(file_path), "error": "empty_doc"})
        except Exception as e:
            print(f"❌ 处理失败 {file_path.name}: {e}")
            failures.append({"file": str(file_path), "error": str(e)})
            continue

    knowledge_base = {
        "version": "2.0",
        "created_time": datetime.now().isoformat(),
        "documents": documents,
        "stats": {
            "total_files": len(file_list),
            "success_files": success_count,
            "failed_files": len(failures),
            "skipped_files": skipped_count,
        },
        "failures": failures,
    }

    json_path = output_path / "knowledge_base.json"
    json_path.write_text(json.dumps(knowledge_base, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_path / "combined_knowledge_base.md"
    md_header = [
        "# 多格式文档知识库 (DocIR v2)",
        "",
        f"**创建时间**: {knowledge_base['created_time']}",
        f"**文档数量**: {success_count}",
        "",
        "---",
        "",
    ]
    md_path.write_text("\n".join(md_header + combined_markdown_parts), encoding="utf-8")

    processed_total = success_count + len(failures)
    compatibility_rate = (success_count / processed_total) if processed_total > 0 else 0.0
    parser_report = {
        "created_time": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "summary": {
            "total_files": len(file_list),
            "processed_files": processed_total,
            "success_files": success_count,
            "failed_files": len(failures),
            "skipped_files": skipped_count,
            "compatibility_rate": round(compatibility_rate, 6),
        },
        "parser_stats": parser_stats,
        "fallback_chains": fallback_chains,
    }
    parser_report_path = output_path / "parser_report.json"
    parser_report_path.write_text(
        json.dumps(parser_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failed_json_path = output_path / "failed_files_report.json"
    failed_json_path.write_text(
        json.dumps(
            {
                "created_time": datetime.now().isoformat(),
                "failed_count": len(failures),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failed_md_path = output_path / "failed_files_report.md"
    failed_md_lines = [
        "# 解析失败文件清单",
        "",
        f"- 生成时间: {datetime.now().isoformat()}",
        f"- 失败文件数: {len(failures)}",
        "",
    ]
    if not failures:
        failed_md_lines.append("全部文件解析成功。")
    else:
        for idx, item in enumerate(failures, start=1):
            failed_md_lines.append(f"## {idx}. {item.get('file', 'unknown')}")
            failed_md_lines.append("")
            failed_md_lines.append(f"- error: {item.get('error', 'unknown')}")
            if item.get("reason"):
                failed_md_lines.append(f"- reason: {item.get('reason')}")
            attempts = item.get("attempts", [])
            if attempts:
                failed_md_lines.append("- parser_attempts:")
                for at in attempts:
                    p_name = at.get("parser", "unknown")
                    p_status = at.get("status", "unknown")
                    p_error = at.get("error", "")
                    if p_error:
                        failed_md_lines.append(f"  - {p_name}: {p_status} | {p_error}")
                    else:
                        failed_md_lines.append(f"  - {p_name}: {p_status}")
            failed_md_lines.append("")
    failed_md_path.write_text("\n".join(failed_md_lines), encoding="utf-8")
    
    print(f"\n=========================================")
    print(f"预处理完成！")
    print(f"成功处理: {success_count}/{len(file_list)} 个文件")
    print(f"兼容率: {compatibility_rate:.2%}")
    print(f"知识库JSON: {json_path}")
    print(f"合并Markdown: {md_path}")
    print(f"解析报告: {parser_report_path}")
    print(f"失败清单(JSON): {failed_json_path}")
    print(f"失败清单(MD): {failed_md_path}")
    print(f"=========================================")

def main():
    code_root = Path(__file__).resolve().parent.parent
    default_input = code_root / "data" / "documents" / "dataset" / "数据集"
    default_output = code_root / "output"

    parser = argparse.ArgumentParser(description='文档预处理脚本')
    parser.add_argument('--input', '-i', default=str(default_input),
                        help=f'输入文档目录或文件路径 (默认: {default_input})')
    parser.add_argument('--output', '-o', default=str(default_output),
                        help=f'输出目录 (默认: {default_output})')
    parser.add_argument(
        '--force-rebuild',
        '-f',
        action='store_true',
        default=True,
        help='强制重新处理所有文件（默认开启）'
    )
    parser.add_argument(
        '--no-force-rebuild',
        action='store_false',
        dest='force_rebuild',
        help='关闭强制重建，已存在目录将跳过'
    )
    parser.add_argument(
        '--tesseract-cmd',
        default=r"F:\Tools\tesseract\tesseract.exe",
        help='Tesseract 可执行文件路径（默认使用本机路径）'
    )
    parser.add_argument(
        '--temp-dir',
        default=r"F:\temp",
        help='运行时临时目录（将写入 TEMP/TMP，默认: F:\\temp）'
    )
    args = parser.parse_args()
    
    preprocess_documents(
        input_dir=args.input,
        output_dir=args.output,
        force_rebuild=args.force_rebuild,
        tesseract_cmd=args.tesseract_cmd,
        temp_dir=args.temp_dir,
    )

if __name__ == "__main__":
    main()