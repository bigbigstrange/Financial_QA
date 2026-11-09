"""MinerU 冒烟测试：解析真实 PDF，检查是否产出我们要的语义块 / 表格 / 图表。

用法：
    python scripts/smoke_mineru.py            # 自动选最小的文本型 PDF
    python scripts/smoke_mineru.py --scanned  # 选扫描型 PDF（验证 OCR）
    python scripts/smoke_mineru.py --index 2  # 选第 N 小的 PDF
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform")
SUBSET = ROOT / "output" / "kb_v3" / "subset_100.json"
OUT_ROOT = ROOT / "output" / "kb_v3" / "parsed"


def pick(index: int, scanned: bool) -> Path:
    data = json.loads(SUBSET.read_text(encoding="utf-8"))
    kind = "scanned" if scanned else "text"
    pdfs = sorted(
        [f for f in data["files"] if f["ext"] == ".pdf" and f.get("pdf_kind") == kind],
        key=lambda f: f["size_kb"],
    )
    if not pdfs:
        raise SystemExit(f"子集里没有 {kind} 类型的 PDF")
    rec = pdfs[min(index, len(pdfs) - 1)]
    return ROOT / "data" / "documents" / rec["rel_path"], rec


def run_mineru(pdf: Path, out_dir: Path, tier: str, ocr_mode: str,
               formats: list[str] | None = None) -> tuple[int, str, float]:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mineru-kit", "parse", str(pdf), "-o", str(out_dir), "--tier", tier]
    # --format 是单选；zip 包含 markdown.md / middle_json.json / structured_content.json / images/
    cmd += ["--format", "zip"]
    cmd += ["--ocr-mode", ocr_mode]
    print("命令:", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    elapsed = time.time() - started
    tail = (proc.stdout or "")[-1500:]
    if proc.returncode != 0:
        tail += "\n--- STDERR ---\n" + (proc.stderr or "")[-2500:]

    # 解压 zip，让后续分析直接看到 middle_json.json 与 images/
    import zipfile

    for zp in out_dir.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(out_dir / "extracted")
    return proc.returncode, tail, elapsed


def is_chart(parent_type: str) -> bool:
    return "chart" in (parent_type or "").lower()


def summarize(out_dir: Path) -> dict:
    info: dict = {"files": [], "middle_json": None, "images": {}}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            info["files"].append((str(path.relative_to(out_dir)), path.stat().st_size))

    mid = next(out_dir.rglob("middle_json.json"), None)
    if mid is None:
        return info
    data = json.loads(mid.read_text(encoding="utf-8"))
    pages = data.get("pages") or []
    type_counts: collections.Counter = collections.Counter()
    block_count = 0
    text_chars = 0
    for page in pages:
        for block in page.get("blocks") or []:
            block_count += 1
            btype = str(block.get("type") or "?")
            type_counts[btype] += 1
            if btype == "text":
                content = block.get("content")
                if isinstance(content, str):
                    text_chars += len(content)
                elif isinstance(content, list):
                    for seg in content:
                        if isinstance(seg, dict) and isinstance(seg.get("content"), str):
                            text_chars += len(seg["content"])
    info["middle_json"] = {
        "schema": data.get("schema"),
        "schema_version": data.get("schema_version"),
        "extensions": data.get("extensions"),
        "pages": len(pages),
        "blocks": block_count,
        "type_counts": dict(type_counts.most_common(15)),
        "text_chars": text_chars,
    }
    imgs = collections.Counter()
    for p in out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            name = p.stem
            parts = name.split("_")
            parent = parts[2] if len(parts) > 2 else "?"
            imgs[parent] += 1
    info["images"] = dict(imgs)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--scanned", action="store_true")
    ap.add_argument("--tier", default="standard")
    ap.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--formats", default="middle_json,markdown,zip",
                    help="逗号分隔，会展开成多个 --format 参数")
    args = ap.parse_args()

    pdf, rec = pick(args.index, args.scanned)
    print("=" * 76)
    print(f"样本: {rec['rel_path']}")
    print(f"  {rec['size_kb']:.1f} KB  {rec.get('total_pages')} 页  "
          f"{rec.get('chars_per_page')} 字/页  类型={rec.get('pdf_kind')}")
    print("=" * 76)

    tag = ("scanned" if args.scanned else "text") + f"_{args.index}"
    out_dir = OUT_ROOT / f"smoke_{tag}"
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    code, tail, elapsed = run_mineru(pdf, out_dir, args.tier, args.ocr_mode, formats)
    print(f"\n退出码={code}  耗时={elapsed:.1f}s")
    if code != 0:
        print("--- 输出尾部 ---")
        print(tail)

    info = summarize(out_dir)
    print(f"\n产出文件（{len(info['files'])} 个）:")
    for name, size in info["files"][:15]:
        print(f"  {size:>10,} B  {name}")
    print(f"\n图片按 parent_type 统计: {info['images']}")
    if info["middle_json"]:
        mj = info["middle_json"]
        print(f"\nMiddleJson: schema={mj['schema']} v{mj['schema_version']}")
        print(f"  extensions={mj['extensions']}")
        print(f"  pages={mj['pages']}  blocks={mj['blocks']}  text_chars={mj['text_chars']:,}")
        print(f"  块类型分布: {mj['type_counts']}")
    else:
        print("\n[!] 未找到 middle_json.json")

    (OUT_ROOT / f"smoke_{tag}_summary.json").write_text(
        json.dumps(
            {"record": rec, "exit_code": code, "elapsed_sec": round(elapsed, 1),
             "out_dir": str(out_dir), "summary": info},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n小结已写入: {OUT_ROOT / f'smoke_{tag}_summary.json'}")


if __name__ == "__main__":
    main()
