"""MinerU 批量解析运行器（可断点续跑、可并行、带失败记录）。

设计要点：
  - 每份文件一个独立输出目录，目录名用相对路径的稳定摘要，避免中文/超长路径问题；
  - 已成功产出的文件默认跳过（resume），失败的重跑；
  - 逐份记录耗时与退出码到 manifest.jsonl，便于统计吞吐；
  - 并发数可配（受 GPU 显存约束，默认 2）。

用法：
    # 先看会处理哪些文件（不执行）
    python scripts/run_mineru_batch.py --dry-run

    # 只处理 PDF（42 份）
    python scripts/run_mineru_batch.py --ext .pdf --workers 2

    # 处理全部
    python scripts/run_mineru_batch.py --workers 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.identity import artifact_key as doc_key  # noqa: E402

DOCS = ROOT / "data" / "documents"
SUBSET = ROOT / "output" / "kb_v3" / "subset_100.json"
OUT_ROOT = ROOT / "output" / "kb_v3" / "parsed"
MANIFEST = ROOT / "output" / "kb_v3" / "mineru_manifest.jsonl"


def already_done(out_dir: Path) -> bool:
    return (out_dir / "extracted" / "middle_json.json").is_file()


def run_one(rec: dict, out_root: Path, tier: str, ocr_mode: str,
            timeout: int, force: bool, shard_pages: int = 0,
            docs_root: Path = DOCS) -> dict:
    rel = rec["rel_path"]
    src = Path(docs_root) / rel
    key = doc_key(rel)
    out_dir = out_root / key
    page_count = int(rec.get("total_pages") or 0)
    result = {
        "rel_path": rel,
        "doc_key": key,
        "ext": rec["ext"],
        "size_kb": rec["size_kb"],
        "total_pages": page_count,
        "sharded": bool(shard_pages and page_count > shard_pages),
        "tier": tier,
        "status": "pending",
        "elapsed_sec": None,
        "out_dir": str(out_dir),
    }
    if not src.is_file():
        result["status"] = "missing_source"
        return result
    if not force and already_done(out_dir):
        result["status"] = "skipped_done"
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    err_tail = ""
    try:
        if result["sharded"]:
            # 大文档走分片解析（单进程跑完整份会因耗时/内存累积崩溃）
            sys.path.insert(0, str(ROOT))
            from src.pipeline.mineru_adapter import run_mineru_sharded

            code = 0
            run_mineru_sharded(
                src, out_dir,
                total_pages=page_count, shard_size=shard_pages,
                tier=tier, ocr_mode=ocr_mode, timeout=timeout,
            )
        else:
            cmd = [
                "mineru-kit", "parse", str(src),
                "-o", str(out_dir),
                "--tier", tier,
                "--format", "zip",
                "--ocr-mode", ocr_mode,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
            code = proc.returncode
            err_tail = ((proc.stderr or "") + (proc.stdout or ""))[-600:]
            for zp in out_dir.glob("*.zip"):
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(out_dir / "extracted")
    except subprocess.TimeoutExpired:
        code = -1
        err_tail = f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        code = -1
        err_tail = f"{type(exc).__name__}: {exc}"[:600]

    result["elapsed_sec"] = round(time.time() - started, 1)
    result["exit_code"] = code
    if code == 0 and already_done(out_dir):
        result["status"] = "ok"
    else:
        result["status"] = "failed"
        result["error_tail"] = err_tail
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(SUBSET))
    ap.add_argument("--docs-root", default=str(DOCS),
                    help="源文件根目录（处理转换后的 PDF 时指向 output/kb_v3/converted）")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tier", default="standard")
    ap.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--ext", action="append", default=None,
                    help="只处理指定扩展名，可重复，如 --ext .pdf --ext .png")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-pages", type=int, default=30,
                    help="PDF 页数超过该值时按页码分片解析（0=禁用）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    files = data["files"]
    if args.ext:
        wanted = {e.lower() for e in args.ext}
        files = [f for f in files if f["ext"].lower() in wanted]
    # 先小后大，尽快暴露问题
    files.sort(key=lambda f: f["size_kb"])
    if args.limit:
        files = files[: args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    docs_root = Path(args.docs_root)

    print(f"待处理 {len(files)} 份，workers={args.workers}，tier={args.tier}，"
          f"docs_root={docs_root.name}")
    if args.dry_run:
        for rec in files[:20]:
            print(f"  {rec['size_kb']:>9.1f} KB  {rec['ext']:6}  {rec['rel_path'][:80]}")
        if len(files) > 20:
            print(f"  ... 其余 {len(files) - 20} 份")
        return

    results: list[dict] = []
    started = time.time()
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # 逐份追加写入：进程中途崩溃也不会丢失已完成记录
    manifest_fh = manifest.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run_one, rec, out_root, args.tier, args.ocr_mode,
                            args.timeout, args.force, args.shard_pages,
                            Path(args.docs_root)): rec
                for rec in files
            }
            done = 0
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 — 单份失败不能拖垮整批
                    rec = futures[fut]
                    res = {
                        "rel_path": rec["rel_path"],
                        "doc_key": doc_key(rec["rel_path"]),
                        "ext": rec["ext"],
                        "size_kb": rec["size_kb"],
                        "tier": args.tier,
                        "status": "failed",
                        "elapsed_sec": None,
                        "error_tail": f"{type(exc).__name__}: {exc}",
                    }
                results.append(res)
                manifest_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                manifest_fh.flush()
                done += 1
                flag = {"ok": "OK  ", "skipped_done": "SKIP", "failed": "FAIL",
                        "missing_source": "MISS"}.get(res["status"], "?   ")
                print(f"[{done}/{len(files)}] {flag} {res.get('elapsed_sec') or '-':>7}s "
                      f"{res['rel_path'][:70]}", flush=True)
    finally:
        manifest_fh.close()

    total = time.time() - started
    counts: dict[str, int] = {}
    for res in results:
        counts[res["status"]] = counts.get(res["status"], 0) + 1

    ok_times = [r["elapsed_sec"] for r in results if r["status"] == "ok" and r["elapsed_sec"]]
    print()
    print(f"完成：{counts}  总耗时 {total/60:.1f} 分钟")
    if ok_times:
        print(f"成功 {len(ok_times)} 份，平均 {sum(ok_times)/len(ok_times):.1f}s/份，"
              f"最快 {min(ok_times):.1f}s，最慢 {max(ok_times):.1f}s")
    print(f"明细已追加到: {manifest}")
    if any(r["status"] not in {"ok", "skipped_done"} for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
