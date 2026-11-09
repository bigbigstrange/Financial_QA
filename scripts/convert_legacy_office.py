"""遗留 Office 格式批量转 PDF（走 Office COM），供 MinerU 统一解析。

为什么统一转 PDF：
    MinerU 对 PDF/图片支持四档全能力（版面、表格、图表、OCR），
    对 Office 格式只支持 `flash` 低质档。把 .doc/.ppt/.xls/.rtf 等先转 PDF，
    可以让**全部文档走同一条高质量解析链路**，表格/图表/OCR 能力一致，评测才可比。

与仓库现有脚本（src/word转pdf.py 等）的差异：
    1. 递归扫描（原脚本用非递归 glob，会漏掉嵌套目录）；
    2. 输出用 `artifact_key(rel_path)` 命名，避免同名不同目录互相覆盖；
    3. 每个格式复用**同一个 COM 实例**并在每 N 份后重启，避免逐份 Dispatch 的开销
       与内存累积（原脚本逐份 Dispatch 且会杀全部 Office 进程）；
    4. 逐份记录到 manifest，支持断点续跑。

用法：
    python scripts/convert_legacy_office.py --dry-run
    python scripts/convert_legacy_office.py                 # 转换子集里的 Office 文件
    python scripts/convert_legacy_office.py --all           # 转换文档库全部 Office 文件
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.identity import artifact_key  # noqa: E402

DOCS = ROOT / "data" / "documents"
OUT_DIR = ROOT / "output" / "kb_v3" / "converted"
MANIFEST = ROOT / "output" / "kb_v3" / "convert_manifest.jsonl"

WORD_EXTS = {".doc", ".docx", ".rtf", ".odt"}
PPT_EXTS = {".ppt", ".pptx", ".pps"}
EXCEL_EXTS = {".xls", ".xlsx"}


def _safe(text: str) -> str:
    """控制台是 GBK 时，把无法编码的字符替换掉。

    文件名里可能出现不换行空格（\\xa0）等字符，直接 print 会抛
    UnicodeEncodeError 并中断整批转换 —— 转换本身是好的，不能死在打印上。
    """
    enc = sys.stdout.encoding or "utf-8"
    try:
        return str(text).encode(enc, errors="replace").decode(enc, errors="replace")
    except (LookupError, UnicodeError):
        return str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# COM 转换
# ---------------------------------------------------------------------------

def _word_to_pdf(word, src: Path, dst: Path) -> None:
    doc = word.Documents.Open(
        str(src), ReadOnly=True, AddToRecentFiles=False,
        ConfirmConversions=False, Visible=False,
    )
    try:
        # Keep cached field values; updating external fields can block unattended export.
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            doc.ExportAsFixedFormat(OutputFileName=str(dst), ExportFormat=17)
        except Exception:  # noqa: BLE001
            doc.SaveAs(str(dst), FileFormat=17)
    finally:
        try:
            doc.Close(SaveChanges=False)
        except Exception:  # noqa: BLE001
            pass


def _ppt_to_pdf(powerpoint, src: Path, dst: Path) -> None:
    pres = powerpoint.Presentations.Open(str(src), WithWindow=False)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        pres.SaveAs(str(dst), 32)   # 32 = ppSaveAsPDF
    finally:
        try:
            pres.Close()
        except Exception:  # noqa: BLE001
            pass


def _excel_to_pdf(excel, src: Path, dst: Path) -> None:
    wb = excel.Workbooks.Open(str(src), ReadOnly=True, UpdateLinks=0)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        wb.ExportAsFixedFormat(0, str(dst))   # 0 = xlTypePDF
    finally:
        try:
            wb.Close(SaveChanges=False)
        except Exception:  # noqa: BLE001
            pass


_APP_SPECS = {
    "word": {"prog_id": "Word.Application", "exts": WORD_EXTS},
    "powerpoint": {"prog_id": "PowerPoint.Application", "exts": PPT_EXTS},
    "excel": {"prog_id": "Excel.Application", "exts": EXCEL_EXTS},
}


def convert_all(
    items: list[dict],
    out_dir: Path,
    *,
    restart_every: int = 40,
    progress=None,
) -> list[dict]:
    """按格式分组批量转换。items: [{rel_path, kind, src, dst}]"""
    import pythoncom
    import win32com.client

    results: list[dict] = []
    by_kind: dict[str, list[dict]] = {}
    for item in items:
        by_kind.setdefault(item["kind"], []).append(item)

    for kind, group in by_kind.items():
        spec = _APP_SPECS[kind]
        app = None
        done_since_restart = 0
        pythoncom.CoInitialize()
        try:
            for index, item in enumerate(group, start=1):
                if app is None:
                    app = win32com.client.DispatchEx(spec["prog_id"])
                    try:
                        app.DisplayAlerts = 0
                        app.AutomationSecurity = 3
                    except Exception:  # noqa: BLE001
                        pass
                    if kind == "word":
                        try:
                            app.Visible = False
                            app.AutomationSecurity = 3   # 禁止宏
                            app.Options.SaveNormalPrompt = False
                        except Exception:  # noqa: BLE001
                            pass
                    done_since_restart = 0

                src = Path(item["src"])
                dst = Path(item["dst"])
                started = time.time()
                status, err = "ok", ""
                try:
                    if kind == "word":
                        _word_to_pdf(app, src, dst)
                    elif kind == "powerpoint":
                        _ppt_to_pdf(app, src, dst)
                    else:
                        _excel_to_pdf(app, src, dst)
                except Exception as exc:  # noqa: BLE001
                    status, err = "failed", f"{type(exc).__name__}: {exc}"[:400]
                elapsed = round(time.time() - started, 1)
                results.append({**item, "status": status, "error": err,
                                "elapsed_sec": elapsed})
                if progress:
                    flag = "OK  " if status == "ok" else "FAIL"
                    progress(f"[{index}/{len(group)}] {flag} {elapsed:>6}s "
                             f"{_safe(item['rel_path'])[-70:]}")
                if err and progress:
                    progress(f"  {err}")
                done_since_restart += 1

                if done_since_restart >= restart_every and index < len(group):
                    try:
                        app.Quit()
                    except Exception:  # noqa: BLE001
                        pass
                    app = None
                    time.sleep(1)
        finally:
            if app is not None:
                try:
                    app.Quit()
                except Exception:  # noqa: BLE001
                    pass
            pythoncom.CoUninitialize()
    return results


# ---------------------------------------------------------------------------
# 清单准备
# ---------------------------------------------------------------------------

def prepare_items(subset_path: Path, out_dir: Path, *, use_all: bool,
                  docs_root: Path) -> list[dict]:
    if use_all:
        files = [p for p in docs_root.rglob("*") if p.is_file()]
        records = [
            {"rel_path": p.relative_to(docs_root).as_posix(), "ext": p.suffix.lower(),
             "topic": None, "org": None}
            for p in files
        ]
    else:
        data = json.loads(subset_path.read_text(encoding="utf-8"))
        records = data["files"]

    items: list[dict] = []
    for rec in records:
        ext = str(rec.get("ext") or "").lower()
        kind = next((k for k, spec in _APP_SPECS.items() if ext in spec["exts"]), None)
        if kind is None:
            continue
        rel = rec["rel_path"]
        if Path(rel).name.startswith("~$"):
            continue
        items.append({
            "rel_path": rel,
            "origin_name": Path(rel).name,
            "origin_ext": ext,
            "kind": kind,
            "src": str(docs_root / rel),
            "dst": str(out_dir / f"{artifact_key(rel)}.pdf"),
            "out_rel": f"{artifact_key(rel)}.pdf",
            "topic": rec.get("topic"),
            "org": rec.get("org"),
            "size_kb": rec.get("size_kb"),
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="遗留 Office → PDF（COM）")
    ap.add_argument("--subset", default=str(ROOT / "output" / "kb_v3" / "subset_100.json"))
    ap.add_argument("--docs-root", default=str(DOCS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--all", action="store_true", help="转换文档库全部 Office 文件")
    ap.add_argument("--restart-every", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = prepare_items(Path(args.subset), out_dir,
                          use_all=args.all, docs_root=Path(args.docs_root).resolve())

    # 断点续跑：产物已存在的直接标 skipped
    pending, skipped = [], []
    for item in items:
        if Path(item["dst"]).is_file() and Path(item["dst"]).stat().st_size > 0:
            skipped.append({**item, "status": "skipped_exists", "error": "",
                            "elapsed_sec": 0})
        else:
            pending.append(item)
    pending.sort(key=lambda x: x.get("size_kb") or 0)
    if args.limit:
        pending = pending[: args.limit]

    by_kind: dict[str, int] = {}
    for item in items:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
    print(f"待转换 {len(items)} 份（按格式 {by_kind}）")
    print(f"  已存在 {len(skipped)}，待处理 {len(pending)}")

    if args.dry_run:
        for item in pending[:20]:
            print(_safe(f"  {item['kind']:10} {item['origin_ext']:6} {item['rel_path'][:78]}"))
        if len(pending) > 20:
            print(f"  ... 其余 {len(pending) - 20} 份")
        return

    started = time.time()
    results = skipped + convert_all(pending, out_dir,
                                    restart_every=args.restart_every,
                                    progress=lambda s: print(_safe(s), flush=True))

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for res in results:
        counts[res["status"]] = counts.get(res["status"], 0) + 1
    print()
    print(f"完成：{counts}  总耗时 {(time.time() - started) / 60:.1f} 分钟")

    # 生成给 MinerU 用的「等价子集」：rel_path 指向转换后的 PDF，
    # 同时保留 origin_* 以便知识库记录原始文件名与格式（溯源不能丢）。
    converted = [r for r in results if r["status"] in {"ok", "skipped_exists"}]
    import fitz

    def page_count(path):
        with fitz.open(path) as document:
            return len(document)

    subset_out = {
        "count": len(converted),
        "source": "converted_from_office",
        "files": [
            {
                "rel_path": r["out_rel"],
                "ext": ".pdf",
                "total_pages": page_count(r["dst"]),
                "size_kb": round(Path(r["dst"]).stat().st_size / 1024, 1)
                if Path(r["dst"]).is_file() else 0,
                "topic": r.get("topic"),
                "org": r.get("org"),
                "origin_rel_path": r["rel_path"],
                "origin_name": r["origin_name"],
                "origin_ext": r["origin_ext"],
                "bucket": r["kind"],
            }
            for r in converted
        ],
    }
    (out_dir / "subset_converted.json").write_text(
        json.dumps(subset_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"转换产物目录: {out_dir}")
    print(f"MinerU 等价子集: {out_dir / 'subset_converted.json'}")


if __name__ == "__main__":
    main()
