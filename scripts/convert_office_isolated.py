"""Convert full corpus with per-file timeouts and owned COM process cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.convert_legacy_office import prepare_items, _word_to_pdf, _excel_to_pdf, _ppt_to_pdf, _APP_SPECS


def worker(item_path):
    import pythoncom
    import win32com.client
    import win32process
    import psutil
    item = json.loads(item_path.read_text(encoding="utf-8"))
    pythoncom.CoInitialize()
    app = None
    owned_pid = None
    before = set(psutil.pids())
    try:
        app = win32com.client.DispatchEx(_APP_SPECS[item["kind"]]["prog_id"])
        name = {"word": "WINWORD.EXE", "excel": "EXCEL.EXE", "powerpoint": "POWERPNT.EXE"}[item["kind"]]
        fresh = [p for p in psutil.process_iter(["pid", "name", "create_time"])
                 if p.pid not in before and (p.info["name"] or "").upper() == name]
        if len(fresh) == 1:
            proc = fresh[0]
            owned_pid = proc.pid
            item_path.with_suffix(".pid.json").write_text(json.dumps({"pid": proc.pid, "created": proc.create_time()}))
        app.AutomationSecurity = 3
        app.DisplayAlerts = 0
        if item["kind"] == "word":
            app.Visible = False
            app.Options.UpdateLinksAtOpen = False
            app.Options.SaveNormalPrompt = False
        fn = {"word": _word_to_pdf, "excel": _excel_to_pdf, "powerpoint": _ppt_to_pdf}[item["kind"]]
        fn(app, Path(item["src"]), Path(item["dst"]))
    finally:
        if app is not None and owned_pid is not None:
            app.Quit()
        pythoncom.CoUninitialize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=Path)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker)
        return
    import fitz
    import psutil
    target = ROOT / "output/kb_full"
    out = target / "converted"
    state = target / "conversion_jobs"
    state.mkdir(parents=True, exist_ok=True)
    items = prepare_items(target / "full_inventory.json", out, use_all=False, docs_root=ROOT / "data/documents")
    manifest = target / "convert_isolated_manifest.jsonl"
    converted = []
    for number, item in enumerate(items, 1):
        started = time.monotonic()
        dst = Path(item["dst"])
        pages = None
        if dst.exists():
            try:
                with fitz.open(dst) as pdf:
                    pages = len(pdf)
            except Exception:
                pass
        status = "skipped_exists" if pages else "ok"
        error = ""
        if not pages:
            job = state / (dst.stem + ".json")
            job.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(job)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            try:
                _, stderr = proc.communicate(timeout=args.timeout)
                if proc.returncode:
                    raise RuntimeError(stderr[-1000:])
                with fitz.open(dst) as pdf:
                    pages = len(pdf)
            except subprocess.TimeoutExpired:
                status, error = "timeout", f"Exceeded {args.timeout} seconds"
                # Terminate only the exact COM instance created by this worker,
                # with creation-time verification to guard against PID reuse.
                pid_file = job.with_suffix(".pid.json")
                if pid_file.exists():
                    owned = json.loads(pid_file.read_text())
                    try:
                        office = psutil.Process(owned["pid"])
                        if office.create_time() == owned["created"]:
                            office.terminate()
                    except psutil.Error:
                        pass
                proc.kill()
                proc.communicate()
            except Exception as exc:
                status, error = "failed", str(exc)
        row = {**item, "status": status, "error": error, "total_pages": pages,
               "elapsed_sec": round(time.monotonic() - started, 2)}
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if pages and status in {"ok", "skipped_exists"}:
            converted.append({"rel_path": dst.name, "ext": ".pdf", "total_pages": pages,
                "size_kb": round(dst.stat().st_size / 1024, 1), "origin_name": item["origin_name"],
                "origin_rel_path": item["rel_path"], "origin_ext": item["origin_ext"]})
        print(f"[{number}/{len(items)}] {status}: {item['origin_name']} {error[:120]}", flush=True)
        subset = out / "subset_converted.json"
        temp = subset.with_suffix(".tmp")
        temp.write_text(json.dumps({"count": len(converted), "files": converted}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(subset)
    print(f"Converted {len(converted)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
