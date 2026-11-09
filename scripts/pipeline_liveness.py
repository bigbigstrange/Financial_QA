"""Read-only detection of this repository's actual processing jobs."""
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "run_mineru_batch.py": "MinerU 批量解析",
    "convert_office_isolated.py": "Office 隔离转换",
    "complete_full_parse.py": "全量解析流水线",
    "build_kb_v3.py": "KB 构建",
    "build_gold_set.py": "金标准检索",
    "ingest_lightrag_verified.py": "KG 已验证灌库",
}


def classify(command, cwd):
    if Path(cwd).resolve() != ROOT:
        return None
    if "-c" in command:
        return None
    return next((SCRIPTS[Path(arg).name] for arg in command if Path(arg).name in SCRIPTS), None)


def live_jobs():
    jobs = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"]):
        try:
            label = classify(process.info["cmdline"] or [], process.info["cwd"] or ".")
            if label:
                jobs.append({"pid": process.pid, "started_at": process.info["create_time"], "label": label})
        except (psutil.Error, OSError, ValueError):
            continue
    return jobs
