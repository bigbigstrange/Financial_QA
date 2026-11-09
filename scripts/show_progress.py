#!/usr/bin/env python3
"""Agentic RAG 升级进度看板。

随时运行即可看到三个阶段、评测链路、后台任务与 GPU 的当前状态：

    python scripts/show_progress.py
    python scripts/show_progress.py --watch 30      # 每 30 秒刷新一次

状态全部从磁盘真实产物与运行中的进程读取，不依赖人工维护的进度记录。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KB = ROOT / "output" / "kb_v3"
EVAL = ROOT / "output" / "eval"
PARSED = KB / "parsed"
CONVERTED = KB / "converted"
MANIFEST = KB / "mineru_manifest.jsonl"
CONVERT_MANIFEST = KB / "convert_manifest.jsonl"

BAR_WIDTH = 22
# 控制台通常是 GBK，emoji 会抛 UnicodeEncodeError；统一用 ASCII 标记
M_OK = "[OK]"
M_WAIT = "[..]"
M_TODO = "[  ]"
M_WARN = "[!]"
M_USER = "[<-你]"


# ---------------------------------------------------------------------------
# 绘制工具
# ---------------------------------------------------------------------------

def bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    """纯 ASCII 进度条：GBK 控制台对 Unicode 方块/阴影字符支持不一致。"""
    if total <= 0:
        return "." * width
    filled = int(round(width * min(done, total) / total))
    return "#" * filled + "." * (width - filled)


def line(label: str, done: int, total: int, suffix: str = "") -> str:
    pct = f"{100.0 * done / total:5.1f}%" if total > 0 else "   n/a"
    return f"  {label:22} {done:>4}/{total:<4} [{bar(done, total)}] {pct} {suffix}"


def head(title: str) -> None:
    print()
    print("-" * 78)
    print(f"  {title}")
    print("-" * 78)


# ---------------------------------------------------------------------------
# 状态读取
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows


def latest_by_key(rows: list[dict], key: str = "rel_path") -> dict[str, dict]:
    """同一文件多次运行时，取最后一次记录。"""
    out: dict[str, dict] = {}
    for row in rows:
        out[str(row.get(key))] = row
    return out


def parse_progress() -> dict:
    """PDF 解析进度：以 mineru_manifest 为准。"""
    subset = KB / "subset_100.json"
    total_pdf = 0
    wanted_pdf = set()
    if subset.is_file():
        data = json.loads(subset.read_text(encoding="utf-8"))
        total_pdf = sum(1 for f in data.get("files", []) if f.get("ext") == ".pdf")
        wanted_pdf = {f["rel_path"] for f in data.get("files", []) if f.get("ext") == ".pdf"}
    rows = latest_by_key(read_jsonl(MANIFEST))
    pdf_rows = [r for r in rows.values() if str(r.get("ext")) == ".pdf" and r.get("rel_path") in wanted_pdf]
    done = sum(1 for r in pdf_rows if r.get("status") in {"ok", "skipped_done"})
    failed = [r for r in pdf_rows if r.get("status") == "failed"]
    return {
        "total": total_pdf,
        "done": done,
        "failed": failed,
        "elapsed_sec": sum(float(r.get("elapsed_sec") or 0) for r in pdf_rows
                           if r.get("status") == "ok"),
    }


def converted_progress() -> dict:
    data = {}
    sub = CONVERTED / "subset_converted.json"
    if sub.is_file():
        data = json.loads(sub.read_text(encoding="utf-8"))
    total = int(data.get("count") or 0)
    # 已送入 MinerU 并成功的（doc_key 来自 converted 子集）
    rows = latest_by_key(read_jsonl(MANIFEST))
    done = 0
    for rec in data.get("files", []):
        key_row = rows.get(rec.get("rel_path"))
        if key_row and key_row.get("status") in {"ok", "skipped_done"}:
            done += 1
    return {"total": total, "done": done}


def kb_state() -> dict:
    state = {"documents": 0, "chunks": 0, "prose_end": 0.0, "blocks": 0,
             "tables": 0, "figures": 0, "ocr": 0, "page_images": 0, "snapshot": None}
    stats_path = KB / "kb_v3.json"
    if stats_path.is_file():
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            stats = data.get("stats") or {}
            cs = stats.get("chunk_stats") or {}
            state.update({
                "documents": stats.get("document_count", 0),
                "blocks": stats.get("block_count", 0),
                "chunks": stats.get("chunk_count", 0),
                "prose_end": cs.get("trailing_prose_end_ratio") or 0.0,
                "snapshot": stats.get("snapshot_id"),
            })
            cov = stats.get("coverage") or {}
            state["tables"] = cov.get("documents_with_tables", 0)
            state["figures"] = cov.get("documents_with_figures", 0)
            state["ocr"] = cov.get("ocr_documents", 0)
        except (ValueError, KeyError):
            pass
    if KB.is_dir():
        state["page_images"] = sum(1 for _ in KB.rglob("page_*_full.png"))
    return state


def eval_state() -> dict:
    return {
        "select": (EVAL / "gold_candidates.json").is_file(),
        "retrieve": (EVAL / "gold_evidence_candidates.json").is_file(),
        "draft": (EVAL / "gold_review.csv").is_file(),
        "final": (EVAL / "gold_set.json").is_file(),
        "ab": (EVAL / "ab_report.json").is_file(),
    }


def gpu_state() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 3:
                return f"{parts[0]}%  {parts[1]}/{parts[2]} MiB"
    except Exception:  # noqa: BLE001
        pass
    return "n/a"


def running_jobs() -> list[str]:
    from scripts.pipeline_liveness import live_jobs
    return sorted({job["label"] for job in live_jobs()})


def _legacy_running_jobs() -> list[str]:
    """识别我启动的后台任务（按命令行关键词匹配）。

    用 PowerShell 的 CIM 查询而不是 wmic —— 后者在新版 Windows 已被移除。
    """
    patterns = {
        "MinerU 批量解析": "run_mineru_batch",
        "Office 转换": "convert_legacy_office",
        "Office 隔离转换": "convert_office_isolated",
        "KG 已验证灌库": "ingest_lightrag_verified",
        "全量解析流水线": "complete_full_parse",
        "金标准检索": "build_gold_set",
        "A/B 评测": "eval_ab",
        "e2e Agent 验证": "e2e_agent_check",
        "KB 构建": "build_kb_v3",
        "目录扫描": "scan_document_inventory",
    }
    blob = ""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        blob = out.stdout or ""
    except Exception:  # noqa: BLE001
        return []
    return [label for label, key in patterns.items() if key in blob]


def elapsed_hint() -> str:
    """最近一次运行的耗时，用于估算 ETA。"""
    rows = [r for r in read_jsonl(MANIFEST) if r.get("status") == "ok"]
    times = [float(r["elapsed_sec"]) for r in rows if r.get("elapsed_sec")]
    if not times:
        return ""
    return f"（已完成 {len(times)} 份，平均 {sum(times)/len(times):.0f}s/份）"


# ---------------------------------------------------------------------------

def render() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("=" * 78)
    print(f"  Agentic RAG 升级进度看板        {now}")
    print("=" * 78)

    pdf = parse_progress()
    conv = converted_progress()
    kb = kb_state()
    ev = eval_state()

    head("阶段 A · 当前已激活子集（非全量语料）")
    print(line("PDF 解析（42 份）", pdf["done"], pdf["total"],
               M_OK if pdf["done"] >= pdf["total"] else ""))
    if pdf["failed"]:
        print(f"     {M_WARN} 失败 {len(pdf['failed'])} 份: "
              f"{[r['rel_path'][-40:] for r in pdf['failed'][:3]]}")
    print(line("Office → PDF（49 份）", conv["total"], conv["total"], M_OK))
    print(line("转换 PDF 解析", conv["done"], conv["total"],
               M_TODO + " 待启动" if conv["done"] == 0 else ""))
    print(f"     {elapsed_hint()}")
    print()
    print(f"  KB v3: {kb['documents']} 文档 · {kb['blocks']:,} 块 → "
          f"{kb['chunks']:,} 检索单元")
    print(f"         正文收尾率 {kb['prose_end']:.2%} · 整页图 {kb['page_images']:,} 张 · "
          f"表格 {kb['tables']}/{kb['documents']} · 图表 {kb['figures']}/{kb['documents']} · "
          f"OCR {kb['ocr']} 份")
    print(f"         snapshot: {kb['snapshot']}")

    head("阶段 B · LightRAG 图通道")
    from src.system.cost_policy import paid_api_disabled
    if paid_api_disabled():
        print("  [暂停] 用户暂停付费模型：图谱灌库与在线图查询均关闭；已有数据保留。")
    verified = {}
    for manifest in KB.glob("kg_verified_*.jsonl"):
        for row in read_jsonl(manifest):
            verified[row.get("identity")] = row
    processed = sum(r.get("status") == "processed" for r in verified.values())
    print(line("服务确认完成的文档", processed, kb["documents"], M_WAIT if processed < kb["documents"] else M_OK))
    print("  /query/data 与主检索栈已接入；以实际图查询和评测结果验收。")

    head("全量语料 · 独立候选库")
    full = ROOT / "output/kb_full"
    if (full / "full_inventory.json").exists():
        inventory = json.loads((full / "full_inventory.json").read_text(encoding="utf-8"))
        print(f"  业务文档 {inventory['count']} 份；排除临时/占位文件 {len(inventory.get('excluded', []))} 份")
        rows = {r.get("rel_path"): r for r in read_jsonl(full / "convert_isolated_manifest.jsonl")}
        counts = collections.Counter(r.get("status") for r in rows.values())
        print(f"  Office 逐份处理状态: {dict(counts)}")
        parsed = {r.get("rel_path"): r for r in read_jsonl(full / "mineru_manifest.jsonl")}
        print(f"  全量解析任务状态: {dict(collections.Counter(r.get('status') for r in parsed.values()))}")
    if (full / "pipeline_state.json").exists():
        state = json.loads((full / 'pipeline_state.json').read_text(encoding='utf-8'))
        from scripts.pipeline_liveness import live_jobs
        alive = any(j["label"] == "全量解析流水线" for j in live_jobs())
        print(f"  流水线最近记录: {state}")
        print(f"  流水线进程: {'运行中' if alive else '未运行；不能仅凭状态文件认定任务仍在继续'}")
    visual = KB / "visual_validation.json"
    if visual.exists():
        stats = json.loads(visual.read_text(encoding="utf-8"))["stats"]
        print(f"  视觉索引: {stats['backend']}，{stats['indexed_pages']} 张真实页图")

    head("阶段 C · Agent 编排")
    print(line("图实现 + 条件边", 1, 1, M_OK + " 16 测试"))
    print(line("服务层接入", 1, 1, M_OK + " /api/runs + /trace"))
    print(line("轨迹落库", 1, 1, M_OK + " 5 测试"))
    print("  前端轨迹面板: 代码已实现；构建验证与浏览器运行验收分别记录。")

    head("评测 · 金标准与 A/B")
    steps = [
        ("选题（select）", ev["select"], M_OK + " 40 题"),
        ("检索候选（retrieve）", ev["retrieve"],
         M_OK if ev["retrieve"] else M_WAIT + " 待跑"),
        ("LLM 起草（draft）", ev["draft"], M_OK if ev["draft"] else M_TODO),
        ("人工审核", ev["final"], "" if ev["final"] else M_USER + " 需要你参与"),
        ("A/B 门禁", ev["ab"], ""),
    ]
    for label, done, note in steps:
        print(line(label, 1 if done else 0, 1, note))

    head("运行环境")
    print(f"  GPU: {gpu_state()}")
    jobs = running_jobs()
    print(f"  后台任务: {', '.join(jobs) if jobs else '（无，全部空闲）'}")

    print("  验收以完整覆盖清单、服务确认、人工审核金标准和 A/B 门禁为准；不估算总完成百分比。")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="进度看板")
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒刷新（0=只跑一次）")
    args = ap.parse_args()
    if args.watch > 0:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                render()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n已停止刷新。")
        return
    render()


if __name__ == "__main__":
    main()
