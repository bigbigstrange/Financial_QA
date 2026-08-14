#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cache_paths import resolve_cache_dir, strip_roaming_user_site_from_sys_path

strip_roaming_user_site_from_sys_path()

import pandas as pd

from src.system import FinancialQASystem


def main():
    parser = argparse.ArgumentParser(description="回归评测脚本（检索/答案/鲁棒/性能）")
    parser.add_argument("--kb-path", default="./output/knowledge_base.json")
    parser.add_argument("--test-set", default="./data/复赛测试集.xlsx")
    parser.add_argument("--output", default="./output/benchmark_report.json")
    parser.add_argument("--parser-report", default="./output/parser_report.json")
    parser.add_argument("--final-output", default="./output/final_report.json")
    parser.add_argument("--image-backend", default="visrag", choices=["colpali", "visrag"])
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    qa = FinancialQASystem(args.kb_path, cache_dir=str(resolve_cache_dir()), image_backend=args.image_backend)
    qa.build_vector_index()

    df = pd.read_excel(args.test_set)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    details = []
    total_latency = 0.0
    success = 0
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        q = str(row["question"])
        t0 = time.time()
        res = qa.answer_question(q, qid)
        dt = time.time() - t0
        total_latency += dt
        ok = len(res.get("evidence", [])) > 0
        success += 1 if ok else 0
        details.append(
            {
                "question_id": qid,
                "latency_sec": round(dt, 4),
                "confidence": res.get("confidence", 0.0),
                "evidence_count": len(res.get("evidence", [])),
                "ok": ok,
            }
        )

    report = {
        "samples": len(details),
        "avg_latency_sec": round(total_latency / max(len(details), 1), 4),
        "success_rate": round(success / max(len(details), 1), 4),
        "details": details,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"benchmark done: {out}")

    parser_report_path = Path(args.parser_report)
    parser_report = {}
    if parser_report_path.exists():
        parser_report = json.loads(parser_report_path.read_text(encoding="utf-8"))

    final_report = {
        "benchmark": report,
        "parser_report": parser_report,
        "summary": {
            "samples": report.get("samples", 0),
            "avg_latency_sec": report.get("avg_latency_sec"),
            "qa_success_rate": report.get("success_rate"),
            "parser_compatibility_rate": parser_report.get("summary", {}).get("compatibility_rate"),
        },
    }
    final_output = Path(args.final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final report done: {final_output}")


if __name__ == "__main__":
    main()
