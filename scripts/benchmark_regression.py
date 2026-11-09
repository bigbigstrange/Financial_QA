#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.capture_phase0_manifest import sha256_file


class _FailedQA:
    def __init__(self, error: Exception):
        self.error = error

    def answer_question(self, question: str, question_id: str) -> dict:
        raise RuntimeError(f"QA initialization failed: {self.error}") from self.error


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 4)


def _slim_evidence(items: list[dict]) -> list[dict]:
    fields = ("block_id", "source_doc", "page", "score", "fusion_score", "text")
    return [{key: item.get(key) for key in fields if key in item} for item in items]


def _process_rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _missing_required_value(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def run_cases(
    qa,
    cases: Iterable[dict],
    clock: Callable[[], float] = time.perf_counter,
    rss_bytes: Callable[[], int] | None = None,
) -> dict:
    rows = list(cases)
    if not rows:
        raise ValueError("test set is empty")
    if any("question_id" not in row or "question" not in row for row in rows):
        raise ValueError("each test case requires question_id and question")
    if any(
        _missing_required_value(row["question_id"]) or _missing_required_value(row["question"])
        for row in rows
    ):
        raise ValueError("each test case requires non-empty question_id and question")

    details = []
    latencies = []
    rss_observations = []
    evidence_responses = 0
    errors = 0
    rss_bytes = rss_bytes or _process_rss_bytes
    for row in rows:
        qid = str(row["question_id"])
        question = str(row["question"])
        started = clock()
        rss_before = rss_bytes()
        try:
            result = qa.answer_question(question, qid)
            latency = round(clock() - started, 4)
            rss_after = rss_bytes()
            evidence = _slim_evidence(result.get("evidence") or [])
            answered = bool((result.get("answer") or "").strip()) and bool(evidence)
            evidence_responses += int(answered)
            details.append(
                {
                    "question_id": qid,
                    "question": question,
                    "status": "completed",
                    "latency_sec": latency,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": rss_after,
                    "rss_delta_bytes": rss_after - rss_before,
                    "answer": result.get("answer") or "",
                    "confidence": result.get("confidence"),
                    "evidence": evidence,
                    "responded_with_evidence": answered,
                    "error_type": None,
                    "error": None,
                }
            )
        except Exception as exc:
            latency = round(clock() - started, 4)
            rss_after = rss_bytes()
            errors += 1
            details.append(
                {
                    "question_id": qid,
                    "question": question,
                    "status": "error",
                    "latency_sec": latency,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": rss_after,
                    "rss_delta_bytes": rss_after - rss_before,
                    "answer": "",
                    "confidence": None,
                    "evidence": [],
                    "responded_with_evidence": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        latencies.append(latency)
        rss_observations.extend((rss_before, rss_after))

    return {
        "schema_version": 1,
        "metrics": {
            "sample_count": len(details),
            "error_count": errors,
            "evidence_response_rate": round(evidence_responses / len(details), 4),
            "avg_latency_sec": round(mean(latencies), 4),
            "p50_latency_sec": _percentile(latencies, 0.50),
            "p95_latency_sec": _percentile(latencies, 0.95),
            "max_observed_rss_bytes": max(rss_observations),
        },
        "cases": details,
    }


def compare_reports(before: dict, after: dict) -> dict:
    before_cases = _cases_by_id(before, label="before")
    after_cases = _cases_by_id(after, label="after")
    if set(before_cases) != set(after_cases):
        missing_after = sorted(set(before_cases) - set(after_cases))
        missing_before = sorted(set(after_cases) - set(before_cases))
        raise ValueError(
            "question_id sets differ: "
            f"missing_after={missing_after}, missing_before={missing_before}"
        )

    details = []
    for raw in before.get("cases") or []:
        qid = str(raw["question_id"])
        old = before_cases[qid]
        new = after_cases[qid]
        old_answer = _normalize_answer(old.get("answer"))
        new_answer = _normalize_answer(new.get("answer"))
        old_evidence = _evidence_ids(old.get("evidence"))
        new_evidence = _evidence_ids(new.get("evidence"))
        old_error = old.get("status") == "error" or bool(old.get("error"))
        new_error = new.get("status") == "error" or bool(new.get("error"))
        old_latency = _optional_number(old.get("latency_sec"))
        new_latency = _optional_number(new.get("latency_sec"))
        details.append(
            {
                "question_id": qid,
                "answer_changed": old_answer != new_answer,
                "evidence_id_changed": old_evidence != new_evidence,
                "before_evidence_ids": old_evidence,
                "after_evidence_ids": new_evidence,
                "before_error": old_error,
                "after_error": new_error,
                "new_error": new_error and not old_error,
                "latency_delta_sec": (
                    round(new_latency - old_latency, 4)
                    if old_latency is not None and new_latency is not None
                    else None
                ),
            }
        )

    return {
        "schema_version": 1,
        "summary": {
            "case_count": len(details),
            "answer_changed_count": sum(item["answer_changed"] for item in details),
            "evidence_id_changed_count": sum(
                item["evidence_id_changed"] for item in details
            ),
            "new_error_count": sum(item["new_error"] for item in details),
        },
        "cases": details,
    }


def _cases_by_id(report: dict, *, label: str) -> dict[str, dict]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label} report requires a cases list")
    mapped: dict[str, dict] = {}
    for item in cases:
        if not isinstance(item, dict) or _missing_required_value(item.get("question_id")):
            raise ValueError(f"{label} report has a case without question_id")
        qid = str(item["question_id"])
        if qid in mapped:
            raise ValueError(f"{label} report has duplicate question_id: {qid}")
        mapped[qid] = item
    return mapped


def _normalize_answer(value) -> str:
    return " ".join(str(value or "").split())


def _evidence_ids(value) -> list[str]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        if not isinstance(item, dict):
            continue
        identity = item.get("block_id") or item.get("evidence_id")
        if identity is not None and str(identity).strip():
            ids.append(str(identity))
    return ids


def _optional_number(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="回归评测脚本（检索/答案/鲁棒/性能）")
    parser.add_argument("--kb-path", default="./output/knowledge_base.json")
    parser.add_argument("--test-set", default="./test/merged.xlsx")
    parser.add_argument("--output", default="./output/baseline/qa_report.json")
    parser.add_argument("--parser-report", default="./output/parser_report.json")
    parser.add_argument("--final-output", default="./output/baseline/final_report.json")
    parser.add_argument("--image-backend", default="visrag", choices=["colpali", "visrag"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--compare-before")
    parser.add_argument("--compare-after")
    parser.add_argument(
        "--comparison-output",
        default="./output/baseline/phase-1-compatibility.json",
    )
    args = parser.parse_args()

    if bool(args.compare_before) != bool(args.compare_after):
        parser.error("--compare-before and --compare-after must be used together")
    if args.compare_before and args.compare_after:
        before = json.loads(Path(args.compare_before).read_text(encoding="utf-8"))
        after = json.loads(Path(args.compare_after).read_text(encoding="utf-8"))
        comparison = compare_reports(before, after)
        comparison_output = Path(args.comparison_output)
        comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"comparison done: {comparison_output}")
        return

    from cache_paths import resolve_cache_dir, strip_roaming_user_site_from_sys_path

    strip_roaming_user_site_from_sys_path()

    import pandas as pd

    from src.system import FinancialQASystem

    try:
        qa = FinancialQASystem(args.kb_path, cache_dir=str(resolve_cache_dir()), image_backend=args.image_backend)
        qa.build_vector_index()
    except Exception as exc:
        qa = _FailedQA(exc)

    df = pd.read_excel(args.test_set)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    required = {"question_id", "question"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"test set missing columns: {missing}")

    report = run_cases(qa, df[["question_id", "question"]].to_dict("records"))
    report["inputs"] = {
        "knowledge_base": {
            "path": str(Path(args.kb_path).resolve()),
            "sha256": sha256_file(Path(args.kb_path)),
        },
        "test_set": {
            "path": str(Path(args.test_set).resolve()),
            "sha256": sha256_file(Path(args.test_set)),
        },
        "image_backend": args.image_backend,
        "max_samples": args.max_samples,
        "resolved_models": {
            "embedding": getattr(qa, "embed_model", None),
            "image_backend": getattr(qa, "image_backend", None),
            "generation": getattr(qa, "_gen_model", None),
        },
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
            "samples": report["metrics"]["sample_count"],
            "avg_latency_sec": report["metrics"]["avg_latency_sec"],
            "evidence_response_rate": report["metrics"]["evidence_response_rate"],
            "parser_compatibility_rate": parser_report.get("summary", {}).get("compatibility_rate"),
        },
    }
    final_output = Path(args.final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"final report done: {final_output}")


if __name__ == "__main__":
    main()
