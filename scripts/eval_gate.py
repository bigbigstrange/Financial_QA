#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型迭代门禁（CI / 上线前）：

  python scripts/eval_gate.py --profile candidate-v2 --max-samples 20

退出码：
  0 = 门禁通过
  2 = 门禁失败（不可晋级）
  1 = 执行错误

可选 --promote：通过后自动写入 active_profile 并重建引擎配置文件。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_paths import strip_roaming_user_site_from_sys_path

strip_roaming_user_site_from_sys_path()

from app.db.init_db import init_db
from app.db.session import SessionLocal
from app.services.eval_service import eval_service
from app.services.model_registry import model_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="模型迭代评测门禁")
    parser.add_argument("--profile", default=None, help="候选 profile，默认 active")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--promote", action="store_true", help="门禁通过则晋级生产")
    args = parser.parse_args()

    init_db()
    profile = args.profile or model_registry.active_profile_name()
    print(f"[eval_gate] profile={profile} max_samples={args.max_samples} promote={args.promote}")

    db = SessionLocal()
    try:
        run = eval_service.run_sync(
            db,
            profile_name=profile,
            max_samples=args.max_samples,
            promote_if_pass=args.promote,
        )
    except Exception as e:
        print(f"[eval_gate] ERROR: {e}")
        return 1
    finally:
        db.close()

    print(
        f"[eval_gate] status={run.status} success_rate={run.success_rate} "
        f"latency={run.avg_latency_sec} evidence={run.evidence_coverage} "
        f"gate_passed={run.gate_passed} report={run.report_path}"
    )
    if run.gate_passed:
        print("[eval_gate] PASS — 可晋级 / 已晋级")
        return 0
    print("[eval_gate] FAIL — 禁止晋级，请查看 report 与 configs/model_registry.json gates")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
