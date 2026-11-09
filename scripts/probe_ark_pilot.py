"""One explicit, tiny Ark call; never prints credentials or prompt content."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.system.llm_api import DeepSeekAPIClient
from src.system.pilot_budget import PilotBudget


ROOT = Path(__file__).resolve().parent.parent


def probe(client: DeepSeekAPIClient) -> str:
    return client.chat(
        [{"role": "user", "content": "请只回复：连接正常"}],
        max_tokens=16,
        node="probe",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="发送一次有预算约束的 Ark 连接探针")
    parser.add_argument("--send", action="store_true", help="确认发送一次付费 API 请求")
    args = parser.parse_args()
    if not args.send:
        parser.error("默认不发送请求；显式加 --send 才会发送一次")

    load_dotenv(ROOT / ".env", override=False)
    if (os.environ.get("FINANCE_LLM_PROVIDER") or "").lower() != "ark":
        raise RuntimeError("未选择 Ark 提供方，拒绝发送探针")
    os.environ["FINANCE_DISABLE_PAID_API"] = "0"  # only this process
    client = DeepSeekAPIClient()
    if client.model != "doubao-seed-2-1-pro-260915":
        raise RuntimeError("模型与已授权的 Pro 版本不符，拒绝发送探针")
    budget = PilotBudget(
        Path(os.environ.get("FINANCE_PILOT_BUDGET_PATH") or ROOT / "output" / "pilot_budget.sqlite3"),
        limit=int(os.environ.get("FINANCE_PILOT_TOKEN_LIMIT") or "1500000"),
        model=client.model,
    )
    before = budget.summary()["spent_tokens"]
    probe(client)
    after = budget.summary()
    print(json.dumps({
        "provider": client.provider,
        "model": client.model,
        "probe_tokens": after["spent_tokens"] - before,
        "pilot_spent_tokens": after["spent_tokens"],
        "pilot_remaining_tokens": after["limit_tokens"] - after["spent_tokens"],
        "stopped": after["stopped"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
