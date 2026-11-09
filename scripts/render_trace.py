# -*- coding: utf-8 -*-
"""把一次 run 的 trace 渲染成"节点时序表"（带耗时条）。

设计取舍：真实链路里首个 retrieve 可能占 90%+ 耗时，等比例时间轴会把后面
十几个节点挤成一团。这里改用**按事件顺序逐行 + sqrt 缩放的耗时条**：
顺序一目了然，耗时差异仍然可见（标注真实毫秒数）。

    python scripts/render_trace.py --trace trace.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F3864"
BLUE = "#2F5597"
GREEN = "#2E6B34"
RED = "#B03A3A"
GREY = "#595959"


def load_events(payload: dict) -> list[dict]:
    for key in ("trace", "events", "steps"):
        if isinstance(payload.get(key), list):
            return [e for e in payload[key] if isinstance(e, dict)]
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    return []


def color_for(decision: str, ms: float) -> str:
    text = decision.lower()
    if any(k in text for k in ("refus", "give_up", "no_evidence", "error", "fail")):
        return RED
    if any(k in text for k in ("useful", "answer", "pass", "verified")):
        return GREEN
    if ms >= 500:
        return "#8A4FBF"
    return BLUE


def draw(payload: dict, title: str, out: Path) -> Path:
    events = load_events(payload)
    if not events:
        raise SystemExit("trace 为空，无法绘图")

    durations = [float(e.get("elapsed_ms") or 0) for e in events]
    total = sum(durations)
    scale = max(math.sqrt(max(durations)), 1.0)

    rows = len(events)
    height = 0.9 + 0.42 * rows
    fig, ax = plt.subplots(figsize=(12.6, height), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, rows + 0.8)
    ax.axis("off")

    ax.text(0.0, rows + 0.45, title, fontsize=13.5, color=NAVY, fontweight="bold")
    ax.text(1.0, rows + 0.45, f"总耗时 {total / 1000:.1f}s　·　{rows} 个节点",
            fontsize=10, color=GREY, ha="right")

    for index, (event, ms) in enumerate(zip(events, durations)):
        y = rows - index
        node = str(event.get("node") or event.get("name") or "node")
        decision = str(event.get("decision") or "")
        color = color_for(decision, ms)

        ax.text(0.0, y, f"{index + 1:02d}", fontsize=9, color=GREY, va="center")
        ax.text(0.045, y, node, fontsize=10, color=NAVY, fontweight="bold", va="center")
        ax.text(0.30, y, decision[:34], fontsize=9, color=GREY, va="center")

        bar_width = 0.40 * math.sqrt(ms) / scale
        ax.add_patch(FancyBboxPatch(
            (0.68, y - 0.13), max(bar_width, 0.004), 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color, edgecolor=color, alpha=0.88,
        ))
        ax.text(0.985, y, f"{ms:,.0f} ms", fontsize=9, color="#333333",
                ha="right", va="center", fontweight="bold" if ms >= 500 else "normal")

    ax.text(0.0, 0.25, "注：条形长度按 √耗时 缩放，避免单个长节点把其余节点压扁；右侧标注为真实毫秒数。",
            fontsize=8.5, color=GREY)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "trace.png"))
    parser.add_argument("--title", default="一次问答的真实节点时序（trace）")
    args = parser.parse_args()

    payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    print(f"written: {draw(payload, args.title, Path(args.out))}")
