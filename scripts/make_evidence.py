# -*- coding: utf-8 -*-
"""生成 README 用的证据图：架构图、评测图、终端演示 GIF。

运行：python scripts/make_evidence.py
产物：docs/evidence/*.png / *.gif
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F3864"
BLUE = "#2F5597"
LIGHT = "#E8EEF7"
GREY = "#595959"
RED = "#B03A3A"
GREEN = "#2E6B34"


def draw_arch() -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=200)
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=NAVY, size=9.2, bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
            linewidth=1.4, facecolor=fc, edgecolor=ec,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=tc, fontweight="bold" if bold else "normal",
                linespacing=1.6)

    def arrow(x1, y1, x2, y2, color=BLUE, lw=1.5, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=13, linewidth=lw,
                                     color=color, shrinkA=1, shrinkB=1))

    # ---- Row A：离线 ----
    ax.text(2, 95, "① 离线：文档 → 可检索的知识", fontsize=12.5, color=NAVY, fontweight="bold")
    box(2, 76, 21, 12, "多格式文档\nPDF / Word / Excel / 图片")
    box(25, 76, 21, 12, "解析 + OCR\n隔离转换 · 失败可回退")
    box(48, 76, 21, 12, "DocIR\n版本 · 原文块 · 页图", bold=True)
    box(71, 76, 21, 12, "切块 + 快照 + 索引\n质量门禁", bold=True)
    for x in (22, 46, 70):
        arrow(x, 82, x + 3.4, 82)

    # 换行连接：A 的末端 → B 的起点
    ax.plot([81.5, 81.5, 11], [76, 66, 66], color="#8FA3C4", lw=1.2, linestyle=(0, (4, 3)), zorder=0)
    arrow(11, 66, 11, 51, color="#8FA3C4", lw=1.2)

    # ---- Row B：在线 ----
    ax.text(2, 60, "② 在线：四通道各自算分 → 融合 → 编排出证据链",
            fontsize=12.5, color=NAVY, fontweight="bold")
    box(2, 38, 21, 12, "四通道检索\nBGE-M3 · BM25\n结构检索 · 页图", bold=True)
    box(25, 38, 19, 12, "RRF 融合\n+ 机构 / 产品约束重排")
    box(47, 38, 21, 12, "LangGraph 证据链\n路由→评分→补查→生成", bold=True, fc="#DCE6F5")
    box(71, 38, 19, 12, "双校验\n依据校验 + 答题校验", fc="#FBE4E4", ec=RED, tc=RED, bold=True)
    for x, w in ((22, 4), (44, 4), (68, 4)):
        arrow(x, 44, x + w, 44)

    # ---- 输出 ----
    box(96, 44, 24, 10, "答案 + 原文块 / 页图引用", fc="#E4F0E4", ec=GREEN, tc=GREEN, bold=True)
    box(96, 30, 24, 10, "证据不足 → 明确拒答", fc="#FBE4E4", ec=RED, tc=RED, bold=True)
    box(96, 18, 24, 8, "SSE 轨迹 + 内部核查工单", size=9.5)
    arrow(90, 46.5, 96, 49, color=GREEN)
    arrow(90, 41.5, 96, 35, color=RED)
    arrow(57, 38, 106, 26.5, color=GREY, lw=1.0)

    ax.text(2, 28, "③ 要么可溯源，要么明确拒答\n④ 全链路受预算与恢复约束",
            fontsize=11.5, color=NAVY, fontweight="bold", linespacing=1.9)

    chips = [("token 账本 + 预算熔断", 2), ("检查点 + 幂等恢复", 32), ("取消阻断后续节点", 62)]
    for text, x in chips:
        ax.add_patch(FancyBboxPatch((x, 6), 26, 7, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    linewidth=0.9, facecolor="#F4F6F9", edgecolor="#C9D3E4"))
        ax.text(x + 13, 9.5, text, ha="center", va="center", fontsize=9.5, color=GREY)
    ax.add_patch(FancyBboxPatch((96, 6), 24, 7, boxstyle="round,pad=0.5,rounding_size=1.2",
                                linewidth=0.9, facecolor="#F4F6F9", edgecolor="#C9D3E4"))
    ax.text(108, 9.5, "无 Key 演示模式", ha="center", va="center", fontsize=9.5, color=GREY)

    path = OUT / "architecture.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_metrics() -> Path:
    data = json.loads((ROOT / "eval" / "results" / "metrics.json").read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    labels = [f"Recall@{data['top_k']}", "拒答准确率", "题目通过率"]
    values = [
        data["recall_at_k"] * 100,
        data["refusal_accuracy"] * 100,
        data["passed"] / data["total"] * 100,
    ]
    bars = ax1.barh(labels, values, color=[NAVY, GREEN, BLUE], height=0.55)
    for bar, v in zip(bars, values):
        ax1.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                 va="center", fontsize=11, color="#333333", fontweight="bold")
    ax1.set_xlim(0, 115)
    ax1.set_xlabel("百分比（%）", fontsize=10)
    ax1.set_title(f"示例语料离线评测（{data['dataset_size']} 题）",
                  fontsize=12, color=NAVY, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.45)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    ax2.axis("off")
    rows = [
        ("语料切块", str(data["corpus_chunks"])),
        ("事实题 / 拒答题", f"{data['factual_total']} / {data['refusal_total']}"),
        ("证据阈值", str(data["evidence_threshold"])),
        ("平均耗时", f"{data['avg_latency_ms']} ms"),
        ("通过", f"{data['passed']} / {data['total']}"),
    ]
    ax2.set_title("运行参数与开销", fontsize=12, color=NAVY, fontweight="bold", loc="left")
    y = 0.86
    for key, val in rows:
        ax2.text(0.02, y, key, fontsize=10.5, color=GREY, transform=ax2.transAxes)
        ax2.text(0.98, y, val, fontsize=11, color="#1A1A1A", transform=ax2.transAxes,
                 ha="right", fontweight="bold")
        ax2.plot([0.02, 0.98], [y - 0.035, y - 0.035], color="#E3E7EE", lw=0.8,
                 transform=ax2.transAxes, clip_on=False)
        y -= 0.16

    fig.tight_layout()
    path = OUT / "metrics.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def collect_demo_text() -> list[str]:
    from app.demo_engine import answer_demo

    questions = [
        "稳盈货币基金A的起购金额是多少？",
        "定期开放产品多久开放一次？",
        "信用卡年费是多少？",
    ]
    lines = ["$ python -c \"from app.demo_engine import answer_demo; ...\"", ""]
    for q in questions:
        r = answer_demo(q, "demo")
        best = r["evidence"][0] if r["evidence"] else None
        lines.append(f"> {q}")
        if best:
            lines.append(f"  [检索] {best['chunk_id']}  score={best['score']:.2f}  (阈值 0.35)")
        else:
            lines.append("  [检索] 无命中  score=0.00  (阈值 0.35)")
        lines.append(f"  [判定] {'生成' if r['verified'] else '拒答'}  confidence={r['confidence']:.2f}")
        answer = r["answer"].split("\n")[0]
        lines.append(f"  [回答] {answer[:44]}…" if len(answer) > 44 else f"  [回答] {answer}")
        lines.append("")
    lines.append("✓ 全链路离线运行：无 API Key、无向量库、无大模型")
    return lines


def render_terminal_gif() -> Path:
    lines = collect_demo_text()
    width, height = 1000, 560
    bg, fg, dim = (18, 22, 30), (226, 232, 240), (128, 140, 158)
    green, red, muted = (110, 200, 140), (230, 120, 120), (170, 185, 205)

    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else font

    per_frame = 9
    frames = []
    for start in range(0, max(len(lines), 1), per_frame):
        chunk = lines[start:start + per_frame]
        img = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, 34], fill=(28, 33, 44))
        d.ellipse([14, 11, 26, 23], fill=(237, 106, 94))
        d.ellipse([34, 11, 46, 23], fill=(245, 191, 79))
        d.ellipse([54, 11, 66, 23], fill=(98, 197, 84))
        d.text((width - 300, 10), "Financial_QA · demo mode", font=title_font, fill=dim)

        y = 56
        for line in chunk:
            color = fg
            if line.startswith("$"):
                color = green
            elif "拒答" in line:
                color = red
            elif line.startswith("✓"):
                color = green
            elif line.startswith("  ["):
                color = muted
            d.text((26, y), line, font=font, fill=color)
            y += 36
        frames.append(img)

    durations = [1500] * (len(frames) - 1) + [3200]
    path = OUT / "terminal-demo.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    return path


if __name__ == "__main__":
    for p in (draw_arch(), draw_metrics(), render_terminal_gif()):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")

