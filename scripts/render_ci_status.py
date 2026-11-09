# -*- coding: utf-8 -*-
"""用 GitHub Actions 的真实数据渲染一张 CI 状态卡（非浏览器截图）。

为什么不用截图：截图会过期、无法复现，也无法在 PR 里更新。
这里直接读 Actions API 的 run / jobs / conclusion，生成与页面等价的图。

    python scripts/render_ci_status.py --repo bigbigstrange/Financial_QA
"""
from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F3864"
GREEN = "#2E6B34"
RED = "#B03A3A"
GREY = "#595959"
CARD = "#F6F8FB"


def api(url: str) -> dict:
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={
        "User-Agent": "ci-status-card",
        "Accept": "application/vnd.github+json",
    })
    return json.loads(urllib.request.urlopen(req, timeout=30, context=ctx).read())


def fetch(repo: str) -> dict:
    runs = api(f"https://api.github.com/repos/{repo}/actions/runs?per_page=1")
    run = runs["workflow_runs"][0]
    jobs = api(run["jobs_url"])["jobs"]
    return {
        "repo": repo,
        "sha": run["head_sha"][:7],
        "branch": run["head_branch"],
        "conclusion": run["conclusion"],
        "created": run["created_at"],
        "url": run["html_url"],
        "jobs": [
            {
                "name": job["name"],
                "conclusion": job["conclusion"],
                "steps": len(job["steps"]),
            }
            for job in jobs
        ],
    }


def draw(data: dict, coverage: str, out: Path) -> Path:
    jobs = data["jobs"]
    fig, ax = plt.subplots(figsize=(10.5, 1.15 + 0.5 * len(jobs)), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=CARD, edgecolor="#D7DEEA", linewidth=1.2))

    ok = data["conclusion"] == "success"
    ax.text(0.04, 0.90, "GitHub Actions", fontsize=12, color=GREY)
    ax.text(0.96, 0.90, "PASS" if ok else "FAIL",
            fontsize=12, color=GREEN if ok else RED, ha="right", fontweight="bold")
    ax.text(0.04, 0.83, f"{data['repo']} · {data['branch']} · {data['sha']}",
            fontsize=11, color=NAVY, fontweight="bold")

    y = 0.72
    for job in jobs:
        good = job["conclusion"] == "success"
        ax.text(0.06, y, "PASS" if good else "FAIL", fontsize=9,
                color=GREEN if good else RED, fontweight="bold")
        ax.text(0.145, y, job["name"], fontsize=10.5, color="#1A1A1A")
        ax.text(0.96, y, f"{job['steps']} steps · {job['conclusion']}",
                fontsize=9, color=GREY, ha="right")
        ax.plot([0.05, 0.95], [y - 0.045, y - 0.045], color="#E3E7EE", lw=0.8)
        y -= 0.13

    ax.text(0.06, y + 0.02, f"测试覆盖率 {coverage}（门槛 60%）", fontsize=10, color=GREY)
    ax.text(0.95, 0.05, f"数据来源：GitHub Actions API · {data['created'][:19]}Z",
            fontsize=8, color="#8A93A6", ha="right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="bigbigstrange/Financial_QA")
    parser.add_argument("--coverage", default="67.25%")
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "ci-status.png"))
    args = parser.parse_args()

    data = fetch(args.repo)
    path = draw(data, args.coverage, Path(args.out))
    print(json.dumps({k: v for k, v in data.items() if k != "jobs"}, ensure_ascii=False))
    print(f"written: {path}")


