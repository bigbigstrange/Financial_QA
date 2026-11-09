"""分层抽样：从 data/documents 选出 100 份覆盖全部文档形态的子集。

选法（确定性，可复现）：
  1. 先按「扩展名 + PDF 类型」分桶，每桶分配名额，保证每种形态都有代表；
  2. 桶内按 (topic, org) 轮转抽样，保证主题与机构多样性；
  3. 全部扫描型 PDF 强制入选（数量少且最容易被漏掉）；
  4. 输出配额与实际结果，便于人工复核。

产出：
  output/kb_v3/subset_100.json     机器可读清单
  output/kb_v3/subset_100.txt      纯路径清单（供 MinerU 批处理）
  output/kb_v3/subset_100_report.md  人工复核报告
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform")
INV = ROOT / "output" / "kb_v3" / "file_inventory.json"
OUT = ROOT / "output" / "kb_v3"

# 各桶名额（合计 100）
QUOTA = {
    "pdf_text": 38,
    "pdf_scanned": 99,     # 扫描件全都要（实际只有 4 份）
    "doc": 18,
    "docx": 12,
    "ppt": 8,
    "pptx": 3,
    "xls": 5,
    "xlsx": 3,
    "image": 5,
    "txt": 4,
    "other": 11,           # 兜底，用于吸收扫描件实际数量与配额差
}


def bucket_of(rec: dict) -> str:
    ext = rec["ext"]
    if ext == ".pdf":
        return "pdf_scanned" if rec.get("pdf_kind") == "scanned" else "pdf_text"
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        return "image"
    if ext == ".doc":
        return "doc"
    if ext == ".docx":
        return "docx"
    if ext in {".ppt", ".pps"}:
        return "ppt"
    if ext == ".pptx":
        return "pptx"
    if ext == ".xls":
        return "xls"
    if ext == ".xlsx":
        return "xlsx"
    if ext in {".txt", ".md"}:
        return "txt"
    return "other"


def diversify(records: list[dict], want: int) -> list[dict]:
    """桶内按 (topic, org) 轮转，最大化主题/机构多样性。"""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for rec in records:
        groups[(rec["topic"], rec["org"])].append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: r["rel_path"])
    keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    picked: list[dict] = []
    exhausted: set[tuple[str, str]] = set()
    while len(picked) < want and len(exhausted) < len(keys):
        for key in keys:
            if key in exhausted:
                continue
            if groups[key]:
                picked.append(groups[key].pop(0))
                if len(picked) >= want:
                    break
            else:
                exhausted.add(key)
    return picked


def main() -> None:
    inv = json.loads(INV.read_text(encoding="utf-8"))
    files = inv["files"]
    for rec in files:
        rec["bucket"] = bucket_of(rec)

    by_bucket: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in files:
        by_bucket[rec["bucket"]].append(rec)

    # 1) 扫描件全选
    selected: list[dict] = list(by_bucket.get("pdf_scanned", []))

    # 2) 其余桶按配额抽
    for bucket, quota in QUOTA.items():
        if bucket in {"pdf_scanned", "other"}:
            continue
        pool = by_bucket.get(bucket, [])
        want = min(quota, len(pool))
        selected.extend(diversify(pool, want))

    # 3) 补足到 100（用 other 桶 + 剩余 pdf_text）
    if len(selected) < 100:
        chosen_paths = {r["rel_path"] for r in selected}
        spare = [
            r for r in files
            if r["rel_path"] not in chosen_paths and r["bucket"] == "other"
        ]
        selected.extend(diversify(spare, 100 - len(selected)))
    if len(selected) < 100:
        chosen_paths = {r["rel_path"] for r in selected}
        spare = [
            r for r in files
            if r["rel_path"] not in chosen_paths and r["bucket"] == "pdf_text"
        ]
        selected.extend(diversify(spare, 100 - len(selected)))

    selected.sort(key=lambda r: r["rel_path"])
    selected = selected[:100]

    # ---- 产出 ----
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subset_100.json").write_text(
        json.dumps(
            {
                "count": len(selected),
                "bucket_quota": QUOTA,
                "bucket_actual": dict(collections.Counter(r["bucket"] for r in selected)),
                "files": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "subset_100.txt").write_text(
        "\n".join(str(ROOT / "data" / "documents" / r["rel_path"]) for r in selected) + "\n",
        encoding="utf-8",
    )

    # ---- 报告 ----
    lines = ["# 100 份子集选型报告", ""]
    lines.append(f"- 文档库总数：{len(files)}")
    lines.append(f"- 入选：{len(selected)}")
    lines.append("")
    lines.append("## 全库 vs 子集：扩展名分布")
    lines.append("")
    lines.append("| 扩展名 | 全库 | 子集 | 覆盖率 |")
    lines.append("|---|---:|---:|---:|")
    all_ext = collections.Counter(r["ext"] or "(无扩展名)" for r in files)
    sub_ext = collections.Counter(r["ext"] or "(无扩展名)" for r in selected)
    for ext, n in all_ext.most_common():
        m = sub_ext.get(ext, 0)
        lines.append(f"| `{ext}` | {n} | {m} | {m / n:.0%} |")
    lines.append("")
    lines.append("## 全库 vs 子集：主题分布")
    lines.append("")
    lines.append("| 主题 | 全库 | 子集 |")
    lines.append("|---|---:|---:|")
    all_topic = collections.Counter(r["topic"] for r in files)
    sub_topic = collections.Counter(r["topic"] for r in selected)
    for t, n in all_topic.most_common():
        lines.append(f"| {t} | {n} | {sub_topic.get(t, 0)} |")
    lines.append("")
    lines.append("## PDF 类型")
    lines.append("")
    lines.append("| 类型 | 全库 | 子集 |")
    lines.append("|---|---:|---:|")
    all_kind = collections.Counter(r.get("pdf_kind", "-") for r in files if r["ext"] == ".pdf")
    sub_kind = collections.Counter(r.get("pdf_kind", "-") for r in selected if r["ext"] == ".pdf")
    for k, n in all_kind.most_common():
        lines.append(f"| {k} | {n} | {sub_kind.get(k, 0)} |")
    lines.append("")
    lines.append("## 入选文件清单")
    lines.append("")
    for r in selected:
        extra = ""
        if r["ext"] == ".pdf":
            extra = f" `{r.get('pdf_kind')}` {r.get('chars_per_page')}字/页 p{r.get('total_pages')}"
        lines.append(f"- `{r['rel_path']}`{extra}")
    (OUT / "subset_100_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"入选 {len(selected)} 份")
    print("桶分布:", dict(collections.Counter(r["bucket"] for r in selected)))
    print("主题分布:", dict(collections.Counter(r["topic"] for r in selected)))
    print()
    print("已写入:")
    print("  output/kb_v3/subset_100.json")
    print("  output/kb_v3/subset_100.txt")
    print("  output/kb_v3/subset_100_report.md")


if __name__ == "__main__":
    main()
