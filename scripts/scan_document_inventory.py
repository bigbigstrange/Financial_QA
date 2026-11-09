"""只读扫描 data/documents：格式分布 / PDF 类型（文本型 vs 扫描型）/ 主题分层。

产出 output/kb_v3/file_inventory.json + 分层抽样候选清单。
不修改任何原始文件。
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(r"C:\code\个人资料\code\Agentic_RAG\finance-qa-platform")
DOCS = ROOT / "data" / "documents"
OUT = ROOT / "output" / "kb_v3"
OUT.mkdir(parents=True, exist_ok=True)

ORG_KEYS = [
    "工商银行", "工行", "建设银行", "建行", "农业银行", "农行", "中国银行",
    "交通银行", "交行", "邮政储蓄", "邮储", "浦发", "浦东发展", "招商",
    "中信", "光大", "民生", "兴业", "平安", "太保", "人寿", "人民保险",
    "华夏", "广发", "北京银行", "上海银行", "宁波银行", "江苏银行", "浙商",
    "恒丰", "渤海", "汇丰", "渣打", "花旗",
]

TOPIC_KEYS = [
    ("annual_report", ["年度报告", "年报", "半年度报告", "业绩报告"]),
    ("research", ["研报", "研究报告", "点评", "追踪", "专题", "策略", "行业报告", "公司研究报告"]),
    ("regulation", ["办法", "规定", "通知", "规则", "条例", "细则", "指引", "指导意见", "管理办法", "暂行"]),
    ("manual", ["手册", "指南", "操作说明", "使用说明", "用户手册", "操作手册", "说明书"]),
    ("faq", ["问答", "常见问题", "问题解答", "FAQ", "答疑"]),
    ("spec", ["规范", "标准", "要求", "实施方案"]),
    ("training", ["培训", "总结", "计划", "方案", "报告"]),
]


def classify_topic(name: str) -> str:
    for topic, keys in TOPIC_KEYS:
        for k in keys:
            if k in name:
                return topic
    return "other"


def classify_org(name: str) -> str:
    hits = [k for k in ORG_KEYS if k in name]
    return hits[0] if hits else "unknown"


def probe_pdf(path: Path, max_pages: int = 8) -> dict:
    """探测 PDF 是文本型还是扫描型。"""
    import fitz

    info = {"pdf_kind": "unknown", "pages_probed": 0, "chars_per_page": 0.0,
            "total_pages": 0, "images": 0, "has_text_layer": False}
    try:
        doc = fitz.open(path)
    except Exception as exc:
        info["pdf_kind"] = f"open_error:{type(exc).__name__}"
        return info
    try:
        info["total_pages"] = doc.page_count
        n = min(max_pages, doc.page_count)
        chars = 0
        imgs = 0
        for i in range(n):
            page = doc[i]
            chars += len((page.get_text() or "").strip())
            imgs += len(page.get_images(full=True))
        info["pages_probed"] = n
        info["images"] = imgs
        info["chars_per_page"] = round(chars / max(1, n), 1)
        info["has_text_layer"] = info["chars_per_page"] >= 50
        if info["chars_per_page"] >= 200:
            info["pdf_kind"] = "text"
        elif info["chars_per_page"] >= 50:
            info["pdf_kind"] = "text_sparse"
        elif imgs > 0:
            info["pdf_kind"] = "scanned"
        else:
            info["pdf_kind"] = "empty"
    finally:
        doc.close()
    return info


def main() -> None:
    files = [p for p in DOCS.rglob("*") if p.is_file()]
    inv = []
    by_ext = collections.Counter()
    by_topic = collections.Counter()
    by_org = collections.Counter()
    by_pdf_kind = collections.Counter()

    for i, p in enumerate(sorted(files), 1):
        rel = p.relative_to(DOCS).as_posix()
        ext = p.suffix.lower()
        rec = {
            "rel_path": rel,
            "name": p.name,
            "stem": p.stem,
            "ext": ext,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "topic": classify_topic(p.stem),
            "org": classify_org(p.stem),
        }
        if ext == ".pdf":
            rec.update(probe_pdf(p))
        inv.append(rec)
        by_ext[ext] += 1
        by_topic[rec["topic"]] += 1
        by_org[rec["org"]] += 1
        if ext == ".pdf":
            by_pdf_kind[rec["pdf_kind"]] += 1
        if i % 100 == 0:
            print(f"  ...已扫描 {i}/{len(files)}", flush=True)

    summary = {
        "total_files": len(inv),
        "by_ext": dict(by_ext.most_common()),
        "by_topic": dict(by_topic.most_common()),
        "by_org": dict(by_org.most_common()),
        "pdf_kind": dict(by_pdf_kind.most_common()),
    }
    (OUT / "file_inventory.json").write_text(
        json.dumps({"summary": summary, "files": inv}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"清单已写入: {OUT / 'file_inventory.json'}")


if __name__ == "__main__":
    main()
