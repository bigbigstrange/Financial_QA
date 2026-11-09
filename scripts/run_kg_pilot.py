"""Prepare a page-scoped graph sample locally. No LightRAG requests are made.

The existing LightRAG server does not report per-inference Ark usage to the
shared pilot ledger. Until that is implemented, this script deliberately has
no insertion command.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from scripts.build_lightrag_kg import ROOT

MAX_CHUNKS = 20
MAX_CHARS = 20_000


def select_chunks(chunks: list[dict], allowlist: dict[str, set[int]]) -> list[dict]:
    """Require exact source and an entirely whitelisted, explicit page span."""
    selected = []
    for row in chunks:
        pages = allowlist.get(row.get("source_doc"))
        start, end = row.get("page_start"), row.get("page_end")
        if not pages or type(start) is not int or type(end) is not int or start > end:
            continue
        if end - start + 1 <= len(pages) and all(page in pages for page in range(start, end + 1)) and (row.get("content") or "").strip():
            selected.append(row)
    return selected


def load_active_chunks(allowlist: dict[str, set[int]], manifest_path: Path | None = None) -> list[dict]:
    """Read only whitelisted rows from the configured active snapshot."""
    if manifest_path is None:
        from app.core.config import get_settings

        manifest_path = Path(get_settings().kb_path)
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_id = manifest.get("snapshot_id")
    if not snapshot_id or not isinstance(manifest.get("versions"), list):
        raise ValueError("活动知识库清单缺少快照身份")
    chunks_path = manifest_path.parent / "chunks.jsonl"
    selected = []
    with chunks_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("source_doc") not in allowlist:
                continue
            if row.get("snapshot_id") != snapshot_id:
                raise ValueError("检索块与活动快照 ID 不一致")
            selected.extend(select_chunks([row], allowlist))
    return selected


def prepare_pilot(chunks: list[dict], allowlist: dict[str, set[int]], root: Path) -> Path:
    """Write a bounded, traceable mini-batch under an isolated pilot directory."""
    root = Path(root).resolve()
    if root.name != "lightrag_pilot" or any(parent.name == "lightrag" for parent in root.parents):
        raise ValueError("图谱试跑只能写入 lightrag_pilot 目录")
    selected = select_chunks(chunks, allowlist)
    if not selected or len(selected) > MAX_CHUNKS:
        raise ValueError("试跑必须选择 1–20 个有页码的检索块")
    if sum(len(row["content"]) for row in selected) > MAX_CHARS:
        raise ValueError("试跑文本超过 20,000 字符上限")
    snapshots = {row.get("snapshot_id") for row in selected if row.get("snapshot_id")}
    if len(snapshots) > 1:
        raise ValueError("试跑检索块来自不同快照")

    root.mkdir(parents=True, exist_ok=True)
    batch = root / f"batch_{uuid.uuid4().hex}"
    batch.mkdir()
    entries = []
    for index, row in enumerate(selected, start=1):
        name = f"chunk_{index:04d}.txt"
        locator = f"{row['source_doc']}#p{row['page_start']}"
        if row["page_end"] != row["page_start"]:
            locator += f"-p{row['page_end']}"
        (batch / name).write_text(f"[来源] {locator}\n[块] {row.get('chunk_id', '')}\n\n{row['content'].strip()}\n", encoding="utf-8")
        entries.append({"source_doc": row["source_doc"], "page_start": row["page_start"],
                        "page_end": row["page_end"], "chunk_id": row.get("chunk_id", ""),
                        "file": name})
    (batch / "manifest.json").write_text(
        json.dumps({"status": "prepared_only", "snapshot_id": next(iter(snapshots), None),
                    "chunks": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return batch


def lightrag_usage_auditable() -> bool:
    """Fail closed: LightRAG's separate process is not yet in the token ledger."""
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线准备小样本图谱输入；不发送至 LightRAG")
    parser.add_argument("--source-doc", help="活动知识库中的精确 source_doc")
    parser.add_argument("--pages", help="逗号分隔的页码，例如 132,133")
    parser.add_argument("--prepare", action="store_true", help="仅生成本地试跑文件")
    args = parser.parse_args(argv)
    if not args.prepare:
        print("默认干跑：未读取文档，未发送 LightRAG 请求。图谱待计量。")
        return 0
    if not args.source_doc or not args.pages:
        parser.error("--prepare 要求 --source-doc 和 --pages")
    try:
        pages = {int(value.strip()) for value in args.pages.split(",")}
    except ValueError:
        parser.error("--pages 必须是逗号分隔的正整数")
    if not pages or min(pages) < 1 or len(pages) > 20:
        parser.error("--pages 必须包含 1–20 个正整数页码")
    allowlist = {args.source_doc: pages}
    batch = prepare_pilot(load_active_chunks(allowlist), allowlist,
                          ROOT / "output" / "lightrag_pilot")
    print(f"离线样本已准备：{batch}；未发送 LightRAG 请求，图谱待计量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
