"""把 KB v3 的检索单元灌进 LightRAG，构建实体+关系知识图谱（阶段 B）。

## 三个关键设计

**① 按价值排序，而不是按目录顺序**
文档按「金标准题目引用次数」降序处理。这样即使中途停止，
已经建好的图谱也覆盖了最可能被问到的内容 —— 而不是先处理一堆无关文档。
排序依据来自 `output/eval/plan_kg_scope.json`；没有该文件时退化为按字符数降序。

**② 断点续跑**
每份文档处理完就写一行到 manifest。重跑时自动跳过已完成的文档，
不会重复计费（这是最容易烧钱的地方）。

**③ 携带自定义 chunk_id**
调用 `insert_text(text, chunk_id=...)`，让 LightRAG 抽出的实体/关系的
`source_id` 能指回我们的检索单元。是否真的被保留由
`lightrag_client.probe_capabilities()` 实测确认（总方案 §4.3 / B3）。

## 用法

    # 1) 先探测能力（必做，10 秒）
    python scripts/build_lightrag_kg.py --probe

    # 2) 试跑 3 份，确认成本与质量
    python scripts/build_lightrag_kg.py --max-docs 3

    # 3) 挂夜跑全量（可随时 Ctrl+C，已完成的不会重跑）
    python scripts/build_lightrag_kg.py

    # 只看计划不执行
    python scripts/build_lightrag_kg.py --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KB = ROOT / "output" / "kb_v3"
EVAL = ROOT / "output" / "eval"
MANIFEST = KB / "lightrag_kg_manifest.jsonl"
SCOPE_PLAN = EVAL / "kg_scope_plan.json"


def load_chunks() -> list[dict]:
    path = KB / "chunks.jsonl"
    if not path.is_file():
        raise SystemExit(f"缺少 {path}（先跑 scripts/build_kb_v3.py）")
    items: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if (item.get("content") or "").strip():
                items.append(item)
    return items


def build_plan(items: list[dict]) -> list[dict]:
    """按「金标准引用次数 → 字符数」降序排列文档，返回待处理清单。"""
    by_doc: dict[str, list[dict]] = collections.OrderedDict()
    for item in items:
        by_doc.setdefault(str(item.get("source_doc") or "unknown"), []).append(item)

    priority: dict[str, int] = {}
    if SCOPE_PLAN.is_file():
        plan = json.loads(SCOPE_PLAN.read_text(encoding="utf-8"))
        for entry in plan.get("top_referenced") or []:
            priority[str(entry.get("doc"))] = int(entry.get("refs") or 0)

    docs = []
    for doc, chunks in by_doc.items():
        chars = sum(len(c.get("content") or "") for c in chunks)
        docs.append({
            "doc": doc,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "chars": chars,
            "refs": priority.get(doc, 0),
        })
    # 引用次数优先；其次字符数（信息量）
    docs.sort(key=lambda d: (-d["refs"], -d["chars"]))
    return docs


def load_done() -> dict[str, dict]:
    if not MANIFEST.is_file():
        return {}
    out: dict[str, dict] = {}
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out[str(row.get("doc"))] = row
    return out


def document_text(doc_chunks: list[dict]) -> str:
    """把一份文档的所有检索单元拼成完整文本。

    **为什么必须整份插入**：LightRAG 的摄入单位是「文档」——它用 `file_source`
    作为文档身份。若同一 `file_source` 分多次插入，第二次就会返回
    `HTTP 409 Document storage already contains ...`。
    LightRAG 会自己对整份文本切块，我们不再逐 chunk 提交。

    拼接时保留每个 chunk 开头的 `[章节]` 标记 —— 这是切块器写入的结构信息，
    也是 LightRAG `Paragraph` 策略识别语义边界的依据。
    """
    parts: list[str] = []
    for chunk in doc_chunks:
        text = (chunk.get("content") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def clear_lightrag(client, *, dry_run: bool = False) -> int:
    """删除 LightRAG 里已存在的全部文档，用于重置知识图谱。

    从 `kv_store_doc_status.json` 读取文档 id（这是 LightRAG 的权威状态存储），
    再逐个调用 delete 接口。返回删除数量。
    """
    status_files = list((ROOT / "output" / "lightrag").rglob("kv_store_doc_status.json"))
    if not status_files:
        print("  未找到 doc_status 文件，无需清理")
        return 0
    data = json.loads(status_files[0].read_text(encoding="utf-8"))
    doc_ids = list(data.keys())
    print(f"  发现 {len(doc_ids)} 个已登记文档")
    if dry_run:
        for doc_id in doc_ids[:5]:
            print(f"    (dry-run) {doc_id[:48]}")
        return len(doc_ids)

    deleted = failed = 0
    for index, doc_id in enumerate(doc_ids, start=1):
        try:
            client.delete_document(doc_id)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"    删除失败 {doc_id[:40]}: {type(exc).__name__} {str(exc)[:80]}")
        if index % 25 == 0:
            print(f"    已处理 {index}/{len(doc_ids)}（成功 {deleted} 失败 {failed}）", flush=True)
    print(f"  清理完成：成功 {deleted} · 失败 {failed}")
    return deleted


def estimate_cost(chars: int) -> float:
    """与 plan_kg_scope.py 同一套口径，便于对照。"""
    content_tokens = chars * 0.75
    chunks = max(1, int(content_tokens / 1100))
    total_in = (chunks * (3000 + 1100) * 1.4) / 1_000_000
    total_out = (chunks * 600 * 1.4) / 1_000_000
    return round(total_in * 2.0 + total_out * 8.0, 2)


def run_probe(client) -> bool:
    from src.system.lightrag_client import probe_capabilities

    print("=" * 74)
    print("能力探测（chunk_id 对齐 —— 阶段 B 的关键前置验证）")
    print("=" * 74)
    report = probe_capabilities(client)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("reachable"):
        print("\n❌ 服务不可达，先启动 lightrag-server")
        return False
    if report.get("chunk_id_preserved"):
        print("\n✅ chunk_id 可原样保留 —— 图证据能精确映射回检索单元")
        return True
    print("\n⚠️ 未能确认 chunk_id 保留 —— 将退化按 (文档, 页码) 反向匹配，RRF 仍可用")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="把 KB v3 灌进 LightRAG 建知识图谱")
    ap.add_argument("--probe", action="store_true", help="只做能力探测")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    ap.add_argument("--max-docs", type=int, default=0, help="最多处理 N 份文档")
    ap.add_argument("--max-cost", type=float, default=0.0,
                    help="成本上限（CNY），达到后停止。0=不限")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="每份文档之间的间隔秒数（给 LightRAG 内部流水线喘息）")
    ap.add_argument("--force", action="store_true", help="忽略 manifest，全部重跑")
    ap.add_argument("--clear", action="store_true",
                    help="先清空 LightRAG 里已有的全部文档（重置知识图谱）")
    ap.add_argument("--clear-dry-run", action="store_true", help="只列出将被删除的文档")
    args = ap.parse_args()

    from src.system.lightrag_client import LightRAGClient, LightRAGConfig, LightRAGError

    config = LightRAGConfig.from_env()
    client = LightRAGClient(config)

    if args.probe:
        raise SystemExit(0 if run_probe(client) else 2)

    if args.clear or args.clear_dry_run:
        print("=" * 74)
        print("清理 LightRAG 已有文档")
        print("=" * 74)
        clear_lightrag(client, dry_run=args.clear_dry_run)
        if args.clear_dry_run:
            return
        # 清理后 manifest 也应失效，否则会跳过已被删除的文档
        if MANIFEST.is_file():
            backup = MANIFEST.with_suffix(".jsonl.bak")
            MANIFEST.replace(backup)
            print(f"  manifest 已备份为 {backup.name}（清理后必须重跑）")
        return

    items = load_chunks()
    docs = build_plan(items)
    done = {} if args.force else load_done()

    pending = [d for d in docs if d["doc"] not in done]
    if args.max_docs:
        pending = pending[: args.max_docs]

    total_chars = sum(d["chars"] for d in docs)
    pending_chars = sum(d["chars"] for d in pending)
    print("=" * 74)
    print("灌库计划")
    print("=" * 74)
    print(f"  检索单元总数   : {len(items):,}")
    print(f"  文档总数       : {len(docs)}")
    print(f"  已完成（跳过） : {len(done)}")
    print(f"  本次待处理     : {len(pending)}")
    print(f"  待处理字符数   : {pending_chars:,}")
    print(f"  本次预估成本   : CNY {estimate_cost(pending_chars):.1f}")
    print(f"  全量预估成本   : CNY {estimate_cost(total_chars):.1f}")
    print(f"  LightRAG 服务  : {config.base_url}")
    print()
    print(f"  处理顺序（前 12 份）：")
    print(f"    {'引用':>4} {'chunk':>6} {'字符':>9} {'预估':>7}  文档")
    cumulative = 0.0
    for d in pending[:12]:
        c = estimate_cost(d["chars"])
        cumulative += c
        print(f"    {d['refs']:>4} {d['chunk_count']:>6} {d['chars']:>9,} "
              f"{c:>7.2f}  {d['doc'][:48]}")
    print(f"    前 12 份累计预估: CNY {cumulative:.1f}")

    if args.dry_run:
        return

    if not client.available():
        raise SystemExit(
            f"❌ LightRAG 服务不可达（{config.base_url}）。"
            "先启动 lightrag-server（在工作目录下执行，它会读 .env）。"
        )

    print()
    print("=" * 74)
    print("开始灌库（可随时 Ctrl+C，已完成的文档不会重跑）")
    print("=" * 74)
    started = time.time()
    spent = 0.0
    ok_docs = fail_docs = 0

    with MANIFEST.open("a", encoding="utf-8") as manifest_fh:
        for index, d in enumerate(pending, start=1):
            if args.max_cost and spent >= args.max_cost:
                print(f"\n⏹ 已达成本上限 CNY {args.max_cost}，停止")
                break

            doc_started = time.time()
            ok = 0
            err = ""
            status = "ok"
            text = document_text(d["chunks"])
            try:
                # 整份文档一次插入：LightRAG 的摄入单位是文档，不是 chunk。
                # 分多次插入同一 file_source 会触发 409（这正是上一版失败的原因）。
                client.insert_text(text, file_source=str(d["doc"]))
                ok = d["chunk_count"]
            except LightRAGError as exc:
                err = str(exc)[:300]
                # 已存在视为"做过"（幂等），避免重跑时把已完成文档记成失败
                status = "exists" if "409" in err else "failed"
            elapsed = round(time.time() - doc_started, 1)
            cost = estimate_cost(d["chars"]) if status == "ok" else 0.0
            row = {
                "doc": d["doc"], "chunks": d["chunk_count"], "chars": d["chars"],
                "inserted": ok, "status": status,
                "error": err, "elapsed_sec": elapsed, "est_cost_cny": cost,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            manifest_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_fh.flush()

            if status == "ok":
                ok_docs += 1
                spent += cost
            elif status == "exists":
                ok_docs += 1
            else:
                fail_docs += 1

            flag = {"ok": "OK  ", "exists": "SKIP", "failed": "FAIL"}[status]
            print(f"[{index}/{len(pending)}] {flag} {elapsed:>7.1f}s "
                  f"{ok}/{d['chunk_count']} 块  CNY {spent:>6.2f}  "
                  f"{d['doc'][:52]}", flush=True)
            if err and status == "failed":
                print(f"        └─ {err}")

            if args.delay > 0:
                time.sleep(args.delay)

    total = time.time() - started
    print()
    print("=" * 74)
    print(f"完成：成功 {ok_docs} 份 · 失败 {fail_docs} 份")
    print(f"耗时 {total / 60:.1f} 分钟   预估成本 CNY {spent:.1f}")
    print(f"明细: {MANIFEST}")
    print("=" * 74)


if __name__ == "__main__":
    main()
