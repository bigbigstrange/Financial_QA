"""金标准评测集构建：分层选题 → 检索候选证据 → LLM 起草 → 人工审核 → 定稿。

为什么要这套流程：
    阶段 3 的门禁失败，最根本的原因是「gold 证据是从一次 20 块冒烟测试里自动抓的」，
    压根不是正确答案。本脚本把评测集做成可审计的流水线，四个阶段各自落盘、可断点续跑：

        select   分层从题库抽样（题型 × 机构 × 文档类型）      → gold_candidates.json
        retrieve 对每题召回候选证据（BM25 + 稠密，RRF 融合）    → gold_evidence_candidates.json
        draft    LLM 起草参考答案并挑选证据                     → gold_review.csv（人工审核）
        finalize 读取审核结果，产出正式金标准 + 分层评测语料     → gold_set.json / corpus_stratified.jsonl

人工审核是**必须**的一环（`review_status` 列），这是本流程与旧做法最大的区别。

用法：
    python scripts/build_gold_set.py --stage select --count 40
    python scripts/build_gold_set.py --stage retrieve
    python scripts/build_gold_set.py --stage draft --max-questions 10   # 先小批量试跑
    # 人工填写 output/eval/gold_review.csv 的 review_status / 修正答案与证据
    python scripts/build_gold_set.py --stage finalize
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "output" / "eval"
CANDIDATES = EVAL_DIR / "gold_candidates.json"
EVIDENCE = EVAL_DIR / "gold_evidence_candidates.json"
REVIEW_CSV = EVAL_DIR / "gold_review.csv"
GOLD_SET = EVAL_DIR / "gold_set.json"
CORPUS = EVAL_DIR / "corpus_stratified.jsonl"

SENT_END = "。！？；"

# 题型判定（按优先级）
QUESTION_TYPES = [
    ("numeric", r"多少|多大|几[年月日个次]|占比|比例|同比|环比|增长率|增速|规模|余额|"
                r"利润|收入|资产|负债|亿元|万元|百分|%|利率|息差|资本充足"),
    ("operation_issue", r"故障|异常|提示|报错|不能|无法|失败|未.{0,4}(弹出|生效|显示)|如何处理|怎么办"),
    ("procedure", r"流程|步骤|如何|怎么|怎样|操作|办理|申请|开通|签约"),
    ("comparison", r"对比|相比|差异|区别|不同|比较|变化"),
    ("fact", r"是什么|包括|有哪些|内容|定义|含义|规定|要求|条件|标准|期限|范围"),
]
ORG_PATTERNS = [
    "工行", "工商银行", "建行", "建设银行", "农行", "农业银行", "中国银行", "中行",
    "交行", "交通银行", "邮储", "邮政储蓄", "浦发", "浦东发展", "招商", "中信",
    "光大", "民生", "兴业", "平安", "太保", "人寿", "华夏", "广发", "北京银行",
    "上海银行", "宁波银行", "江苏银行", "浙商", "恒丰", "渤海",
]


def classify_question(text: str) -> str:
    for name, pattern in QUESTION_TYPES:
        if re.search(pattern, text or ""):
            return name
    return "other"


def detect_org(text: str) -> str:
    hits = [o for o in ORG_PATTERNS if o in (text or "")]
    return hits[0] if hits else "unknown"


def load_questions(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_excel(path)
    rows = []
    for _, row in df.iterrows():
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        rows.append({
            "question_id": str(row.get("question_id")),
            "question": q,
            "question_type": classify_question(q),
            "org": detect_org(q),
        })
    return rows


# ---------------------------------------------------------------------------
# stage 1: 分层选题
# ---------------------------------------------------------------------------

def stage_select(args) -> None:
    questions = load_questions(Path(args.questions))
    print(f"题库共 {len(questions)} 题")

    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)

    rng = random.Random(args.seed)
    total = args.count
    weights = {
        "numeric": 0.30, "fact": 0.25, "procedure": 0.20,
        "operation_issue": 0.15, "comparison": 0.05, "other": 0.05,
    }
    picked: list[dict] = []
    for qtype, weight in weights.items():
        pool = by_type.get(qtype, [])
        if not pool:
            continue
        want = max(1, int(round(total * weight)))
        # 题型内按机构打散，避免全被同一家银行占满
        rng.shuffle(pool)
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for item in pool:
            buckets[item["org"]].append(item)
        keys = sorted(buckets.keys(), key=lambda k: -len(buckets[k]))
        cursor = {k: 0 for k in keys}
        taken = []
        while len(taken) < min(want, len(pool)):
            progressed = False
            for k in keys:
                if cursor[k] < len(buckets[k]) and len(taken) < want:
                    taken.append(buckets[k][cursor[k]])
                    cursor[k] += 1
                    progressed = True
            if not progressed:
                break
        picked.extend(taken)

    picked = picked[:total]
    rng.shuffle(picked)
    for i, item in enumerate(picked, start=1):
        item["gold_id"] = f"g{i:03d}"

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES.write_text(json.dumps({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.questions),
        "seed": args.seed,
        "count": len(picked),
        "type_distribution": dict(collections.Counter(q["question_type"] for q in picked)),
        "org_distribution": dict(collections.Counter(q["org"] for q in picked)),
        "questions": picked,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已选 {len(picked)} 题")
    print("  题型分布:", dict(collections.Counter(q["question_type"] for q in picked)))
    print("  机构分布:", dict(collections.Counter(q["org"] for q in picked).most_common(8)))
    print(f"  → {CANDIDATES}")


# ---------------------------------------------------------------------------
# stage 2: 检索候选证据
# ---------------------------------------------------------------------------

class _DenseIndex:
    """轻量稠密索引：直接用 sentence-transformers，不依赖完整检索栈装配。"""

    def __init__(self, items: list[dict], model_name: str, batch_size: int = 16):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.items = items
        self.np = np
        device = (os.environ.get("FINANCE_EMBED_DEVICE") or "").strip() or None
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        # 与 src/system/retrieval.py 保持一致：限制序列长度以控制显存
        max_seq = (os.environ.get("FINANCE_EMBED_MAX_SEQ_LEN") or "").strip()
        self.model.max_seq_length = int(max_seq) if max_seq.isdigit() and int(max_seq) > 0 else 512
        texts = [it["content"] for it in items]
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            emb = self.model.encode(batch, batch_size=len(batch),
                                    show_progress_bar=False, convert_to_numpy=True)
            emb = np.asarray(emb, dtype=np.float32)
            faiss.normalize_L2(emb)
            vectors.append(emb)
        self.index = faiss.IndexFlatIP(int(self.model.get_sentence_embedding_dimension()))
        if vectors:
            self.index.add(np.vstack(vectors))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        import faiss

        q = self.model.encode([query], convert_to_numpy=True).astype(self.np.float32)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0])
                if 0 <= i < len(self.items)]


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def stage_retrieve(args) -> None:
    from src.system.retrieval import KnowledgeStore
    from src.system.sparse import SparseRetriever

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    store = KnowledgeStore(str(Path(args.kb)))
    items = store.load()
    snapshots = {item.get("snapshot_id") for item in items}
    if len(snapshots) != 1 or None in snapshots:
        raise ValueError("候选证据必须来自一个明确的知识库快照")
    snapshot_id = next(iter(snapshots))
    print(f"知识库检索单元: {len(items)}")

    sparse = SparseRetriever(items)
    dense = None
    if args.retriever in {"dense", "hybrid"}:
        print(f"加载稠密模型: {args.embed_model} …", flush=True)
        dense = _DenseIndex(items, args.embed_model, batch_size=args.embed_batch)
        print("稠密索引就绪", flush=True)

    out = []
    covered = 0
    for index, item in enumerate(candidates["questions"], start=1):
        question = item["question"]
        rank_lists: list[list[int]] = []
        bm25_hits = sparse.search(question, top_k=args.top_k)
        bm25_ids = [h["chunk_id"] for h in bm25_hits]
        id_to_idx = {it["chunk_id"]: i for i, it in enumerate(items)}
        rank_lists.append([id_to_idx[c] for c in bm25_ids if c in id_to_idx])
        dense_scores: dict[str, float] = {}
        if dense is not None:
            pairs = dense.search(question, top_k=args.top_k)
            rank_lists.append([i for i, _ in pairs])
            dense_scores = {items[i]["chunk_id"]: s for i, s in pairs}

        fused = _rrf(rank_lists)
        ranked_ids = [
            items[i]["chunk_id"]
            for i in sorted(fused.keys(), key=lambda x: -fused[x])[: args.top_k]
        ]
        by_id = {it["chunk_id"]: it for it in items}
        evidence = []
        for cid in ranked_ids:
            it = by_id.get(cid)
            if not it:
                continue
            evidence.append({
                "chunk_id": cid,
                "document_version": it.get("document_version"),
                "block_ids": it.get("block_ids") or [],
                "source_doc": it.get("source_doc"),
                "page": it.get("page_start"),
                "section_path": it.get("section_path"),
                "type": it.get("type"),
                "bm25_rank": bm25_ids.index(cid) + 1 if cid in bm25_ids else None,
                "dense_score": round(dense_scores.get(cid, 0.0), 4),
                "rrf": round(fused.get(id_to_idx.get(cid, -1), 0.0), 6),
                "content": (it.get("content") or "")[:900],
            })
        has_evidence = bool(evidence)
        covered += 1 if has_evidence else 0
        out.append({
            **{k: item[k] for k in ("gold_id", "question_id", "question",
                                    "question_type", "org")},
            "retrieved_count": len(evidence),
            "covered": has_evidence,
            "candidates": evidence,
        })
        if index % 5 == 0 or index == len(candidates["questions"]):
            print(f"  已检索 {index}/{len(candidates['questions'])}", flush=True)

    evidence_path = Path(getattr(args, "evidence_output", None) or EVIDENCE)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kb_path": str(args.kb),
        "kb_snapshot_id": snapshot_id,
        "review_status": "unreviewed_candidates_only",
        "retriever": args.retriever,
        "top_k": args.top_k,
        "covered_questions": covered,
        "questions": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"有候选证据的题目: {covered}/{len(out)}")
    print(f"  → {evidence_path}")


# ---------------------------------------------------------------------------
# stage 3: LLM 起草参考答案 + 挑证据
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = (
    "你是金融知识库的评测标注助手。基于给定的候选证据回答问题，并挑选真正支撑答案的证据。"
    "严格要求：\n"
    "1) 只依据候选证据作答，不得引入外部知识；证据不足时把 sufficient 设为 false，"
    "并把 answer 写成「知识库中无依据」；\n"
    "2) 证据编号必须来自候选列表，最多 3 条，宁少勿滥；\n"
    "3) 只输出 JSON，不要 markdown、不要解释。\n"
    '输出格式：{"answer": "…", "evidence": [1, 3], "sufficient": true, "note": "…"}'
)


def _build_draft_prompt(question: str, candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] 文档={c.get('source_doc')} 页={c.get('page')} "
            f"章节={c.get('section_path') or '-'} 类型={c.get('type')}\n"
            f"{c.get('content', '')[:700]}"
        )
    return (
        f"问题：{question}\n\n"
        f"候选证据（编号从 1 开始）：\n" + "\n\n".join(lines) +
        "\n\n请输出 JSON。"
    )


def stage_draft(args) -> None:
    from src.system.llm_api import DeepSeekAPIClient

    client = DeepSeekAPIClient()
    if not client.enabled:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，无法起草参考答案")

    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    questions = data["questions"]
    if args.max_questions:
        questions = questions[: args.max_questions]

    rows = []
    used = 0
    for index, item in enumerate(questions, start=1):
        candidates = item.get("candidates") or []
        answer, evidence_idx, sufficient, note = "", [], False, ""
        if candidates:
            try:
                raw = client.chat(
                    [
                        {"role": "system", "content": _DRAFT_SYSTEM},
                        {"role": "user",
                         "content": _build_draft_prompt(item["question"], candidates)},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                )
                used += 1
                payload = _extract_json(raw)
                answer = str(payload.get("answer") or "").strip()
                sufficient = bool(payload.get("sufficient"))
                note = str(payload.get("note") or "")[:200]
                for x in payload.get("evidence") or []:
                    try:
                        i = int(x)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= i <= len(candidates):
                        evidence_idx.append(i)
            except Exception as exc:  # noqa: BLE001
                note = f"LLM 失败: {type(exc).__name__}: {exc}"[:200]
        else:
            note = "无候选证据（知识库未覆盖）"

        chosen = [candidates[i - 1] for i in evidence_idx if i - 1 < len(candidates)]
        rows.append({
            "gold_id": item["gold_id"],
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "org": item["org"],
            "question": item["question"],
            "reference_answer_draft": answer,
            "sufficient_draft": sufficient,
            "evidence_chunk_ids": "|".join(c["chunk_id"] for c in chosen),
            "evidence_pages": "|".join(
                f"{c.get('source_doc')}#p{c.get('page')}" for c in chosen
            ),
            "candidate_count": len(candidates),
            "note": note,
            # 人工填写：approved / rejected / needs_fix
            "review_status": "pending_review",
            "reviewer_answer": "",
            "reviewer_evidence": "",
            "reviewer_note": "",
        })
        if index % 5 == 0 or index == len(questions):
            print(f"  已起草 {index}/{len(questions)}", flush=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"LLM 调用 {used} 次")
    print(f"  → {REVIEW_CSV}（请人工填写 review_status / reviewer_answer / reviewer_evidence）")


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"响应中找不到 JSON: {text[:200]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# stage 4: 定稿 + 分层语料
# ---------------------------------------------------------------------------

def stage_finalize(args) -> None:
    if not REVIEW_CSV.is_file():
        raise SystemExit(f"缺少审核文件: {REVIEW_CSV}")
    with REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    approved, rejected, pending = [], [], []
    for row in rows:
        status = (row.get("review_status") or "").strip().lower()
        if status in {"approved", "ok", "yes", "y", "1", "true"}:
            approved.append(row)
        elif status in {"rejected", "no", "n", "0", "false"}:
            rejected.append(row)
        else:
            pending.append(row)

    print(f"审核结果：通过 {len(approved)} · 否决 {len(rejected)} · 待审 {len(pending)}")
    if pending:
        print("  ⚠️ 仍有待审题目，未通过审核的不会进入金标准")

    cases = []
    for row in approved:
        evidence = (row.get("reviewer_evidence") or row.get("evidence_chunk_ids") or "").strip()
        chunk_ids = [c.strip() for c in evidence.split("|") if c.strip()]
        pages_raw = (row.get("evidence_pages") or "").strip()
        cases.append({
            "gold_id": row["gold_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "question_type": row.get("question_type"),
            "org": row.get("org"),
            "reference_answer": (row.get("reviewer_answer") or
                                 row.get("reference_answer_draft") or "").strip(),
            "gold_chunk_ids": chunk_ids,
            "gold_pages": [p for p in pages_raw.split("|") if p],
            "verified_by": "human",
            "note": (row.get("reviewer_note") or row.get("note") or "")[:300],
        })

    gold = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "人工审核通过的金标准；gold_chunk_ids 是实际包含答案的检索单元",
        "count": len(cases),
        "cases": cases,
    }
    GOLD_SET.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {GOLD_SET}")

    if not cases:
        print("没有通过审核的题目，跳过语料生成")
        return

    _build_stratified_corpus(cases, args)


def _build_stratified_corpus(cases: list[dict], args) -> None:
    """生成分层随机采样的评测语料。

    旧做法是「按文件顺序取前 N 块」，结果 2996 块里有 31.7% 来自同一份文档，
    指标完全不可信。这里改为：
      1. 全部 gold chunk 必选；
      2. 其余按 (source_doc) 分层、每层限额，避免单文档垄断；
      3. 固定随机种子，结果可复现。
    """
    from src.system.retrieval import KnowledgeStore

    store = KnowledgeStore(str(Path(args.kb)))
    items = store.load()
    by_id = {it["chunk_id"]: it for it in items}
    print(f"知识库检索单元: {len(items)}")

    gold_ids = {cid for case in cases for cid in case["gold_chunk_ids"]}
    missing = sorted(cid for cid in gold_ids if cid not in by_id)
    if missing:
        print(f"  ⚠️ {len(missing)} 个 gold chunk 不在当前知识库中（可能来自未解析文档）")

    rng = random.Random(args.seed)
    # 按来源文档分层
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_doc[str(it.get("source_doc"))].append(it)
    for bucket in by_doc.values():
        rng.shuffle(bucket)

    budget = max(0, args.background - len(gold_ids & set(by_id)))
    per_doc_cap = max(1, math.ceil(budget / max(1, len(by_doc)))) if by_doc else 0
    picked_ids: list[str] = []
    docs = sorted(by_doc.keys())
    cursor = 0
    while len(picked_ids) < budget and docs:
        progressed = False
        for doc in docs:
            bucket = by_doc[doc]
            taken = sum(1 for cid in picked_ids
                        if by_id.get(cid, {}).get("source_doc") == doc)
            while bucket and taken < per_doc_cap:
                it = bucket.pop()
                if it["chunk_id"] in gold_ids or it["chunk_id"] in picked_ids:
                    continue
                picked_ids.append(it["chunk_id"])
                taken += 1
                progressed = True
                if len(picked_ids) >= budget:
                    break
            if len(picked_ids) >= budget:
                break
        if not progressed:
            break

    corpus_ids = [cid for cid in gold_ids if cid in by_id] + picked_ids
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as fh:
        for cid in corpus_ids:
            it = by_id[cid]
            fh.write(json.dumps({
                "chunk_id": cid,
                "block_id": cid,
                "content": it.get("content"),
                "source_doc": it.get("source_doc"),
                "page": it.get("page_start"),
                "section_path": it.get("section_path"),
                "type": it.get("type"),
                "is_gold": cid in gold_ids,
            }, ensure_ascii=False) + "\n")

    doc_counts = collections.Counter(by_id[cid].get("source_doc") for cid in corpus_ids)
    top_share = (doc_counts.most_common(1)[0][1] / len(corpus_ids)) if corpus_ids else 0
    print(f"语料 {len(corpus_ids)} 块（gold {len(gold_ids & set(by_id))} + 背景 {len(picked_ids)}）")
    print(f"  涉及文档 {len(doc_counts)} 份；单文档最大占比 {top_share:.1%}（旧做法为 31.7%）")
    print(f"  → {CORPUS}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="金标准评测集构建")
    ap.add_argument("--stage", required=True,
                    choices=["select", "retrieve", "draft", "finalize", "all"])
    ap.add_argument("--questions", default=str(ROOT / "test" / "merged.xlsx"))
    ap.add_argument("--kb", default=str(ROOT / "output" / "kb_v3" / "kb_v3.json"))
    ap.add_argument("--count", type=int, default=40, help="选题数量")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--top-k", type=int, default=10, help="每题候选证据数")
    ap.add_argument("--retriever", default="hybrid",
                    choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--evidence-output", default=None, help="候选证据独立输出路径（不覆盖历史审核材料）")
    ap.add_argument("--embed-model", default=os.environ.get("FINANCE_EMBED_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--background", type=int, default=2000, help="分层语料中背景块数量")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in {"select", "all"}:
        stage_select(args)
    if args.stage in {"retrieve", "all"}:
        stage_retrieve(args)
    if args.stage in {"draft", "all"}:
        stage_draft(args)
    if args.stage in {"finalize", "all"}:
        stage_finalize(args)


if __name__ == "__main__":
    main()
