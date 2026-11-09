"""段落优先 + 句边界 + 句对齐 overlap 的确定性切块器（chunker-v3）。

设计目标（对照旧链路的实测缺陷）：
    旧：`page.get_text("blocks")` 的行级碎片，83.5% 的块 < 50 字，
        84.5% 的块从句子中间被切断，检索单元没有完整语义。
    新：以「段落 / 表格行组 / 图表」为语义单元，按目标长度聚合并只在句末标点处落刀，
        相邻 chunk 之间保留按完整句子对齐的 overlap，保证跨块事实可被召回。

三条硬规则：
    1. 表格、图表、公式是原子单元，永不与正文合并，也不接受 overlap 前缀
       （表格行组自身已通过 `serialize_group` 重复了章节/表头/单位，是自包含的）。
    2. overlap 必须句对齐：重叠前缀永远由完整句子组成，绝不出现半句话。
    3. 确定性：相同 DocIR 与相同 chunker_version 必须产出完全相同的 chunk_id。

参数可用环境变量覆盖，便于做切块消融实验：
    FINANCE_CHUNK_TARGET_CHARS / MAX_CHARS / MIN_CHARS / OVERLAP_CHARS
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .docir import DocBlock
from .identity import stable_digest

CHUNKER_VERSION = "chunker-v3"

# ---- 默认参数 ----------------------------------------------------------------
DEFAULT_TARGET_CHARS = 600      # 目标长度（中文约 400 tokens，BGE-M3 检索甜点区）
DEFAULT_MAX_CHARS = 900         # 单 chunk 上限（含 overlap 后的最终长度）
DEFAULT_MIN_CHARS = 150         # 过短则与相邻同类 chunk 合并
DEFAULT_OVERLAP_CHARS = 96      # 相邻 chunk 的重叠长度下限（≈16%）
_OVERLAP_MAX_MULTIPLIER = 1.5   # overlap 前缀长度上界倍数，保证最终块长有界

# 句末标点：中英文句号/叹号/问号/分号
SENTENCE_END = "。！？；!?;"
# 允许跟在句末标点之后一起算入句子的收尾符
CLOSING = "”’）)】」》〉…"

# 原子单元：独立成 chunk，不参与聚合，不接受 overlap
# 注意：figure 不在此列 —— 图注/图内文字很短，与相邻正文合并反而更好检索；
#       表格行组必须保持原子（它自带 [列]/[行 N] 结构，混入正文会破坏格式）。
ATOMIC_KINDS = {"table", "table_rows", "equation"}

# 表格/图表类块类型（用于判定 chunk 的 block_type）
_TABLE_KINDS = {"table", "table_rows"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = DEFAULT_TARGET_CHARS
    max_chars: int = DEFAULT_MAX_CHARS
    min_chars: int = DEFAULT_MIN_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    overlap_for_atomic: bool = False
    chunker_version: str = CHUNKER_VERSION

    @classmethod
    def from_env(cls) -> "ChunkConfig":
        return cls(
            target_chars=_env_int("FINANCE_CHUNK_TARGET_CHARS", DEFAULT_TARGET_CHARS),
            max_chars=_env_int("FINANCE_CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS),
            min_chars=_env_int("FINANCE_CHUNK_MIN_CHARS", DEFAULT_MIN_CHARS),
            overlap_chars=_env_int("FINANCE_CHUNK_OVERLAP_CHARS", DEFAULT_OVERLAP_CHARS),
        )

    @property
    def effective_max(self) -> int:
        """聚合正文时使用的上限。

        overlap 前缀最长可达 `_OVERLAP_MAX_MULTIPLIER * overlap_chars`
        （见 `_sentence_tail`），这里预留该空间，
        保证「正文 + overlap 前缀」的最终长度不超过 max_chars。
        """
        reserved = int(self.overlap_chars * _OVERLAP_MAX_MULTIPLIER)
        return max(self.min_chars, self.max_chars - reserved)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    block_ids: List[str]
    page_start: Optional[int]
    page_end: Optional[int]
    bbox_union: Optional[List[float]]
    section_path: Optional[str]
    block_type: str
    parent_block_id: Optional[str]
    char_len: int
    token_estimate: int
    overlap_prev_chars: int = 0
    quality_flags: List[str] = field(default_factory=list)
    # 组末尾若是正文单元，这里记录它是否在句末正确收尾。
    # 组末尾是表格/图注/标题等结构单元时为 None（这些本就不以句号结尾，
    # 不应算作「正文被切断」，否则指标会被结构性结尾稀释）。
    trailing_prose_end: Optional[bool] = None

    def to_retrieval_item(self, *, source_doc: str, document_id: str,
                          document_version: str,
                          snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        """转成检索层（KnowledgeStore / FAISS / BM25 / LightRAG）消费的字典。

        字段命名与 `src/system/retrieval.py` 的 runtime block 契约保持一致，
        检索层因此可以只用 `content` / `block_id` 两个键工作。
        """
        return {
            "block_id": self.chunk_id,          # 检索层用 block_id 作为主键
            "chunk_id": self.chunk_id,
            "block_ids": list(self.block_ids),
            "content": self.text,
            "source_doc": source_doc,
            "document_id": document_id,
            "document_version": document_version,
            "snapshot_id": snapshot_id,
            "page": self.page_start,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bbox": self.bbox_union,
            "section_path": self.section_path,
            "type": self.block_type,
            "parent_block_id": self.parent_block_id,
            "char_len": self.char_len,
            "token_estimate": self.token_estimate,
            "overlap_prev_chars": self.overlap_prev_chars,
            "trailing_prose_end": self.trailing_prose_end,
            "quality_flags": list(self.quality_flags),
            "score": 0.0,
        }


# ---------------------------------------------------------------------------
# 语义单元
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    kind: str                      # heading | paragraph | table | figure | equation
    text: str
    blocks: List[DocBlock]
    atomic: bool = False

    @property
    def char_len(self) -> int:
        return len(self.text)


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 约 1 字 1 token，其他按 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"
              or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af")
    other = len(text) - cjk
    return max(1, int(cjk + other / 4))


def _unit_kind(block: DocBlock) -> str:
    bt = (block.block_type or "").lower()
    if bt == "heading":
        return "heading"
    if bt in _TABLE_KINDS:
        return "table"
    if bt in {"figure", "figure_caption"}:
        return "figure"
    if bt in {"equation", "formula"}:
        return "equation"
    return "paragraph"


def _unit_text(block: DocBlock) -> str:
    """从块里取参与检索的确定性文本。

    表格行组的 `text` 已由 `tables.serialize_group()` 序列化为
    `[章节]/[表格]/[单位]/[列]/[行 N]` 形式，此处直接使用。
    """
    text = (block.text or "").strip()
    if text:
        return text
    bt = (block.block_type or "").lower()
    # 兜底：表格块 text 为空时用结构化字段重建（旧知识库常见此问题）
    if bt in _TABLE_KINDS or bt == "table":
        return _rebuild_table_text(block)
    # 图表：`text` 为空时用「图题 + 图内文字」拼出可检索文本
    if bt in {"figure", "figure_caption"}:
        parts: List[str] = []
        if block.caption:
            parts.append(f"[图] {block.caption}")
        meta = block.meta or {}
        fig_text = meta.get("figure_text") or meta.get("chart_text") or ""
        if fig_text:
            parts.append(str(fig_text))
        if block.section_path:
            parts.insert(0, f"[章节] {block.section_path}")
        return "\n".join(parts)
    if block.formula_latex:
        return f"[公式] {block.formula_latex}"
    if block.table_md:
        return block.table_md.strip()
    return ""


def _rebuild_table_text(block: DocBlock) -> str:
    lines: List[str] = []
    if block.section_path:
        lines.append(f"[章节] {block.section_path}")
    if block.caption:
        lines.append(f"[表格] {block.caption}")
    if block.unit:
        lines.append(f"[单位] {block.unit}")
    headers = [str(h) for h in (block.table_headers or [])]
    rows = block.table_rows or []
    if headers:
        lines.append("[列] " + " | ".join(headers))
    elif rows:
        lines.append("[列] " + " | ".join(f"列{i + 1}" for i in range(len(rows[0]))))
    for idx, row in enumerate(rows):
        lines.append(f"[行 {idx + 1}] " + " | ".join(str(c) for c in row))
    if not lines and block.table_md:
        return block.table_md.strip()
    return "\n".join(lines)


def build_units(blocks: Sequence[DocBlock]) -> List[Unit]:
    """DocBlock 序列 → 语义单元序列。

    - 表格父块（`block_type="table"`）在已有行组子块时被跳过，
      避免同一张表的内容重复入库（行组已重复章节/表头/单位，是自包含的）。
    - 连续正文块保持为独立单元，由 `build_groups` 负责聚合。
    """
    block_list = list(blocks)
    parents_with_row_groups = {
        b.parent_id
        for b in block_list
        if (b.block_type or "").lower() == "table_rows" and b.parent_id
    }

    units: List[Unit] = []
    for block in block_list:
        bt = (block.block_type or "").lower()
        if bt == "table" and block.block_id in parents_with_row_groups:
            continue
        kind = _unit_kind(block)
        text = _unit_text(block)
        if not text:
            continue
        units.append(
            Unit(
                kind=kind,
                text=text,
                blocks=[block],
                atomic=kind in ATOMIC_KINDS,
            )
        )
    return units


# ---------------------------------------------------------------------------
# 句子边界工具
# ---------------------------------------------------------------------------

def _ends_with_sentence_punct(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if stripped[-1] in SENTENCE_END:
        return True
    # 允许 “。” 这类「标点 + 收尾符」结尾
    for ch in reversed(stripped):
        if ch in CLOSING or ch.isspace():
            continue
        return ch in SENTENCE_END
    return False


def _split_at_sentence(text: str, limit: int) -> Tuple[str, str]:
    """在 limit 之前最后一个句末标点处切分，返回 (head, tail)。

    找不到合适边界时按 limit 硬切，由调用方打 `hard_split` 标记。
    """
    if len(text) <= limit:
        return text, ""
    best = -1
    for i in range(min(limit, len(text)) - 1, -1, -1):
        if text[i] in SENTENCE_END:
            best = i
            break
    if best < 0:
        return text[:limit], text[limit:]
    return text[: best + 1], text[best + 1 :]


def _sentence_tail(text: str, want: int) -> str:
    """取 text 末尾一段：长度尽量接近 want，且起点落在完整句子边界上。

    优先向后扫描（在 want 之前找最近句末标点），这样重叠量不少于 want；
    若那样得到的片段超过 `_OVERLAP_MAX_MULTIPLIER * want`，改为向前扫描
    （在 want 之后找最近句末标点），宁可少重叠，也不让最终块长失控。
    两条路径都找不到句末标点时退化为按字符切。
    """
    if want <= 0 or len(text) <= want:
        return text
    start = len(text) - want

    back = -1
    for i in range(start, -1, -1):
        if text[i] in SENTENCE_END:
            back = i
            break
    if back >= 0:
        candidate = text[back + 1 :].lstrip()
        if len(candidate) <= want * _OVERLAP_MAX_MULTIPLIER:
            return candidate

    for i in range(start, len(text)):
        if text[i] in SENTENCE_END:
            return text[i + 1 :].lstrip()

    return text[start:]


# ---------------------------------------------------------------------------
# 分组
# ---------------------------------------------------------------------------

def _join(units: Sequence[Unit]) -> str:
    return "\n".join(u.text.strip() for u in units if u.text.strip()).strip()


def _flush_point(buf: Sequence[Unit], config: ChunkConfig) -> int:
    """返回 buf 中应该切分的位置 k：group = buf[:k]，余下 buf[k:] 留给下一组。

    从后往前找「以句末标点结尾」的前缀，让 chunk 尽量在完整句子处收尾。
    找不到满足 min_chars 的句子边界时返回 len(buf)（整组接收，保持原行为）。
    """
    if not buf:
        return 0
    for k in range(len(buf), 0, -1):
        head = buf[:k]
        if len(_join(head)) < config.min_chars:
            break
        if _ends_with_sentence_punct(head[-1].text):
            return k
    return len(buf)


def build_groups(units: Sequence[Unit], config: ChunkConfig) -> List[List[Unit]]:
    """语义单元 → 分组（每组将成为一个 chunk）。

    规则：
        - 原子单元（表格/图表/公式）独立成组；
        - heading 开启新组并向前合并（让 chunk 带上章节上下文）；
        - 正文按 effective_max 聚合，达到 target 即封口，
          **封口点回退到最后一个句末标点**，残余单元顺延到下一组
          （这样既不从句中落刀，也不丢任何内容）。
    """
    groups: List[List[Unit]] = []
    buf: List[Unit] = []

    def flush_at(k: int) -> None:
        nonlocal buf
        if k <= 0:
            return
        groups.append(buf[:k])
        buf = buf[k:]

    def flush_buffer() -> None:
        """按最后一个句子边界切开 buf；残余单元留给下一组。"""
        nonlocal buf
        if not buf:
            return
        k = _flush_point(buf, config)
        if 0 < k < len(buf):
            flush_at(k)
        else:
            groups.append(buf)
            buf = []

    for unit in units:
        if unit.atomic:
            # 原子单元（表格行组）前后都要在句子边界断开，
            # 否则上一段正文会被从句子中间切断。
            flush_buffer()
            if buf:
                groups.append(buf)      # 全是无句末边界的残余，只能原样保留
                buf = []
            groups.append([unit])
            continue

        if unit.kind == "heading":
            # 软边界：只有当前已累积足够正文时才另起一组，
            # 否则把标题并入当前组（MinerU 的 paragraph_title 很密集，
            # 无条件断开会让每个 chunk 只有 1~2 段正文，聚合失效）。
            if len(_join(buf)) >= config.min_chars:
                groups.append(buf)
                buf = []
            buf.append(unit)
            continue

        # 正文：先看会不会超上限
        if buf and len(_join(buf)) + unit.char_len + 1 > config.effective_max:
            flush_at(_flush_point(buf, config))
        buf.append(unit)
        if len(_join(buf)) >= config.target_chars:
            flush_at(_flush_point(buf, config))

    if buf:
        flush_buffer()
        if buf:
            groups.append(buf)
    return groups


def _merge_short_groups(
    groups: List[List[Unit]],
    config: ChunkConfig,
) -> List[List[Unit]]:
    """过短的非原子组与相邻同类组合并，避免产出语义不足的碎片。"""
    merged: List[List[Unit]] = []
    for group in groups:
        is_atomic = len(group) == 1 and group[0].atomic
        if is_atomic:
            merged.append(group)
            continue
        if not merged:
            merged.append(group)
            continue
        prev = merged[-1]
        prev_atomic = len(prev) == 1 and prev[0].atomic
        if prev_atomic:
            merged.append(group)
            continue
        if len(_join(group)) < config.min_chars:
            candidate = prev + group
            if len(_join(candidate)) <= config.max_chars:
                merged[-1] = candidate
                continue
        merged.append(group)
    return merged


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------

def _apply_overlap(
    groups: List[List[Unit]], config: ChunkConfig
) -> List[Tuple[str, int]]:
    """把每个组渲染成文本，并为相邻正文组加上句对齐的 overlap 前缀。

    返回 (最终文本, overlap 前缀字符数)。

    重叠锚点取自「上一组自身的正文」，而不是上一组加过前缀后的最终文本，
    否则前缀会一代代向前漂移、越叠越长。
    原子组（表格/图表/公式）不接收前缀，也不会作为下一组的重叠来源。
    """
    out: List[Tuple[str, int]] = []
    prev_body: Optional[str] = None
    prev_is_atomic = False

    for group in groups:
        body = _join(group)
        is_atomic = len(group) == 1 and group[0].atomic
        prefix = ""
        allow_overlap = config.overlap_for_atomic or not is_atomic
        if allow_overlap and prev_body and not prev_is_atomic:
            tail = _sentence_tail(prev_body, config.overlap_chars)
            # 上一组整体都很短时 tail == prev_body，不整段重复，避免文本翻倍
            if tail and tail != prev_body:
                prefix = tail
        text = f"{prefix}\n{body}".strip() if prefix else body

        # 兜底：只裁剪 overlap 前缀，绝不裁剪正文。
        # 正文本身超长的情况由 build_chunks 的句子切分处理，这里截断会丢内容。
        if prefix and len(text) > config.max_chars:
            room = config.max_chars - len(body) - 1
            prefix = _sentence_tail(prefix, room) if room > 0 else ""
            text = f"{prefix}\n{body}".strip() if prefix else body

        out.append((text, len(prefix)))
        prev_body = body
        prev_is_atomic = is_atomic
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _bbox_union(blocks: Sequence[DocBlock]) -> Optional[List[float]]:
    boxes = [b.bbox for b in blocks if b.bbox and len(b.bbox) == 4]
    if not boxes:
        return None
    return [
        round(min(b[0] for b in boxes), 1),
        round(min(b[1] for b in boxes), 1),
        round(max(b[2] for b in boxes), 1),
        round(max(b[3] for b in boxes), 1),
    ]


def _chunk_block_type(blocks: Sequence[DocBlock]) -> str:
    """判定 chunk 的类型；混合内容的块一律按正文计（便于指标分层统计）。"""
    kinds = [_unit_kind(b) for b in blocks]
    if not kinds:
        return "paragraph"
    if "table" in kinds:
        return "table"
    if all(k == "figure" for k in kinds):
        return "figure"
    if all(k == "heading" for k in kinds):
        return "heading"
    if "equation" in kinds and all(k == "equation" for k in kinds):
        return "equation"
    return "paragraph"


def build_chunks(
    blocks: Iterable[DocBlock],
    *,
    document_version: str,
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """DocBlock 序列 → Chunk 列表（确定性）。

    chunk_id = "ck_" + sha256(document_version | block_ids | chunker_version)[:32]
    """
    config = config or ChunkConfig.from_env()
    block_list = list(blocks)
    units = build_units(block_list)
    if not units:
        return []

    groups = build_groups(units, config)
    groups = _merge_short_groups(groups, config)
    overlapped = _apply_overlap(groups, config)

    chunks: List[Chunk] = []
    for group, (text, group_prefix_chars) in zip(groups, overlapped):
        is_atomic = len(group) == 1 and group[0].atomic

        # 超长正文按句子切分，并在段间补句对齐 overlap
        # （原子单元不切分，交给表格行组自身的粒度）
        # segments 元素：(文本, 该段携带的 overlap 字符数, 是否被硬切)
        segments: List[Tuple[str, int, bool]] = []
        if not is_atomic and len(text) > config.max_chars:
            limit = config.effective_max
            raw: List[Tuple[str, bool]] = []
            rest = text
            while rest:
                head, remainder = _split_at_sentence(rest, limit)
                if not head:
                    head, remainder = rest[:limit], rest[limit:]
                    raw.append((head, True))
                else:
                    raw.append((head, not _ends_with_sentence_punct(head)))
                rest = remainder
            segments.append((raw[0][0], 0, raw[0][1]))
            for (prev, _prev_hard), (seg, hard) in zip(raw, raw[1:]):
                overlap = _sentence_tail(prev, config.overlap_chars)
                if overlap and overlap != prev:
                    segments.append((f"{overlap}\n{seg}".strip(), len(overlap), hard))
                else:
                    segments.append((seg, 0, hard))
        else:
            segments = [(text, group_prefix_chars, False)]

        group_blocks = [b for u in group for b in u.blocks]
        block_ids = [b.block_id for b in group_blocks]
        pages = [b.page for b in group_blocks if b.page is not None]
        section_path = next(
            (b.section_path for b in group_blocks if b.section_path), None
        )
        parent_id = next(
            (b.parent_id for b in group_blocks if b.parent_id), None
        )
        btype = _chunk_block_type(group_blocks)
        # 判定该 chunk 是否参与「正文被切断」统计：
        #   - 组末尾必须是正文单元（不是表格/图注/标题）
        #   - 且该单元本身足够长（>=100 字），排除"买入""基础数据"这类短标签
        #     （它们本就不以句号结尾，算进来只会制造假失败）
        last_block = group[-1].blocks[-1] if (group and group[-1].blocks) else None
        last_unit_text = (last_block.text or "") if last_block else ""
        group_ends_with_prose = (
            last_block is not None
            and _unit_kind(last_block) == "paragraph"
            and len(last_unit_text.strip()) >= 100
        )

        for seg_index, (raw_segment, overlap_chars, hard_cut) in enumerate(segments):
            segment = raw_segment.strip()
            if not segment:
                continue
            seg_flags: List[str] = []
            if hard_cut:
                seg_flags.append("hard_split")
            if not _ends_with_sentence_punct(segment):
                seg_flags.append("not_sentence_end")
            if len(segment) < config.min_chars and not is_atomic:
                seg_flags.append("under_min_chars")
            seg_block_ids = list(block_ids)
            chunk_id = "ck_" + stable_digest(
                document_version,
                ",".join(seg_block_ids),
                config.chunker_version,
                seg_index,
            )[:32]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=segment,
                    block_ids=seg_block_ids,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    bbox_union=_bbox_union(group_blocks),
                    section_path=section_path,
                    block_type=btype,
                    parent_block_id=parent_id,
                    char_len=len(segment),
                    token_estimate=estimate_tokens(segment),
                    overlap_prev_chars=overlap_chars,
                    trailing_prose_end=(
                        _ends_with_sentence_punct(segment)
                        if (group_ends_with_prose and seg_index == len(segments) - 1)
                        else None
                    ),
                    quality_flags=sorted(set(seg_flags)),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# 质量统计（供 src/pipeline/quality.py 聚合）
# ---------------------------------------------------------------------------

def chunk_stats(chunks: Sequence[Chunk]) -> Dict[str, Any]:
    if not chunks:
        return {
            "chunk_count": 0,
            "chunk_len_mean": 0.0,
            "chunk_len_p50": 0,
            "chunk_len_p90": 0,
            "chunk_under_min_count": 0,
            "chunk_hard_split_count": 0,
            "chunk_not_sentence_end_count": 0,
            "chunk_sentence_end_ratio": 0.0,
            "chunk_overlap_pair_count": 0,
            "chunk_type_counts": {},
            "sentence_end_ratio_by_type": {},
            "chunk_len_p50_by_type": {},
            "prose_chunk_count": 0,
            "prose_sentence_end_ratio": 0.0,
            "trailing_prose_checked": 0,
            "trailing_prose_cut_count": 0,
            "trailing_prose_end_ratio": 0.0,
        }

    lens = sorted(c.char_len for c in chunks)
    n = len(lens)
    under = sum(1 for c in chunks if "under_min_chars" in c.quality_flags)
    hard = sum(1 for c in chunks if "hard_split" in c.quality_flags)
    bad_end = sum(1 for c in chunks if "not_sentence_end" in c.quality_flags)
    overlaps = sum(1 for c in chunks if c.overlap_prev_chars > 0)

    type_counts: Dict[str, int] = {}
    type_total: Dict[str, int] = {}
    type_end_ok: Dict[str, int] = {}
    type_lens: Dict[str, List[int]] = {}
    for c in chunks:
        t = c.block_type
        type_counts[t] = type_counts.get(t, 0) + 1
        type_total[t] = type_total.get(t, 0) + 1
        if "not_sentence_end" not in c.quality_flags:
            type_end_ok[t] = type_end_ok.get(t, 0) + 1
        type_lens.setdefault(t, []).append(c.char_len)

    # 散文块（含句末标点）与元数据块（分析师联系方式/免责声明/地址/参考文献等
    # 本来就不以句号结尾）必须分开统计，否则"是否被从句中切断"会被稀释。
    prose = [c for c in chunks if any(p in c.text for p in "。！？；")]
    prose_end_ok = sum(1 for c in prose if "not_sentence_end" not in c.quality_flags)

    # ★ 最严格的「正文是否被切断」指标：只看组末尾是正文单元的 chunk，
    #   并要求它以句末标点收尾。组末尾是表格/图注/标题的 chunk 不参与
    #   （它们本就不以句号结尾，算进来只会制造假失败）。
    checked = [c for c in chunks if c.trailing_prose_end is not None]
    cut = sum(1 for c in checked if not c.trailing_prose_end)

    def _p50(values: List[int]) -> int:
        s = sorted(values)
        return s[len(s) // 2]

    return {
        "chunk_count": n,
        "chunk_len_mean": round(sum(lens) / n, 1),
        "chunk_len_p50": lens[n // 2],
        "chunk_len_p90": lens[min(n - 1, int(n * 0.9))],
        "chunk_under_min_count": under,
        "chunk_hard_split_count": hard,
        "chunk_not_sentence_end_count": bad_end,
        "chunk_sentence_end_ratio": round(1.0 - bad_end / n, 4),
        "chunk_overlap_pair_count": overlaps,
        "chunk_type_counts": type_counts,
        # 分层指标：表格/图注本来就不以句号结尾，混在一起会掩盖真实问题
        "sentence_end_ratio_by_type": {
            t: round(type_end_ok.get(t, 0) / type_total[t], 4) for t in type_total
        },
        "chunk_len_p50_by_type": {t: _p50(v) for t, v in type_lens.items()},
        # 散文块（含句末标点）的句末完整率
        "prose_chunk_count": len(prose),
        "prose_sentence_end_ratio": (
            round(prose_end_ok / len(prose), 4) if prose else 0.0
        ),
        # ★ 核心验收指标：正文结尾的 chunk 有多少在句末正确收尾
        "trailing_prose_checked": len(checked),
        "trailing_prose_cut_count": cut,
        "trailing_prose_end_ratio": (
            round(1.0 - cut / len(checked), 4) if checked else 0.0
        ),
    }
