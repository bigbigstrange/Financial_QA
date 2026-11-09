"""Canonical identities for immutable document versions and evidence blocks."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def stable_digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParseContext:
    document_id: str
    source_name: str
    source_sha256: str
    source_size_bytes: int
    parser_config_hash: str
    chunker_version: str

    @classmethod
    def for_file(
        cls,
        file_path: str | Path,
        document_id: str,
        parser_config_hash: str = "parser-v2",
        chunker_version: str = "chunk-v2",
    ) -> "ParseContext":
        path = Path(file_path)
        return cls(
            document_id=document_id,
            source_name=path.name,
            source_sha256=file_sha256(path),
            source_size_bytes=path.stat().st_size,
            parser_config_hash=parser_config_hash,
            chunker_version=chunker_version,
        )


def make_document_version(
    context: ParseContext,
    parser_name: str,
    parser_version: str,
) -> str:
    return "dv_" + stable_digest(
        context.document_id,
        context.source_sha256,
        "2",
        parser_name,
        parser_version,
        context.parser_config_hash,
        context.chunker_version,
    )[:32]


def make_block_id(
    document_version: str,
    structural_locator: str,
    block_type: str,
) -> str:
    return "blk_" + stable_digest(
        document_version,
        structural_locator,
        block_type,
    )[:32]


def artifact_key(rel_path: str) -> str:
    """把相对路径映射成稳定、安全、定长的产物目录名。

    中文/超长路径在 Windows 上易触发路径长度与编码问题，
    因此产物目录统一用「净化后的文件名前缀 + 路径摘要」命名。
    解析脚本与知识库构建脚本共用此函数，避免两边实现漂移。
    """
    digest = hashlib.sha256(str(rel_path).encode("utf-8")).hexdigest()[:20]
    stem = Path(rel_path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:40]
    return f"{safe}__{digest}"
