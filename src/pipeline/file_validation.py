"""Bounded input validation for document uploads."""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class FileValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FileLimits:
    max_bytes: int = 50 * 1024 * 1024
    max_zip_entries: int = 2000
    max_uncompressed_bytes: int = 500 * 1024 * 1024
    max_pages: int = 2000
    max_table_rows: int = 50000


@dataclass(frozen=True)
class ValidatedFile:
    source_sha256: str
    size_bytes: int
    detected_format: str
    display_name: str


_OOXML_REQUIRED = {
    ".docx": "word/document.xml",
    ".xlsx": "xl/workbook.xml",
    ".pptx": "ppt/presentation.xml",
}
_OLE_EXTENSIONS = {".doc", ".xls", ".ppt", ".pps"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def validate_input_file(path: Path, limits: FileLimits) -> ValidatedFile:
    path = Path(path)
    size = path.stat().st_size
    if size > limits.max_bytes:
        raise FileValidationError("size_limit", f"文件大小超过限制: {path.name}")

    suffix = path.suffix.lower()
    detected = _resolve_signature(path, suffix, limits)
    digest = _streaming_sha256(path)
    return ValidatedFile(
        source_sha256=digest,
        size_bytes=size,
        detected_format=detected,
        display_name=Path(path.name).name,
    )


def _resolve_signature(path: Path, suffix: str, limits: FileLimits) -> str:
    head = _read_head(path, 16)
    if suffix == ".pdf":
        _require(head.startswith(b"%PDF"), "signature_mismatch", "PDF 文件签名不匹配", path)
        return ".pdf"
    if suffix in _OOXML_REQUIRED:
        _validate_ooxml(path, suffix, limits)
        return suffix
    if suffix in _OLE_EXTENSIONS:
        ole_prefix = bytes.fromhex("d0cf11e0a1b11ae1")
        _require(
            head.startswith(ole_prefix),
            "signature_mismatch",
            "Office 旧格式签名不匹配",
            path,
        )
        return suffix
    if suffix == ".rtf":
        _require(head.lstrip().startswith(b"{\\rtf"), "signature_mismatch", "RTF 签名不匹配", path)
        return ".rtf"
    if suffix in {".txt", ".md"}:
        return suffix
    if suffix in _IMAGE_EXTENSIONS:
        _validate_image_signature(head, suffix, path)
        return suffix
    raise FileValidationError("unsupported_format", f"不支持的文件类型: {path.name}")


def _validate_ooxml(path: Path, suffix: str, limits: FileLimits) -> None:
    required = _OOXML_REQUIRED[suffix]
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) > limits.max_zip_entries:
                raise FileValidationError(
                    "archive_entry_limit",
                    f"压缩包条目超过限制: {path.name}",
                )
            if "[Content_Types].xml" not in names:
                raise FileValidationError(
                    "missing_ooxml_part",
                    f"OOXML 缺少 [Content_Types].xml: {path.name}",
                )
            if required not in names:
                raise FileValidationError(
                    "missing_ooxml_part",
                    f"OOXML 缺少 {required}: {path.name}",
                )
            total = 0
            for info in archive.infolist():
                if _is_macro_member(info.filename):
                    raise FileValidationError(
                        "macro_embedded",
                        f"OOXML 包含宏文件: {path.name}",
                    )
                total += info.file_size
                if total > limits.max_uncompressed_bytes:
                    raise FileValidationError(
                        "decompressed_size_limit",
                        f"OOXML 解压后大小超过限制: {path.name}",
                    )
    except zipfile.BadZipFile as exc:
        raise FileValidationError(
            "signature_mismatch",
            f"OOXML 不是有效压缩包: {path.name}",
        ) from exc


def _validate_image_signature(head: bytes, suffix: str, path: Path) -> None:
    if suffix == ".png":
        _require(head.startswith(b"\x89PNG\r\n\x1a\n"), "signature_mismatch", "PNG 签名不匹配", path)
    elif suffix in {".jpg", ".jpeg"}:
        _require(head.startswith(b"\xff\xd8\xff"), "signature_mismatch", "JPEG 签名不匹配", path)
    elif suffix == ".bmp":
        _require(head.startswith(b"BM"), "signature_mismatch", "BMP 签名不匹配", path)
    elif suffix == ".tiff":
        _require(
            head.startswith((b"II*\x00", b"MM\x00*")),
            "signature_mismatch",
            "TIFF 签名不匹配",
            path,
        )


def _is_macro_member(name: str) -> bool:
    lowered = name.lower()
    return "vba" in lowered and lowered.endswith((".bin", ".vba"))


def _require(condition: bool, code: str, message: str, path: Path) -> None:
    if not condition:
        raise FileValidationError(code, f"{message}: {path.name}")


def _read_head(path: Path, size: int) -> bytes:
    with path.open("rb") as source:
        return source.read(size)


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
