"""Build a single trusted page-image request for the bounded Ark pilot."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any


def pilot_visual_approved(evidence: dict) -> bool:
    """Only an operator-reviewed page ID and digest may be sent to Ark."""
    page_id = (os.environ.get("FINANCE_PILOT_VISUAL_PAGE_ID") or "").strip()
    digest = (os.environ.get("FINANCE_PILOT_VISUAL_SHA256") or "").strip().lower()
    if (os.environ.get("FINANCE_PILOT_VISUAL") != "1" or not page_id or
            len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
        return False
    return (hmac.compare_digest(str(evidence.get("page_id") or ""), page_id) and
            hmac.compare_digest(str(evidence.get("image_sha256") or "").lower(), digest))


def build_image_messages(question: str, evidence: dict, allowed_root: Path) -> list[dict[str, Any]]:
    reference = evidence.get("image_ref") or evidence.get("image_path")
    source = evidence.get("source_doc")
    page = evidence.get("vision_page") or evidence.get("page_start") or evidence.get("page")
    if not reference or not isinstance(source, str) or not source or type(page) is not int or page < 1:
        raise ValueError("图像证据缺少可信来源或页码")
    root = Path(allowed_root).resolve(strict=True)
    path = Path(reference).resolve(strict=True) if Path(reference).exists() else None
    if path is None or not path.is_relative_to(root):
        raise ValueError("图像不在可信页图目录或文件不存在")
    if path.stat().st_size > 5_000_000:
        raise ValueError("图像大于 5 MB 上限")
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError("只允许 PNG 或 JPEG 页图")
    digest = evidence.get("image_sha256")
    if not isinstance(digest, str) or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("页图与检索证据的摘要不符")
    data_url = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": f"请仅根据这张页图回答：{question}\n来源：{source} 第{page}页。看不清时明确说明，勿推测。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]
