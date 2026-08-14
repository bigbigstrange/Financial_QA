# -*- coding: utf-8 -*-
"""向量缓存目录：默认放在仓库根目录 vector_cache（与 code/ 同级），可用 CACHE_DIR 覆盖。

放在 code/ 顶层，避免 import src.system.* 时连带加载 pandas 等重依赖。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Set


def strip_roaming_user_site_from_sys_path() -> None:
    """
    Windows + conda 时，Python 会把 %APPDATA%\\Python\\...\\site-packages 插到 sys.path，
    与用户目录里的 sklearn/scipy 混载，链路极重，易在低内存机器上 import MemoryError。
    在导入 sentence_transformers 等重库之前调用一次。
    """
    if sys.platform != "win32":
        return
    for p in list(sys.path):
        norm = p.replace("\\", "/")
        if "site-packages" in norm and "/Roaming/Python" in norm:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def code_root() -> Path:
    """code/ 目录（与本文件同级）。"""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """含 code/ 的仓库根目录（vector_cache 默认与此同级）。"""
    return code_root().parent


def default_project_vector_cache() -> Path:
    """默认缓存：<项目根>/vector_cache。"""
    return project_root() / "vector_cache"


def resolve_cache_dir(explicit: Optional[str] = None) -> Path:
    """
    解析 FAISS 等向量缓存目录。

    优先级：显式参数 > 环境变量 CACHE_DIR > <项目根>/vector_cache
    """
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = (os.environ.get("CACHE_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return default_project_vector_cache()


def prune_incomplete_vector_cache(cache_dir: Path) -> List[str]:
    """
    删除不完整的 *.faiss / 同名 *.pkl 对（0 字节、缺一半、孤立 pkl）。
    返回已删除文件路径列表。
    """
    removed: List[str] = []
    d = Path(cache_dir)
    if not d.is_dir():
        return removed

    for faiss_path in list(d.glob("*.faiss")):
        pkl_path = faiss_path.with_suffix(".pkl")
        bad = False
        try:
            if faiss_path.stat().st_size == 0:
                bad = True
            elif not pkl_path.is_file():
                bad = True
            elif pkl_path.stat().st_size == 0:
                bad = True
        except OSError:
            bad = True
        if bad:
            for p in (faiss_path, pkl_path):
                if p.is_file():
                    p.unlink()
                    removed.append(os.fspath(p))

    for pkl_path in list(d.glob("*.pkl")):
        if not pkl_path.with_suffix(".faiss").is_file():
            try:
                pkl_path.unlink()
                removed.append(os.fspath(pkl_path))
            except OSError:
                pass

    return removed


def prune_legacy_and_active(legacy_under_code: Path, active: Path) -> List[str]:
    """在 legacy（code/vector_cache）与当前 active 目录上各执行一次 prune，去重。"""
    out: List[str] = []
    seen: Set[str] = set()
    for p in (legacy_under_code.resolve(), active.resolve()):
        key = os.fspath(p)
        if key in seen:
            continue
        seen.add(key)
        out.extend(prune_incomplete_vector_cache(p))
    return out
