# -*- coding: utf-8 -*-
"""
本地模型根目录与 Hugging Face 镜像。

- FINANCE_MODEL_ROOT：其它本地模型（bge、clip、reranker 等）根目录，默认 F:/Tools/money_tools/models。
- FINANCE_GEN_MODEL：生成模型目录；未设置时默认使用 F:/Tools/money_tools/models/deepseek-r1-1.5b（若存在）。
- 未命中本地目录时仍使用原 Hugging Face 模型 ID，由 HF_ENDPOINT 走镜像拉取。
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def ensure_hf_mirror() -> None:
    """在未显式配置时使用 hf-mirror，须在首次联网拉取模型前调用。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def local_model_root() -> Path:
    return Path(os.environ.get("FINANCE_MODEL_ROOT", "F:/Tools/money_tools/models"))


# 默认生成模型（你本机常用路径；可用环境变量 FINANCE_GEN_MODEL 覆盖）
_DEFAULT_GEN_LOCAL = Path(r"F:/Tools/money_tools/models/deepseek-r1-1.5b")


def _is_usable_hf_folder(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_embed_model(fallback: str = "BAAI/bge-m3") -> str:
    """文本向量：本地 bge_large（sentence_transformers 快照）否则回退并走镜像。"""
    p = local_model_root() / "bge_large"
    return str(p.resolve()) if _is_usable_hf_folder(p) else fallback


def resolve_gen_model(fallback: str = "Qwen/Qwen2-1.5B-Instruct") -> str:
    """生成模型：FINANCE_GEN_MODEL > 完整本地目录 > HF ID。

    仅有空目录/无 config.json 的路径会被跳过，避免加载失败后只能拼接证据。
    """
    env = os.environ.get("FINANCE_GEN_MODEL", "").strip()
    if env:
        p = Path(env)
        if _is_usable_hf_folder(p):
            return str(p.resolve())
        if (not p.exists()) and ("/" in env) and (not re.match(r"^[A-Za-z]:", env)):
            return env
        print(f"[gen] FINANCE_GEN_MODEL 无效（缺 config.json）: {env}", flush=True)
    if _is_usable_hf_folder(_DEFAULT_GEN_LOCAL):
        return str(_DEFAULT_GEN_LOCAL.resolve())
    if _DEFAULT_GEN_LOCAL.is_dir():
        print(
            f"[gen] 跳过不完整目录 {_DEFAULT_GEN_LOCAL}（无 config.json，可能未下载完）。",
            flush=True,
        )
    for name in ("Qwen2-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", "deepseek-r1-1.5b", "DeepSeek-R1-Distill-Qwen-1.5B"):
        p = local_model_root() / name
        if _is_usable_hf_folder(p):
            return str(p.resolve())
    return fallback


def resolve_clip_model(fallback: str = "openai/clip-vit-base-patch32") -> str:
    p = local_model_root() / "clip"
    return str(p.resolve()) if _is_usable_hf_folder(p) else fallback


def resolve_colpali(fallback: str = "vidore/colpali-v1.2") -> str:
    p = local_model_root() / "colpali-v1.2"
    return str(p.resolve()) if _is_usable_hf_folder(p) else fallback


def resolve_embed_device() -> str:
    """
    文本/图像向量编码设备。
    FINANCE_EMBED_DEVICE：cuda | cuda:0 | cpu | auto（默认 auto：有 GPU 用 cuda，否则 cpu）
    """
    import torch

    env = os.environ.get("FINANCE_EMBED_DEVICE", "auto").strip().lower()
    if env == "cpu":
        return "cpu"
    if env.startswith("cuda"):
        return env if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_embed_batch_size(device: str) -> int:
    """encode 批大小；可用 FINANCE_EMBED_BATCH_SIZE 覆盖。"""
    env = os.environ.get("FINANCE_EMBED_BATCH_SIZE", "").strip()
    if env.isdigit():
        return max(1, int(env))
    return 64 if str(device).startswith("cuda") else 16


# 任意子模块 import model_settings 后即设置镜像，避免仅 import agent 时未走 retrieval 的情况
ensure_hf_mirror()
