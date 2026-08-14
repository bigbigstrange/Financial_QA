#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唯一入口：加载已有向量索引（不强制重建），读取问题表，生成全部答案 JSON。

默认路径（均在 code 目录下相对本文件）：
  - 知识库: output/knowledge_base.json
  - 问题表: test/merged.xlsx（需含 question_id、question 列）
  - 输出:   output/qa_results.json（同目录生成 qa_results.md）
  - 向量缓存: 环境变量 CACHE_DIR；未设置时为项目根下 vector_cache（与 code/ 同级，例如 …/代码和模型/vector_cache）

请先完成 preprocess 与索引构建（scripts/build_index.py，默认用缓存不全量重算）。
换知识库或换嵌入模型后，再运行 build_index 并加 --force-rebuild。

生成模型默认使用本地 F:/Tools/money_tools/models/deepseek-r1-1.5b（可用环境变量 FINANCE_GEN_MODEL 覆盖）。
"""

from __future__ import annotations

import faulthandler
import os
import sys

# 原生扩展崩溃时尽量在终端打出 C 栈（Windows 上若直接闪退可配合事件查看器）
faulthandler.enable()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cache_paths import prune_legacy_and_active, resolve_cache_dir, strip_roaming_user_site_from_sys_path

strip_roaming_user_site_from_sys_path()

from src.system.model_settings import resolve_embed_device
from src.system.qa_engine import FinancialQASystem

DEFAULT_KB = ROOT / "output" / "knowledge_base.json"
DEFAULT_QUESTIONS = ROOT / "test" / "merged.xlsx"
DEFAULT_OUT = ROOT / "output" / "qa_results.json"
DEFAULT_CACHE = resolve_cache_dir()
LEGACY_VECTOR_CACHE = ROOT / "vector_cache"


def main() -> None:
    kb = DEFAULT_KB
    questions = DEFAULT_QUESTIONS
    out_json = DEFAULT_OUT

    if not kb.is_file():
        print(f"错误：找不到知识库文件: {kb}")
        sys.exit(1)
    if not questions.is_file():
        print(f"错误：找不到问题表: {questions}")
        sys.exit(1)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("金融问答 — 批量生成答案")
    print(f"知识库: {kb}")
    print(f"问题表: {questions}")
    print(f"输出:   {out_json}")
    print(f"向量缓存: {DEFAULT_CACHE}")
    print(f"向量编码设备: {resolve_embed_device()}")
    print("向量索引: 仅加载已有缓存（force_rebuild=False）")
    print("=" * 50)

    removed = prune_legacy_and_active(LEGACY_VECTOR_CACHE, DEFAULT_CACHE)
    if removed:
        print(f"已清理不完整缓存文件 {len(removed)} 个（含旧 code/vector_cache 与当前缓存目录）")

    qa = FinancialQASystem(mmd_directory=str(kb), cache_dir=str(DEFAULT_CACHE))
    qa.build_vector_index(force_rebuild=False)

    print(f"知识块数量: {len(qa.knowledge_chunks)}，开始逐题推理…")
    results = qa.process_test_set(str(questions))
    qa.save_results(results, str(out_json))

    print(f"\n完成：已写入 {out_json} 与 {out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
