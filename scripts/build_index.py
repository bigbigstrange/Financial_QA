#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量索引构建脚本
用于构建FAISS向量索引
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_ROOT))

from cache_paths import prune_legacy_and_active, resolve_cache_dir

from src.system import FinancialQASystem

def build_vector_index(mmd_path, cache_dir=None, force_rebuild=True):
    """
    构建向量索引
    
    Args:
        mmd_path: 合并的markdown文件路径
        cache_dir: 缓存目录（默认与 FinancialQASystem 一致：CACHE_DIR 或 <项目根>/vector_cache）
        force_rebuild: 为 True 时忽略已有 FAISS 缓存并全量重算向量
    """
    if cache_dir is None:
        cache_dir = str(resolve_cache_dir())
    pruned = prune_legacy_and_active(CODE_ROOT / "vector_cache", Path(cache_dir).resolve())
    if pruned:
        print(f"已清理不完整缓存文件 {len(pruned)} 个")

    print(f"=========================================")
    print(f"开始构建向量索引...")
    print(f"知识库文件: {mmd_path}")
    print(f"缓存目录: {cache_dir}")
    print(f"强制重建: {force_rebuild}")
    print(f"=========================================")
    
    # 检查文件是否存在
    if not os.path.exists(mmd_path):
        print(f"[错误] 知识库文件不存在: {mmd_path}")
        print(f"请先运行 preprocess.py 生成知识库文件")
        return
    
    # 初始化问答系统
    qa_system = FinancialQASystem(
        mmd_directory=mmd_path,
        cache_dir=cache_dir
    )
    
    try:
        # 构建向量索引
        qa_system.build_vector_index(force_rebuild=force_rebuild)
        
        print(f"\n=========================================")
        print(f"[完成] 向量索引构建完成！")
        print(f"文本块数量: {len(qa_system.knowledge_chunks)}")
        print(f"向量索引状态: {'ready' if qa_system.vector_index is not None else 'not_ready'}")
        print(f"=========================================")
        
    except Exception as e:
        print(f"[失败] 构建失败: {e}")
        import traceback
        traceback.print_exc()

def clear_cache(cache_dir=None):
    """清理向量缓存"""
    import shutil

    if cache_dir is None:
        cache_dir = str(resolve_cache_dir())
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"[完成] 已清理缓存目录: {cache_dir}")
    else:
        print(f"[提示] 缓存目录不存在: {cache_dir}")

def list_cache(cache_dir=None):
    """列出缓存文件"""
    if cache_dir is None:
        cache_dir = str(resolve_cache_dir())
    cache_path = Path(cache_dir)
    if cache_path.exists():
        print(f"缓存文件列表 ({cache_dir}):")
        for file in cache_path.glob("*"):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  - {file.name} ({size_mb:.1f} MB)")
    else:
        print(f"缓存目录不存在: {cache_dir}")

def main():
    parser = argparse.ArgumentParser(description='向量索引构建脚本')
    parser.add_argument('--mmd-path', '-m', 
                        default='./output/knowledge_base.json',
                        help='知识库JSON文件路径 (默认: ./output/knowledge_base.json)')
    parser.add_argument(
        '--cache-dir',
        '-c',
        default=None,
        help='缓存目录（默认：环境变量 CACHE_DIR，否则为 <项目根>/vector_cache，与 code/ 同级）',
    )
    parser.add_argument(
        '--force-rebuild',
        '-f',
        action='store_true',
        help='忽略已有缓存，全量重算向量（默认：有缓存则跳过重建）',
    )
    parser.add_argument('--clear-cache', action='store_true',
                        help='清空 vector_cache 后全量重建')
    parser.add_argument('--list-cache', action='store_true',
                        help='列出缓存文件')
    
    args = parser.parse_args()

    cache_dir = args.cache_dir if args.cache_dir else str(resolve_cache_dir())

    if args.list_cache:
        list_cache(cache_dir)
        return

    if args.clear_cache:
        clear_cache(cache_dir)

    build_vector_index(
        mmd_path=args.mmd_path,
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild or args.clear_cache,
    )

if __name__ == "__main__":
    main()