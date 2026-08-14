#!/bin/bash

# 构建索引脚本
# 用于在Docker环境中构建向量索引

set -e

echo "========================================="
echo "开始构建向量索引..."
echo "========================================="

# 环境变量
MMD_PATH=${MMD_PATH:-"/app/output/knowledge_base.json"}
CACHE_DIR=${CACHE_DIR:-"/app/vector_cache"}
FORCE_REBUILD=${FORCE_REBUILD:-"false"}

echo "配置信息:"
echo "  - 知识库路径: $MMD_PATH"
echo "  - 缓存目录: $CACHE_DIR"
echo "  - 强制重建: $FORCE_REBUILD"
echo

# 检查知识库文件是否存在
if [ ! -f "$MMD_PATH" ]; then
    echo "❌ 错误: 知识库文件不存在: $MMD_PATH"
    echo "请先运行 preprocess.py 生成知识库"
    exit 1
fi

# 构建索引
EXTRA=""
if [ "$FORCE_REBUILD" = "true" ]; then
  EXTRA="--force-rebuild"
fi
python /app/scripts/build_index.py \
    --mmd-path "$MMD_PATH" \
    --cache-dir "$CACHE_DIR" \
    $EXTRA

echo "========================================="
echo "✅ 向量索引构建完成！"
echo "========================================="