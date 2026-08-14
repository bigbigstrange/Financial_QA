#!/bin/bash
# 容器内批量生成答案（与本地直接运行 python main.py 等价）

set -e
echo "运行 python main.py …"
exec python /app/main.py
