#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.system.kg import build_kg_from_kb


def main():
    parser = argparse.ArgumentParser(description="离线知识图谱抽取脚本")
    parser.add_argument("--kb-path", default="./output/knowledge_base.json", help="知识库 JSON 路径")
    parser.add_argument("--output-dir", default="./output", help="输出目录")
    args = parser.parse_args()

    result = build_kg_from_kb(args.kb_path, args.output_dir)
    print("KG build done:")
    for k, v in result.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
