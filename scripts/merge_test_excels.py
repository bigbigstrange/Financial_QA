#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 test 目录下的两个 Excel 文件（按行纵向拼接，列名对齐，缺失填 NaN）。

默认目录：优先 <项目根>/code/test（若其中有至少 2 个 xlsx），否则 <项目根>/test。
也可通过参数指定目录或两个文件路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    # code/scripts/this_file.py -> 上级的上级的上级 = 项目根（与 code 同级）
    return Path(__file__).resolve().parent.parent.parent


def _default_test_dir() -> Path:
    root = _repo_root()
    code_test = root / "code" / "test"
    if code_test.is_dir() and len(_collect_xlsx(code_test)) >= 2:
        return code_test
    return root / "test"


def _collect_xlsx(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(d.glob("*.xlsx"), key=lambda x: x.name.lower()):
        if p.name.lower() == "merged.xlsx":
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 test 目录下的两个 Excel（纵向拼接）")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="存放 xlsx 的目录，默认：优先 <项目根>/code/test，否则 <项目根>/test",
    )
    parser.add_argument("--file1", type=Path, default=None, help="第一个 xlsx（可选，与 --file2 成对使用）")
    parser.add_argument("--file2", type=Path, default=None, help="第二个 xlsx")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出路径，默认：<目录>/merged.xlsx",
    )
    parser.add_argument(
        "--sheet",
        type=int,
        default=0,
        help="读取的工作表索引或名称：默认 0（第一个表）；也可传字符串表名",
    )
    args = parser.parse_args()

    if args.file1 is not None or args.file2 is not None:
        if args.file1 is None or args.file2 is None:
            print("错误：--file1 与 --file2 必须同时指定。", file=sys.stderr)
            return 1
        f1, f2 = args.file1.resolve(), args.file2.resolve()
        for p in (f1, f2):
            if not p.is_file():
                print(f"错误：文件不存在: {p}", file=sys.stderr)
                return 1
        out_dir = f1.parent
    else:
        test_dir = (args.dir or _default_test_dir()).resolve()
        files = _collect_xlsx(test_dir)
        if len(files) < 2:
            print(
                f"错误：在目录中需要至少 2 个 .xlsx 文件: {test_dir}\n"
                f"当前找到: {[p.name for p in files]}\n"
                "请将两个表格放入该目录，或使用 --file1 / --file2 指定路径。",
                file=sys.stderr,
            )
            return 1
        if len(files) > 2:
            print(
                f"提示：目录中共 {len(files)} 个 xlsx，将按文件名排序取前两个合并：\n"
                f"  1) {files[0].name}\n  2) {files[1].name}\n"
                "若需指定其它组合，请使用 --file1 与 --file2。",
            )
        f1, f2 = files[0], files[1]
        out_dir = test_dir

    try:
        df1 = pd.read_excel(f1, sheet_name=args.sheet)
        df2 = pd.read_excel(f2, sheet_name=args.sheet)
    except Exception as e:
        print(f"读取 Excel 失败: {e}", file=sys.stderr)
        return 1

    merged = pd.concat([df1, df2], axis=0, ignore_index=True, sort=False)

    output = args.output
    if output is None:
        output = out_dir / "merged.xlsx"
    else:
        output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        merged.to_excel(output, index=False, engine="openpyxl")
    except Exception as e:
        print(f"写入失败: {e}", file=sys.stderr)
        return 1

    print(f"已合并: {f1.name} + {f2.name} -> {output}")
    print(f"行数: {len(df1)} + {len(df2)} = {len(merged)}，列数: {len(merged.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
