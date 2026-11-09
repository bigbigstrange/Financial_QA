# -*- coding: utf-8 -*-
"""测试引导：让 `pytest` 在全新克隆的仓库里也能跑通。

背景：部分测试（如模型注册表的晋级校验）会查询 `eval_runs` 等表，
这些表原先依赖本地已存在的 `data_app.db`，导致新环境首次运行即失败。
这里在会话开始时创建一个独立的临时 SQLite 库并建表，测试之间互不污染。

注意：只覆盖 DATABASE_URL，不改动 Agent 模式与付费开关，
以免影响依赖真实运行语义的用例。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="finance-qa-pytest-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP_DIR / 'test.db').as_posix()}")


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_database() -> None:
    """建表 + schema 升级，保证依赖数据库的测试可独立运行。"""
    from app.db.init_db import init_db

    init_db()
    yield
