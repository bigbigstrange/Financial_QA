# -*- coding: utf-8 -*-
"""测试引导：让 `pytest` 在全新克隆的仓库里也能跑通。

部分测试（如模型注册表晋级校验）会查询 `eval_runs` 等表，这些表原先依赖
本地已存在的 `data_app.db`。这里在会话开始时按默认配置建库建表；
`data_app.db` 已在 .gitignore 中，不会被提交。
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_database() -> None:
    """建表 + schema 升级，保证依赖数据库的测试可独立运行。"""
    from app.db.init_db import init_db

    init_db()
    yield
