# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "金融知识库问答平台"
    api_prefix: str = "/api"
    debug: bool = False

    # SQLite 便于本机一键跑通；生产可改 PostgreSQL
    database_url: str = f"sqlite:///{(ROOT / 'data_app.db').as_posix()}"

    jwt_secret: str = "change-me-finance-qa-dev-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    cors_origins: str = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "http://127.0.0.1:8000,http://localhost:8000"
    )

    kb_path: str = str(ROOT / "output" / "knowledge_base.json")
    cache_dir: str = str(ROOT / "vector_cache")
    kg_dir: str = str(ROOT / "output")
    model_registry_path: str = str(ROOT / "configs" / "model_registry.json")
    eval_golden_path: str = str(ROOT / "test" / "merged.xlsx")
    eval_output_dir: str = str(ROOT / "output" / "evals")

    finance_web_demo: bool = True
    enable_graph: bool = False
    skip_image: bool = True
    max_blocks: int = 8000
    embed_batch_size: int = 8
    image_backend: str = "visrag"

    # 内存水位（百分比相对本机/容器物理内存）
    mem_warn_pct: float = 70.0
    mem_soft_pct: float = 85.0
    mem_hard_pct: float = 92.0
    # 生成模型空闲卸载（秒）；0=不自动卸载
    gen_idle_unload_sec: int = 600
    # 同时推理并发上限，防止堆叠 OOM
    max_concurrent_infer: int = 1

    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: str = "admin123"
    bootstrap_admin_email: str = "admin@example.com"

    host: str = "127.0.0.1"
    port: int = 8000

    # pydantic-settings 自动映射同名环境变量：MEM_HARD_PCT 等

    @property
    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
