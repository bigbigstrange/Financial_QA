# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEV_DATABASE_URL = f"sqlite:///{(ROOT / 'data_app.db').as_posix()}"
DEV_CORS_ORIGINS = (
    "http://127.0.0.1:5173,http://localhost:5173,"
    "http://127.0.0.1:8000,http://localhost:8000"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        protected_namespaces=("settings_",),
    )

    app_name: str = "金融知识库问答平台"
    api_prefix: str = "/api"
    app_env: Literal["development", "production"] = "development"
    debug: bool = False

    # SQLite 便于本机一键跑通；生产可改 PostgreSQL
    database_url: str = DEV_DATABASE_URL

    jwt_secret: str = "change-me-finance-qa-dev-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    cors_origins: str = DEV_CORS_ORIGINS

    kb_path: str = str(ROOT / "output" / "knowledge_base.json")
    cache_dir: str = str(ROOT / "vector_cache")
    kg_dir: str = str(ROOT / "output")
    model_registry_path: str = str(ROOT / "configs" / "model_registry.json")
    eval_golden_path: str = str(ROOT / "test" / "merged.xlsx")
    eval_output_dir: str = str(ROOT / "output" / "evals")
    doc_artifact_root: str = str(ROOT / "output" / "phase2")

    finance_web_demo: bool = True
    # 离线演示模式：不依赖大模型/向量库，用示例语料演示检索与拒答链路
    finance_demo_mode: bool = False
    enable_graph: bool = False
    skip_image: bool = True
    max_blocks: int = 8000
    embed_batch_size: int = 8
    image_backend: str = "visrag"
    # 阶段 C：Agent 编排模式。graph=LangGraph 编排；legacy=原有轻量 DocumentAgent
    finance_agent_mode: str = "legacy"

    # 内存水位（百分比相对本机/容器物理内存）
    mem_warn_pct: float = 70.0
    mem_soft_pct: float = 85.0
    mem_hard_pct: float = 92.0
    # 生成模型空闲卸载（秒）；0=不自动卸载
    gen_idle_unload_sec: int = 600
    # 同时推理并发上限，防止堆叠 OOM
    max_concurrent_infer: int = 1
    run_timeout_seconds: int = Field(default=300, ge=1, le=3600)

    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: str = "admin123"
    bootstrap_admin_email: str = "admin@example.com"

    host: str = "127.0.0.1"
    port: int = 8000

    # pydantic-settings 自动映射同名环境变量：MEM_HARD_PCT 等

    @model_validator(mode="after")
    def validate_production(self) -> "Settings":
        if self.app_env != "production":
            if self.host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("non-loopback HOST requires APP_ENV=production")
            return self
        if self.debug:
            raise ValueError("DEBUG must be false in production")
        if self.jwt_algorithm != "HS256":
            raise ValueError("production JWT_ALGORITHM must be HS256")
        secret = self.jwt_secret.strip()
        password = self.bootstrap_admin_password.strip()
        if len(secret) < 32 or secret in {"change-me-finance-qa-dev-secret", "change-me-in-production"}:
            raise ValueError("production requires a unique JWT_SECRET of at least 32 characters")
        if len(password) < 12 or password == secret:
            raise ValueError("production requires a distinct BOOTSTRAP_ADMIN_PASSWORD of at least 12 characters")
        if "*" in self.cors_origins or self.cors_origins == DEV_CORS_ORIGINS:
            raise ValueError("production requires explicit CORS_ORIGINS (empty for same-origin)")
        if self.database_url == DEV_DATABASE_URL:
            raise ValueError("production requires an explicit persistent DATABASE_URL")
        if self.database_url.startswith("sqlite:"):
            path = self.database_url.removeprefix("sqlite:///")
            if path == self.database_url or not (path.startswith("/") or Path(path).is_absolute()):
                raise ValueError("production SQLite DATABASE_URL must use an absolute path")
        return self

    @property
    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

