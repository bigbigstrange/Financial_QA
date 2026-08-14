# -*- coding: utf-8 -*-
"""
企业平台入口：鉴权会话 + 问答 + 内存防护 + 模型迭代评测。

启动:
  python -m app.main
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 尽早加载 .env（含 DEEPSEEK_API_KEY），且不覆盖已有环境变量
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

# 演示模式：禁用 CUDA + 限制线程，避免 Windows 原生闪退（无 Python 堆栈）
_demo_flag = (os.environ.get("FINANCE_WEB_DEMO") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "n",
}
_has_api_key = bool(
    (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("FINANCE_API_KEY") or "").strip()
)
if _demo_flag:
    # 必须在 import torch 之前生效
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("FINANCE_EMBED_DEVICE", "cpu")
    os.environ.setdefault("FINANCE_GEN_DEVICE", "cpu")
    os.environ.setdefault("FINANCE_MAX_BLOCKS", "2000")
    os.environ.setdefault("FINANCE_SKIP_IMAGE", "1")
    os.environ.setdefault("ENABLE_GRAPH", "false")
    os.environ.setdefault("FINANCE_EMBED_BATCH_SIZE", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault(
        "FINANCE_DEMO_KEYWORDS",
        "掌上银行|网银盾|数字证书|证书在线|客服|浦东发展|浦发|建行|基金|账户|网银|"
        "手机银行|邮政储蓄|邮储|系统升级|95580|40088|交通银行|交行|USBkey|USBKey|"
        "总资产|我的账户|明细查询|激活",
    )
    # 有 API Key：云端生成；重排默认 hybrid（简单 heuristic，难例 api）
    os.environ.setdefault("FINANCE_RERANK_BACKEND", "hybrid")
    if _has_api_key:
        os.environ.setdefault("FINANCE_GEN_BACKEND", "api")
        os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
        os.environ["FINANCE_SKIP_GEN"] = "0"
    else:
        os.environ.setdefault("FINANCE_SKIP_GEN", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, chat, ops
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.init_db import init_db
from app.services.qa_runtime import qa_runtime

settings = get_settings()
setup_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 只做轻量初始化；模型加载放到 main() 主线程，避免 Windows 后台线程闪退
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(ops.router, prefix=settings.api_prefix)


@app.get("/api/health")
def legacy_health():
    return qa_runtime.status()


FRONTEND_DIST = ROOT / "frontend" / "dist"
WEB_DIR = ROOT / "web"

if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="fe-assets")

    @app.get("/")
    def spa_index():
        return FileResponse(FRONTEND_DIST / "index.html")

elif WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def web_index():
        return FileResponse(WEB_DIR / "index.html")


def main() -> None:
    import uvicorn

    print("=" * 56)
    print(settings.app_name)
    print(f"API:  http://{settings.host}:{settings.port}/api")
    print(f"Docs: http://{settings.host}:{settings.port}/docs")
    print(f"Admin: {settings.bootstrap_admin_user} / {settings.bootstrap_admin_password}")
    print(f"Demo memory mode: {settings.finance_web_demo}")
    print(f"Embed device: {os.environ.get('FINANCE_EMBED_DEVICE', 'auto')}")
    print(f"Gen backend: {os.environ.get('FINANCE_GEN_BACKEND', 'auto')}")
    print(f"Rerank backend: {os.environ.get('FINANCE_RERANK_BACKEND', 'auto')}")
    print(f"DeepSeek model: {os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')}")
    print(f"API key loaded: {'yes' if _has_api_key else 'no'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print("=" * 56)

    # 主线程预热（不要用后台线程加载 torch，Windows 易原生崩溃）
    skip_warm = (os.environ.get("FINANCE_SKIP_WARMUP") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not skip_warm:
        print("[platform] loading QA engine in main thread (may take 1-3 min)...", flush=True)
        try:
            qa_runtime.ensure_loaded()
            print("[platform] QA engine ready.", flush=True)
        except Exception as e:
            print(f"[platform] warmup failed, will retry on first request: {e}", flush=True)

    # 传 app 对象，避免再 import 一遍丢环境变量副作用
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
