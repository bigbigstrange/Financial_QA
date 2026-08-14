#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融多模态问答 — 简易 Web 前端入口。

默认启用「演示精简模式」，避免全量 40 万+ 块把内存撑爆：
  FINANCE_WEB_DEMO=1（默认）
  FINANCE_MAX_BLOCKS=8000
  FINANCE_DEMO_KEYWORDS=掌上银行|网银盾|…
  FINANCE_SKIP_IMAGE=1
  ENABLE_GRAPH=false

启动（在 code 目录）:
  python web_app.py

浏览器打开 http://127.0.0.1:7860

全量模式（需大内存）:
  $env:FINANCE_WEB_DEMO='0'; python web_app.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Web 默认演示模式：精简知识块 + 跳过图像塔，保证本机能启动
_DEMO = (os.environ.get("FINANCE_WEB_DEMO") or "1").strip().lower() not in {"0", "false", "no", "n"}
if _DEMO:
    os.environ.setdefault(
        "FINANCE_DEMO_KEYWORDS",
        "掌上银行|网银盾|数字证书|证书在线|客服|浦东发展|浦发|建行|基金|账户|网银|手机银行|邮政储蓄|邮储|系统升级|95580|40088|交通银行|交行|USBkey|USBKey|总资产|我的账户|明细查询|激活",
    )
    os.environ.setdefault("FINANCE_MAX_BLOCKS", "8000")
    os.environ.setdefault("FINANCE_SKIP_IMAGE", "1")
    os.environ.setdefault("ENABLE_GRAPH", "false")
    os.environ.setdefault("FINANCE_EMBED_BATCH_SIZE", "8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cache_paths import prune_legacy_and_active, resolve_cache_dir, strip_roaming_user_site_from_sys_path

strip_roaming_user_site_from_sys_path()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.system.model_settings import resolve_embed_device, resolve_gen_model
from src.system.qa_engine import FinancialQASystem

WEB_DIR = ROOT / "web"
DEFAULT_KB = ROOT / "output" / "knowledge_base.json"
DEFAULT_CACHE = resolve_cache_dir()
LEGACY_VECTOR_CACHE = ROOT / "vector_cache"

app = FastAPI(title="金融多模态问答", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_qa: Optional[FinancialQASystem] = None
_state: Dict[str, Any] = {
    "ready": False,
    "loading": False,
    "error": None,
    "kb_path": str(DEFAULT_KB),
    "cache_dir": str(DEFAULT_CACHE),
    "chunks": 0,
    "gen_model": "",
    "embed_device": "",
    "demo": _DEMO,
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(12, ge=1, le=30)
    show_thinking: bool = Field(False, description="是否在响应中附带思考过程（前端开关）")


class AskResponse(BaseModel):
    question_id: str
    question: str
    answer: str
    thinking: str = ""
    sub_questions: List[str] = []
    confidence: float
    evidence: List[Dict[str, Any]]
    elapsed_sec: float
    show_thinking: bool = False


def _load_engine() -> None:
    global _qa
    if _state["ready"] and _qa is not None:
        return
    if _state["loading"]:
        raise HTTPException(status_code=503, detail="模型正在加载，请稍候…")

    _state["loading"] = True
    _state["error"] = None
    try:
        if not DEFAULT_KB.is_file():
            raise FileNotFoundError(f"找不到知识库: {DEFAULT_KB}")

        prune_legacy_and_active(LEGACY_VECTOR_CACHE, DEFAULT_CACHE)
        enable_graph = (os.getenv("ENABLE_GRAPH") or "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        print(
            f"[web] demo={_DEMO} max_blocks={os.getenv('FINANCE_MAX_BLOCKS')} "
            f"skip_image={os.getenv('FINANCE_SKIP_IMAGE')} graph={enable_graph}",
            flush=True,
        )
        qa = FinancialQASystem(
            mmd_directory=str(DEFAULT_KB),
            cache_dir=str(DEFAULT_CACHE),
            kg_dir=str(ROOT / "output"),
            enable_graph=enable_graph,
            image_backend=os.getenv("IMAGE_BACKEND", "visrag"),
        )
        qa.build_vector_index(force_rebuild=False)
        # 不在启动时预热生成模型，避免与向量模型叠加 OOM；首次提问再加载

        _qa = qa
        _state["ready"] = True
        _state["chunks"] = len(qa.knowledge_chunks)
        _state["gen_model"] = resolve_gen_model()
        _state["embed_device"] = resolve_embed_device()
    except Exception as e:
        _state["ready"] = False
        _state["error"] = f"{e}\n{traceback.format_exc()}"
        raise
    finally:
        _state["loading"] = False


@app.on_event("startup")
def on_startup() -> None:
    import threading

    def _bg() -> None:
        try:
            _load_engine()
            print("[web] QA engine ready.", flush=True)
        except Exception as e:
            print(f"[web] QA engine failed to load: {e}", flush=True)

    threading.Thread(target=_bg, daemon=True).start()


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ready": _state["ready"],
        "loading": _state["loading"],
        "error": _state["error"],
        "kb_path": _state["kb_path"],
        "cache_dir": _state["cache_dir"],
        "chunks": _state["chunks"],
        "gen_model": _state["gen_model"],
        "embed_device": _state["embed_device"],
        "demo": _state["demo"],
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问题不能为空")

    if not _state["ready"] or _qa is None:
        if _state["loading"]:
            raise HTTPException(status_code=503, detail="模型正在加载，请稍候刷新状态后再试")
        if _state["error"]:
            raise HTTPException(status_code=503, detail=f"引擎未就绪: {_state['error'][:500]}")
        try:
            _load_engine()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    assert _qa is not None
    qid = f"web-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    try:
        result = _qa.answer_question(q, qid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    elapsed = time.time() - t0
    evidence = result.get("evidence") or []
    slim = []
    for ev in evidence[:8]:
        if isinstance(ev, dict):
            slim.append(
                {
                    "text": (ev.get("text") or "")[:400],
                    "source_doc": ev.get("source_doc"),
                    "page": ev.get("page"),
                    "block_id": ev.get("block_id"),
                    "score": ev.get("score"),
                }
            )
        else:
            slim.append({"text": str(ev)[:400]})

    return AskResponse(
        question_id=qid,
        question=q,
        answer=result.get("answer") or "",
        thinking=(result.get("thinking") or "") if req.show_thinking else "",
        sub_questions=list(result.get("sub_questions") or []),
        confidence=float(result.get("confidence") or 0.0),
        evidence=slim,
        elapsed_sec=round(elapsed, 2),
        show_thinking=bool(req.show_thinking),
    )


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="缺少 web/index.html")
    return FileResponse(index_path)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "7860"))
    print("=" * 50)
    print("金融多模态问答 Web")
    print(f"打开浏览器: http://{host}:{port}")
    print(f"知识库: {DEFAULT_KB}")
    print(f"向量缓存: {DEFAULT_CACHE}")
    print(f"演示精简模式: {_DEMO}")
    if _DEMO:
        print("  （仅加载与示例题相关的文档子集，省内存；全量请设 FINANCE_WEB_DEMO=0）")
    print("=" * 50)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
