"""OpenAI 兼容的本地 embedding 服务（BGE-M3），供 LightRAG 使用。

## 为什么需要它

LightRAG 的 `EMBEDDING_BINDING` 只接受 HTTP 端点（`openai` / `ollama` / `gemini` /
`bedrock` / `jina` …），**没有"本地直接加载模型"的选项**：

    default=get_binding_env_value("EMBEDDING_BINDING", "ollama"),
    "ollama": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),

所以要接本地 BGE-M3，只有两条路：
  1. 装 Ollama 并再下载一份 bge-m3（约 1.2GB）；
  2. **起一个最小的 OpenAI 兼容端点，复用项目里已经下好、已在 GPU 上跑通的 BGE-M3。**

本文件走第 2 条：省掉一个运行时依赖和一次模型下载，且向量与检索侧完全同源
（同一份权重、同一套归一化），不会出现"embedding 模型不一致导致向量不可比"的问题。

## 用法

    python scripts/serve_bge_m3.py --port 9622

    GET  /health
    POST /v1/embeddings   {"input": ["文本1", "文本2"], "model": "bge-m3"}
        -> {"object":"list","data":[{"object":"embedding","index":0,"embedding":[...]}],
            "model":"bge-m3","usage":{"prompt_tokens":N,"total_tokens":N}}
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Union

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与检索侧保持同一套模型解析与镜像设置
from src.system.model_settings import (  # noqa: E402
    ensure_hf_mirror,
    resolve_embed_device,
    resolve_embed_model,
)

ensure_hf_mirror()


class EmbeddingBackend:
    """BGE-M3 单例后端：懒加载 + 分批编码 + L2 归一化。"""

    def __init__(self, model_name: str, device: str, batch_size: int,
                 max_seq_length: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def dimension(self) -> int:
        self._ensure()
        return int(self._model.get_sentence_embedding_dimension())

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer

        print(f"[embed] 正在加载 {self.model_name} (device={self.device}) …", flush=True)
        started = time.time()
        self._model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )
        if self.max_seq_length > 0:
            self._model.max_seq_length = self.max_seq_length
        print(
            f"[embed] 就绪，dim={self._model.get_sentence_embedding_dimension()} "
            f"max_seq={self.max_seq_length} 耗时 {time.time() - started:.1f}s",
            flush=True,
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        import faiss
        import numpy as np

        self._ensure()
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            emb = self._model.encode(
                batch, batch_size=len(batch),
                show_progress_bar=False, convert_to_numpy=True,
            )
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            # 与检索侧一致：L2 归一化后用内积等价余弦
            faiss.normalize_L2(emb)
            vectors.extend(emb.tolist())
        return vectors


class EmbeddingRequest(BaseModel):
    """必须定义在模块级。

    本文件启用了 `from __future__ import annotations`，所有注解都会变成字符串；
    FastAPI 解析 `body: EmbeddingRequest` 时要在**模块全局**里查这个名字。
    若把模型定义在 `build_app()` 内部，解析就会失败，FastAPI 会把 body 当成
    query 参数，返回 `{"detail":[{"loc":["query","body"]...}]}`。
    """

    input: Union[str, List[str]]
    model: str | None = None
    encoding_format: str | None = None


def build_app(backend: EmbeddingBackend):
    from fastapi import FastAPI

    app = FastAPI(title="Local BGE-M3 Embedding (OpenAI-compatible)")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": backend.model_name,
            "dimension": backend.dimension,
            "device": backend.device,
            "max_seq_length": backend.max_seq_length,
        }

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingRequest) -> dict:
        raw = body.input
        texts = [raw] if isinstance(raw, str) else list(raw)
        vectors = backend.encode([str(t) for t in texts])
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "model": body.model or backend.model_name,
            "usage": {
                # OpenAI 兼容端点的 usage 只用于统计，这里给字符数近似
                "prompt_tokens": sum(len(t) for t in texts),
                "total_tokens": sum(len(t) for t in texts),
            },
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 BGE-M3 embedding 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9622)
    ap.add_argument("--model", default=os.environ.get("FINANCE_EMBED_MODEL") or "")
    ap.add_argument("--device", default="")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("FINANCE_EMBED_BATCH_SIZE", "16") or 16))
    ap.add_argument("--max-seq-length", type=int,
                    default=int(os.environ.get("FINANCE_EMBED_MAX_SEQ_LEN", "512") or 512))
    ap.add_argument("--preload", action="store_true", help="启动时立即加载模型")
    args = ap.parse_args()

    model_name = args.model or resolve_embed_model()
    device = args.device or resolve_embed_device()
    backend = EmbeddingBackend(model_name, device, args.batch_size, args.max_seq_length)
    if args.preload:
        backend._ensure()

    import uvicorn

    print(f"[embed] 监听 http://{args.host}:{args.port}  "
          f"(POST /v1/embeddings, GET /health)", flush=True)
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
