import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from graph.graph import app
from graph.runtime import DEFAULT_TIMEOUT_SECONDS, RECURSION_LIMIT

server = FastAPI(title="Agentic RAG Chat")
executor = ThreadPoolExecutor(max_workers=2)


@server.get("/")
async def root():
    html_path = Path(__file__).parent / "frontend.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _make_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return str(obj)


@server.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        async def _empty():
            yield f"data: {json.dumps({'node': 'done'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        deadline = time.monotonic() + DEFAULT_TIMEOUT_SECONDS
        try:
            for output in app.stream({"question": question}, {"recursion_limit": RECURSION_LIMIT}):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"graph execution exceeded {DEFAULT_TIMEOUT_SECONDS}s"
                    )
                node_name = list(output.keys())[0]
                node_data = output[node_name]
                serialized = {}
                for k, v in node_data.items():
                    if k == "documents":
                        serialized[k] = [
                            {"content": d.page_content, "source": d.metadata.get("source", "")}
                            for d in v
                        ]
                    else:
                        serialized[k] = _make_serializable(v)
                payload = json.dumps(
                    {"node": node_name, "data": serialized}, ensure_ascii=False
                )
                asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(json.dumps({"node": "error", "data": str(e)}, ensure_ascii=False)),
                loop,
            )
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    executor.submit(_run)

    async def _stream():
        while True:
            chunk = await queue.get()
            if chunk is None:
                yield f"data: {json.dumps({'node': 'done'}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {chunk}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8000)

