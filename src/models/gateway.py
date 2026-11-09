from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Protocol

import httpx

from src.contracts import ContractError, ErrorCode, ModelSnapshot


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    requested_model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    response_format: dict[str, Any] | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    requested_model: str
    actual_model: str
    provider: str
    request_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    finish_reason: str | None = None


@dataclass(frozen=True)
class ModelChunk:
    text: str
    index: int
    request_id: str | None = None


class ModelTransport(Protocol):
    provider: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> Iterator[ModelChunk]: ...


class ModelGateway:
    def __init__(
        self,
        snapshot: ModelSnapshot,
        transport: ModelTransport,
        *,
        max_retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.snapshot = snapshot
        self.transport = transport
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> ModelResponse:
        request = self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            request_id=request_id,
        )
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport.complete(request)
                return replace(
                    response,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except ContractError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
            except (TimeoutError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    raise ContractError(
                        ErrorCode.MODEL_TIMEOUT,
                        "模型请求超时",
                        retryable=True,
                    ) from exc
            except Exception as exc:
                raise ContractError(
                    ErrorCode.MODEL_UNAVAILABLE,
                    "模型服务暂不可用",
                    retryable=False,
                ) from exc
            self._sleep(0.1 * (attempt + 1))
        raise AssertionError("unreachable")

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        response = self.generate(messages, **kwargs)
        text = response.text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced:
            text = fenced.group(1)
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractError(
                ErrorCode.MODEL_RESPONSE_INVALID,
                "模型未返回有效 JSON",
            ) from exc

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[ModelChunk]:
        request = self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            extra_body=extra_body,
        )
        try:
            yield from self.transport.stream(request)
        except ContractError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ContractError(
                ErrorCode.MODEL_TIMEOUT,
                "模型流式请求超时",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise ContractError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型流式连接中断",
                retryable=True,
            ) from exc

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
        request_id: str | None = None,
    ) -> ModelRequest:
        model = self.snapshot.requested_model or self.snapshot.actual_model
        if not model:
            raise ContractError(
                ErrorCode.INVALID_REQUEST,
                "模型快照未指定 requested_model",
            )
        return ModelRequest(
            messages=list(messages),
            requested_model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format=response_format,
            request_id=request_id or uuid.uuid4().hex,
            extra_body=dict(extra_body or {}),
        )


class OpenAICompatibleTransport:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        provider: str,
        timeout: float = 120.0,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.timeout = float(timeout)
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=self.timeout)
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, api_key='<redacted>')"
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        data, headers = self._post(request, stream=False)
        try:
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("content is not text")
        except (KeyError, IndexError, TypeError) as exc:
            raise ContractError(
                ErrorCode.MODEL_RESPONSE_INVALID,
                "模型响应缺少文本内容",
            ) from exc
        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        return ModelResponse(
            text=text.strip(),
            requested_model=request.requested_model,
            actual_model=str(data.get("model") or ("" if self.provider == "ark" else request.requested_model)),
            provider=self.provider,
            request_id=str(
                data.get("id")
                or headers.get("x-request-id")
                or request.request_id
            ),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    def stream(self, request: ModelRequest) -> Iterator[ModelChunk]:
        self._require_key()
        payload = self._payload(request, stream=True)
        headers = self._headers(request)
        try:
            with self._client_factory() as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    self._raise_for_status(response)
                    index = 0
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_text = line[5:].strip()
                        if data_text == "[DONE]":
                            return
                        try:
                            data = json.loads(data_text)
                            text = data["choices"][0]["delta"].get("content") or ""
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                            raise ContractError(
                                ErrorCode.MODEL_RESPONSE_INVALID,
                                "模型流包含无效事件",
                            ) from exc
                        if text:
                            yield ModelChunk(
                                text=text,
                                index=index,
                                request_id=str(data.get("id") or request.request_id),
                            )
                            index += 1
        except httpx.TimeoutException as exc:
            raise ContractError(
                ErrorCode.MODEL_TIMEOUT,
                "模型请求超时",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ContractError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型服务请求失败",
                retryable=True,
            ) from exc

    def _post(
        self,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        self._require_key()
        try:
            with self._client_factory() as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(request),
                    json=self._payload(request, stream=stream),
                )
                self._raise_for_status(response)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise ContractError(
                        ErrorCode.MODEL_RESPONSE_INVALID,
                        "模型服务返回了非 JSON 响应",
                    ) from exc
                if not isinstance(data, dict):
                    raise ContractError(
                        ErrorCode.MODEL_RESPONSE_INVALID,
                        "模型服务返回了无效 JSON 结构",
                    )
                return data, response.headers
        except httpx.TimeoutException as exc:
            raise ContractError(
                ErrorCode.MODEL_TIMEOUT,
                "模型请求超时",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ContractError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型服务请求失败",
                retryable=True,
            ) from exc

    def _require_key(self) -> None:
        if not self.api_key:
            raise ContractError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型服务未配置凭据",
            )

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "ark":
            # Cloud trace correlation only; this header is not an idempotency key.
            headers["X-Client-Request-Id"] = request.request_id
        return headers

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code >= 500 or (
                exc.response.status_code == 429 and self.provider != "ark"
            )
            raise ContractError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"模型服务返回 HTTP {exc.response.status_code}",
                retryable=retryable,
            ) from exc

    @staticmethod
    def _payload(request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.requested_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        payload.update(request.extra_body)
        return payload


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
