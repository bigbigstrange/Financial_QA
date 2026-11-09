"""Tavily Search adapter. Only the user's question leaves this process."""

from __future__ import annotations

import ipaddress
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def _public_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username or parsed.password
                or host == "localhost"
                or host.endswith((".localhost", ".local", ".internal"))):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            if "." not in host:
                return None
        return value
    except ValueError:
        return None


class TavilySearch:
    """Search public pages via Tavily's POST /search API."""

    def __init__(self, api_key: str, *, timeout: float = 8.0,
                 allowed_domains: tuple[str, ...] = ()):
        self.api_key = api_key
        self.timeout = timeout
        self.allowed_domains = tuple(domain.lower().strip().lstrip(".")
                                     for domain in allowed_domains if domain.strip())

    @classmethod
    def from_env(cls) -> TavilySearch | None:
        key = (os.getenv("TAVILY_API_KEY") or "").strip()
        domains = tuple(part.strip() for part in
                        (os.getenv("FINANCE_WEB_ALLOWED_DOMAINS") or "").split(",")
                        if part.strip())
        if (os.getenv("APP_ENV") or "").strip().lower() == "production" and not domains:
            return None
        return cls(key, allowed_domains=domains) if key else None

    def _domain_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        host = (urlsplit(url).hostname or "").lower()
        return any(host == domain or host.endswith("." + domain)
                   for domain in self.allowed_domains)

    def search(self, query: str) -> list[dict]:
        if not self.api_key:
            return []
        body = json.dumps({
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": False,
            "safe_search": True,
        }).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_000_001)
        except HTTPError as exc:
            raise RuntimeError(f"Tavily HTTP {exc.code}") from None
        except URLError as exc:
            raise RuntimeError("Tavily search unavailable") from exc
        if len(raw) > 1_000_000:
            raise ValueError("Tavily response too large")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Invalid Tavily response")

        items = []
        seen = set()
        for result in payload["results"][:5]:
            if not isinstance(result, dict):
                continue
            url = _public_url(result.get("url"))
            content = str(result.get("content") or "").strip()[:1200]
            if not url or not self._domain_allowed(url) or not content or url in seen:
                continue
            seen.add(url)
            items.append({
                "chunk_id": f"web:{len(items) + 1}",
                "source_doc": str(result.get("title") or url)[:200],
                "source_url": url,
                "content": content,
                "type": "web",
                "retrieval_channels": ["web"],
            })
        return items
