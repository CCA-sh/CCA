"""Web tools: WebFetch and WebSearch."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from cca.tools.base import Tool


def _html_to_markdown(html: str, base_url: str = "") -> str:
    """Convert HTML to markdown. Uses markdownify if available."""
    try:
        import markdownify
        return markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
    except ImportError:
        # Fallback: strip tags
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class WebFetchTool(Tool):
    name = "WebFetch"
    description = (
        "Fetch a URL and process its content. HTML is converted to markdown. "
        "The prompt parameter describes what information to extract from the page. "
        "Will fail for authenticated or private URLs. "
        "HTTP URLs are upgraded to HTTPS automatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Fully-formed URL to fetch."},
            "prompt": {
                "type": "string",
                "description": "What to extract or summarize from the fetched content.",
            },
        },
        "required": ["url", "prompt"],
    }

    def __init__(self):
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl = 900  # 15 minutes

    async def execute(self, url: str, prompt: str) -> str:
        import time

        # Upgrade to HTTPS
        if url.startswith("http://"):
            url = "https://" + url[7:]

        # Check cache
        if url in self._cache:
            ts, content = self._cache[url]
            if time.time() - ts < self._cache_ttl:
                return self._apply_prompt(content, prompt)

        try:
            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; cca-agent/0.1)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type or "text/plain" in content_type:
                    if "text/html" in content_type:
                        text = _html_to_markdown(response.text, base_url=url)
                    else:
                        text = response.text
                else:
                    text = f"[Binary content: {content_type}, {len(response.content)} bytes]"

        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} for {url}"
        except httpx.RequestError as e:
            return f"Error fetching {url}: {e}"

        self._cache[url] = (time.time(), text)

        # Truncate very large pages
        if len(text) > 100_000:
            text = text[:100_000] + "\n\n[... content truncated at 100,000 chars ...]"

        return self._apply_prompt(text, prompt)

    def _apply_prompt(self, content: str, prompt: str) -> str:
        # The LLM that called this tool will process the result directly.
        # We return the content prefixed with the prompt context so the
        # calling LLM can answer it. In a more advanced implementation this
        # would do a secondary LLM call; here we just return the content.
        return f"[Fetched content — prompt: {prompt}]\n\n{content}"


class WebSearchTool(Tool):
    name = "WebSearch"
    description = (
        "Search the web and return results. "
        "Supports allowed_domains (only include results from these) and "
        "blocked_domains (never include results from these). "
        "Use this for current events, documentation, or any information beyond "
        "training knowledge."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (minimum 2 characters).",
                "minLength": 2,
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only include results from these domains.",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Never include results from these domains.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: str | None = None,
    ):
        self.provider = provider
        self.api_key = api_key

    async def execute(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        if self.provider == "brave" and self.api_key:
            return await self._brave_search(query, allowed_domains, blocked_domains)
        elif self.provider == "serper" and self.api_key:
            return await self._serper_search(query, allowed_domains, blocked_domains)
        else:
            return await self._ddg_search(query, allowed_domains, blocked_domains)

    async def _ddg_search(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._ddg_sync(query, allowed_domains, blocked_domains),
        )

    def _ddg_sync(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return "Error: ddgs not installed. Run: pip install ddgs"

        # Add domain filters to query
        if allowed_domains:
            site_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
            query = f"({query}) ({site_filter})"
        if blocked_domains:
            for d in blocked_domains:
                query += f" -site:{d}"

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))
        except Exception as e:
            return f"Search error: {e}"

        if not results:
            return f"No results for: {query}"

        lines = []
        for r in results:
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"**{title}**\n{url}\n{body[:300]}\n")

        return "\n---\n".join(lines)

    async def _brave_search(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        params: dict[str, Any] = {"q": query, "count": 10}
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("web", {}).get("results", [])
        lines = []
        for r in results:
            url = r.get("url", "")
            if blocked_domains and any(d in url for d in (blocked_domains or [])):
                continue
            if allowed_domains and not any(d in url for d in (allowed_domains or [])):
                continue
            title = r.get("title", "")
            desc = r.get("description", "")
            lines.append(f"**{title}**\n{url}\n{desc}\n")
        return "\n---\n".join(lines) if lines else "No results."

    async def _serper_search(
        self,
        query: str,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 10},
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("organic", [])
        lines = []
        for r in results:
            url = r.get("link", "")
            if blocked_domains and any(d in url for d in (blocked_domains or [])):
                continue
            if allowed_domains and not any(d in url for d in (allowed_domains or [])):
                continue
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            lines.append(f"**{title}**\n{url}\n{snippet}\n")
        return "\n---\n".join(lines) if lines else "No results."
