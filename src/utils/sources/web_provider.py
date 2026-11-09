"""General web search provider supporting DuckDuckGo and Tavily."""

import asyncio
import logging
import re
from typing import List, Optional

import httpx
from ddgs import DDGS
from tavily import AsyncTavilyClient

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.config import config
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class WebProvider(BaseSourceProvider):
    """General web search provider with automatic Tavily / DuckDuckGo switching."""

    def __init__(self, provider_type: Optional[str] = None, max_results: int = 5):
        super().__init__(
            name="web",
            source_type=SourceType.WEB,
            max_results=max_results,
            failure_threshold=4,
            reset_timeout=45.0
        )
        self.provider_type = provider_type or config.search_provider
        self.min_delay_seconds = 1.0

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("web", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        # Try Tavily if configured
        if self.provider_type == "tavily" and config.tavily_api_key:
            try:
                results = await self._search_tavily(query, limit)
                self.circuit_breaker.record_success()
                return results
            except Exception as e:
                logger.warning(f"[Web] Tavily failed, falling back to DuckDuckGo: {e}")

        # DuckDuckGo search
        try:
            results = await self._search_ddg(query, limit)
            self.circuit_breaker.record_success()
            return results
        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[Web] DuckDuckGo search error: {e}")
            return []

    async def _search_ddg(self, query: str, limit: int) -> List[SearchResult]:
        logger.info(f"[Web] Searching DuckDuckGo for: '{query}' (limit={limit})")

        def _sync_search():
            try:
                ddgs = DDGS()
                res = list(ddgs.text(query, max_results=limit))
                if res:
                    return res
                # Try with simplified query if long
                clean_q = re.sub(r'(?i)\b(meaning of|definition of|whats|what is|term|historical context|modern usage|implications)\b', '', query).strip()
                if clean_q and clean_q != query:
                    return list(ddgs.text(clean_q, max_results=limit))
                return []
            except Exception as e:
                logger.warning(f"DDG search error: {e}")
                return []

        raw_results = await asyncio.to_thread(_sync_search)
        results = []

        for item in raw_results:
            results.append(SearchResult(
                query=query,
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                source_type=SourceType.WEB.value
            ))

        # If standard DDG returned no results (rate limit / timeouts), use guaranteed encyclopedia fallback
        if not results:
            logger.info(f"[Web] DDG returned 0 results for '{query}', invoking encyclopedia fallback...")
            fallback_items = await self._fallback_encyclopedia_search(query, limit)
            results.extend(fallback_items)

        logger.info(f"[Web] Found {len(results)} web results")
        return results

    async def _fallback_encyclopedia_search(self, query: str, limit: int) -> List[SearchResult]:
        """Guaranteed fallback using Wikipedia Full-Text Search API, Page Summaries, and DDG Instant Answers."""
        results: List[SearchResult] = []
        headers = {'User-Agent': 'ApexDeepResearchBot/1.0 (academic research; contact@apex-research.org)'}

        words = [w for w in re.findall(r'[a-zA-Z0-9]+', query) if w.lower() not in {
            "meaning", "of", "the", "whats", "what", "is", "in", "and", "or", "for", "tell", "me", "about",
            "term", "context", "history", "historical", "cultural", "modern", "usage", "roots", "origins",
            "implications", "overview", "definition"
        }]
        core_keyword = words[0] if words else (query.split()[0] if query.split() else query)

        async with httpx.AsyncClient(headers=headers, timeout=6.0) as client:
            # 1. DuckDuckGo Instant Answer API with query and core_keyword
            for q_try in [query, core_keyword]:
                if results:
                    break
                try:
                    ddg_url = "https://api.duckduckgo.com/"
                    r = await client.get(ddg_url, params={"q": q_try, "format": "json"})
                    if r.status_code == 200:
                        data = r.json()
                        abstract = data.get("AbstractText", "")
                        if abstract:
                            results.append(SearchResult(
                                query=query,
                                title=f"{data.get('Heading', q_try)} - Reference",
                                url=data.get("AbstractURL") or f"https://duckduckgo.com/?q={q_try}",
                                snippet=abstract,
                                source_type=SourceType.WEB.value
                            ))
                except Exception as ex:
                    logger.debug(f"DDG instant answer error: {ex}")

            # 2. Wikipedia Full-Text Search API (action=query&list=search)
            search_queries_to_try = [query]
            if core_keyword and core_keyword != query:
                search_queries_to_try.append(core_keyword)

            for sq in search_queries_to_try:
                if len(results) >= limit:
                    break
                try:
                    wp_api = "https://en.wikipedia.org/w/api.php"
                    params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": sq,
                        "utf8": "1",
                        "format": "json",
                        "srlimit": limit
                    }
                    res = await client.get(wp_api, params=params)
                    if res.status_code == 200:
                        data = res.json()
                        search_items = data.get("query", {}).get("search", [])
                        for item in search_items:
                            title = item.get("title", "")
                            if not title:
                                continue
                            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                            if any(r.url == page_url for r in results):
                                continue

                            s_res = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
                            extract = ""
                            if s_res.status_code == 200:
                                extract = s_res.json().get("extract", "")
                            if not extract:
                                raw_snippet = item.get("snippet", "")
                                extract = re.sub(r'<[^>]+>', '', raw_snippet).strip()

                            if extract:
                                results.append(SearchResult(
                                    query=query,
                                    title=f"{title} - Wikipedia",
                                    url=page_url,
                                    snippet=extract,
                                    source_type=SourceType.WEB.value
                                ))

                            if len(results) >= limit:
                                break
                except Exception as ex:
                    logger.debug(f"Wikipedia full-text search error: {ex}")

            # 3. Wiktionary definition fallback for core keyword if still empty
            if not results and core_keyword:
                try:
                    wikt_url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{core_keyword.lower()}"
                    w_res = await client.get(wikt_url)
                    if w_res.status_code == 200:
                        wdata = w_res.json()
                        for entry in wdata.get("en", []):
                            for d in entry.get("definitions", []):
                                clean_def = re.sub(r'<[^>]+>', '', d.get("definition", "")).strip()
                                if clean_def:
                                    results.append(SearchResult(
                                        query=query,
                                        title=f"{core_keyword} - Wiktionary",
                                        url=f"https://en.wiktionary.org/wiki/{core_keyword.lower()}",
                                        snippet=clean_def,
                                        source_type=SourceType.WEB.value
                                    ))
                                    if len(results) >= limit:
                                        break
                except Exception as ex:
                    logger.debug(f"Wiktionary error: {ex}")

        return results

    async def _search_tavily(self, query: str, limit: int) -> List[SearchResult]:
        logger.info(f"[Web] Searching Tavily for: '{query}' (limit={limit})")
        client = AsyncTavilyClient(api_key=config.tavily_api_key)
        response = await client.search(
            query=query,
            max_results=limit,
            search_depth="basic"
        )
        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                query=query,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source_type=SourceType.WEB.value
            ))
        logger.info(f"[Web] Found {len(results)} Tavily web results")
        return results

