"""Recent news and industry developments provider."""

import asyncio
import logging
import time
from typing import List, Optional
from urllib.parse import urlparse

from ddgs import DDGS
from tavily import AsyncTavilyClient

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.config import config
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class NewsProvider(BaseSourceProvider):
    """Searches news sources for recent developments, announcements, and market analysis."""

    def __init__(self, max_results: int = 5):
        super().__init__(
            name="news",
            source_type=SourceType.NEWS,
            max_results=max_results,
            failure_threshold=4,
            reset_timeout=45.0
        )
        self.min_delay_seconds = 1.0

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("news", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        # If Tavily key is present, try Tavily News first; fallback to DuckDuckGo News
        if config.tavily_api_key:
            try:
                results = await self._search_tavily_news(query, limit)
                if results:
                    self.circuit_breaker.record_success()
                    return results
            except Exception as e:
                logger.warning(f"[News] Tavily news failed, falling back to DuckDuckGo: {e}")

        try:
            results = await self._search_ddg_news(query, limit)
            self.circuit_breaker.record_success()
            return results
        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[News] Search error: {e}")
            return []

    async def _search_ddg_news(self, query: str, limit: int) -> List[SearchResult]:
        logger.info(f"[News] Querying DuckDuckGo News for: '{query}' (limit={limit})")
        
        def _sync_fetch():
            ddgs = DDGS()
            return list(ddgs.news(query, max_results=limit))

        raw_results = await asyncio.to_thread(_sync_fetch)
        results = []

        for item in raw_results:
            title = item.get("title", "")
            url = item.get("url") or item.get("href", "")
            body = item.get("body", "")
            date = item.get("date", "")
            source = item.get("source", "")
            
            # Domain check for credibility
            cred_score = 65.0
            factors = [f"News outlet: {source or 'Web News'}"]
            if date:
                factors.append(f"Published: {date[:10]}")

            domain = urlparse(url).netloc.lower()
            if any(dom in domain for dom in ["reuters.com", "bloomberg.com", "ft.com", "wsj.com", "bbc.com", "apnews.com"]):
                cred_score = 92.0
                factors.append("Premier financial / investigative news wire")
            elif any(dom in domain for dom in ["techcrunch.com", "theverge.com", "wired.com", "arstechnica.com"]):
                cred_score = 85.0
                factors.append("Reputable technology publication")

            results.append(SearchResult(
                query=query,
                title=title,
                url=url,
                snippet=body,
                content=f"{title}\n\nSource: {source} ({date})\n\n{body}",
                source_type=SourceType.NEWS.value,
                published_date=date[:10] if date else None,
                journal_or_venue=source or domain,
                credibility_score=cred_score,
                credibility_level="high" if cred_score >= 80 else "medium",
                credibility_factors=factors
            ))

        logger.info(f"[News] Found {len(results)} news articles")
        return results

    async def _search_tavily_news(self, query: str, limit: int) -> List[SearchResult]:
        client = AsyncTavilyClient(api_key=config.tavily_api_key)
        response = await client.search(
            query=query,
            max_results=limit,
            topic="news"
        )
        results = []
        for item in response.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            pub_date = item.get("published_date", "")

            results.append(SearchResult(
                query=query,
                title=title,
                url=url,
                snippet=content[:300],
                content=content,
                source_type=SourceType.NEWS.value,
                published_date=pub_date[:10] if pub_date else None,
                journal_or_venue="Tavily News Feed",
                credibility_score=80.0,
                credibility_level="high",
                credibility_factors=["Verified news index via Tavily"]
            ))
        return results

