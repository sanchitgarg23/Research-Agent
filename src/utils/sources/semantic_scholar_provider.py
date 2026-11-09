"""Semantic Scholar scientific publications search provider."""

import asyncio
import logging
from typing import List, Optional

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.utils.web_utils import HTTPClientManager
from src.config import config
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class SemanticScholarProvider(BaseSourceProvider):
    """Searches Semantic Scholar for scientific literature, citations, and abstracts."""

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        super().__init__(
            name="semantic_scholar",
            source_type=SourceType.SEMANTIC_SCHOLAR,
            max_results=max_results,
            failure_threshold=3,
            reset_timeout=60.0
        )
        self.api_key = api_key or config.semantic_scholar_api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.min_delay_seconds = 1.0  # Public rate limit: ~1 request per second

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("semantic_scholar", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        headers = {
            "Accept": "application/json",
            "User-Agent": "DeepResearchAgent/2.0 (Academic Research Assistant)"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,citationCount,year,publicationDate,venue,externalIds,url,openAccessPdf"
        }

        try:
            logger.info(f"[SemanticScholar] Querying publications for: '{query}' (limit={limit})")
            client = await HTTPClientManager.get_instance().get_client()

            response = await client.get(self.base_url, params=params, headers=headers, timeout=15.0)

            if response.status_code == 429:
                self.circuit_breaker.record_failure()
                raise RateLimitError("Semantic Scholar rate limit reached", retry_after=45, service="semantic_scholar")

            if response.status_code != 200:
                self.circuit_breaker.record_failure()
                logger.warning(f"[SemanticScholar] API status {response.status_code}: {response.text[:150]}")
                return []

            data = response.json()
            papers = data.get("data", [])
            results = []

            for item in papers:
                title = item.get("title") or "Untitled Publication"
                abstract = item.get("abstract") or ""
                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                citations = item.get("citationCount")
                venue = item.get("venue") or ""
                pub_date = item.get("publicationDate") or str(item.get("year", ""))
                
                # Identifiers
                ext_ids = item.get("externalIds", {}) or {}
                doi = ext_ids.get("DOI")
                arxiv_id = ext_ids.get("ArXiv")

                # URL fallback chain
                paper_url = item.get("url") or ""
                if not paper_url and doi:
                    paper_url = f"https://doi.org/{doi}"
                elif not paper_url and arxiv_id:
                    paper_url = f"https://arxiv.org/abs/{arxiv_id}"

                snippet = abstract[:300] + ("..." if len(abstract) > 300 else "")
                if not snippet and venue:
                    snippet = f"Published in {venue} ({pub_date}). Authors: {', '.join(authors[:3])}."

                # Credibility calculation
                cred_score = 80.0
                factors = ["Indexed in Semantic Scholar academic graph"]
                if citations is not None:
                    if citations > 100:
                        cred_score = 95.0
                        factors.append(f"Highly cited paper ({citations} citations)")
                    elif citations > 10:
                        cred_score = 88.0
                        factors.append(f"Established citations ({citations} citations)")
                    else:
                        factors.append(f"Recent/emerging paper ({citations} citations)")
                if venue:
                    factors.append(f"Published in venue: {venue}")

                results.append(SearchResult(
                    query=query,
                    title=title,
                    url=paper_url,
                    snippet=snippet,
                    content=abstract,
                    source_type=SourceType.SEMANTIC_SCHOLAR.value,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    authors=authors,
                    published_date=pub_date,
                    journal_or_venue=venue or "Academic Conference/Journal",
                    citation_count=citations,
                    credibility_score=cred_score,
                    credibility_level="high",
                    credibility_factors=factors
                ))

            self.circuit_breaker.record_success()
            logger.info(f"[SemanticScholar] Found {len(results)} publications")
            return results

        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[SemanticScholar] Search error: {e}")
            return []

