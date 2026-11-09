"""Crossref publication metadata and DOI registry provider."""

import asyncio
import logging
import re
from typing import List, Optional

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.utils.web_utils import HTTPClientManager
from src.config import config
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class CrossrefProvider(BaseSourceProvider):
    """Searches Crossref for peer-reviewed journal publications, official DOIs, and citation counts."""

    def __init__(self, mailto: Optional[str] = None, max_results: int = 5):
        super().__init__(
            name="crossref",
            source_type=SourceType.CROSSREF,
            max_results=max_results,
            failure_threshold=3,
            reset_timeout=60.0
        )
        self.mailto = mailto or config.crossref_mailto
        self.base_url = "https://api.crossref.org/works"
        self.min_delay_seconds = 0.5

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("crossref", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        headers = {
            "User-Agent": f"DeepResearchAgent/2.0 (mailto:{self.mailto})",
            "Accept": "application/json"
        }
        params = {
            "query": query,
            "rows": limit,
            "select": "DOI,title,author,container-title,publisher,published,is-referenced-by-count,abstract,URL"
        }

        try:
            logger.info(f"[Crossref] Querying peer-reviewed literature for: '{query}' (limit={limit})")
            client = await HTTPClientManager.get_instance().get_client()

            response = await client.get(self.base_url, params=params, headers=headers, timeout=15.0)

            if response.status_code == 429:
                self.circuit_breaker.record_failure()
                raise RateLimitError("Crossref rate limit reached", retry_after=30, service="crossref")

            if response.status_code != 200:
                self.circuit_breaker.record_failure()
                logger.warning(f"[Crossref] Status {response.status_code}")
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])
            results = []

            for item in items:
                # Titles is a list
                titles = item.get("title", [])
                title = titles[0] if titles else "Untitled Work"
                
                doi = item.get("DOI")
                url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
                
                # Container title (journal / conference)
                containers = item.get("container-title", [])
                venue = containers[0] if containers else item.get("publisher", "Peer-Reviewed Venue")
                
                # Citations
                citations = item.get("is-referenced-by-count", 0)
                
                # Authors
                authors = []
                for a in item.get("author", []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    full = f"{given} {family}".strip()
                    if full:
                        authors.append(full)

                # Published Date
                date_parts = item.get("published", {}).get("date-parts", [[]])[0]
                pub_date = "-".join(str(p).zfill(2) for p in date_parts) if date_parts else None

                # Clean abstract (Crossref abstracts often have JATS XML tags like <jats:p>)
                raw_abstract = item.get("abstract", "")
                abstract = re.sub(r'<[^>]+>', '', raw_abstract).strip() if raw_abstract else ""

                snippet = abstract[:300] + ("..." if len(abstract) > 300 else "")
                if not snippet:
                    snippet = f"Peer-reviewed publication in {venue} ({pub_date or 'Recent'}). Verified DOI: {doi}."

                cred_score = 90.0
                factors = ["Official Crossref registered DOI", f"Venue: {venue}"]
                if citations > 50:
                    cred_score = 98.0
                    factors.append(f"Highly referenced ({citations} citations)")
                elif citations > 0:
                    factors.append(f"Referenced ({citations} citations)")

                results.append(SearchResult(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=abstract or snippet,
                    source_type=SourceType.CROSSREF.value,
                    doi=doi,
                    authors=authors,
                    published_date=pub_date,
                    journal_or_venue=venue,
                    citation_count=citations,
                    credibility_score=cred_score,
                    credibility_level="high",
                    credibility_factors=factors
                ))

            self.circuit_breaker.record_success()
            logger.info(f"[Crossref] Retrieved {len(results)} registered works")
            return results

        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[Crossref] Search error: {e}")
            return []

