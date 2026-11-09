"""arXiv academic paper search provider."""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import quote_plus

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.utils.web_utils import HTTPClientManager
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivProvider(BaseSourceProvider):
    """Searches arXiv for academic preprints and scientific papers."""

    def __init__(self, max_results: int = 5):
        super().__init__(
            name="arxiv",
            source_type=SourceType.ARXIV,
            max_results=max_results,
            failure_threshold=3,
            reset_timeout=60.0
        )
        self.min_delay_seconds = 1.5  # Polite crawl rate for arXiv API
        self.base_url = "https://export.arxiv.org/api/query"

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("arxiv", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        try:
            logger.info(f"[arXiv] Querying papers for: '{query}' (limit={limit})")
            client = await HTTPClientManager.get_instance().get_client()

            # Clean query for arXiv API
            clean_q = re.sub(r'[^\w\s-]', '', query).strip()
            params = {
                "search_query": f"all:{clean_q}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            response = await client.get(self.base_url, params=params, timeout=15.0)

            if response.status_code == 429:
                self.circuit_breaker.record_failure()
                raise RateLimitError("arXiv API rate limit reached", retry_after=30, service="arxiv")

            if response.status_code != 200:
                self.circuit_breaker.record_failure()
                raise SearchError(f"arXiv API returned status {response.status_code}")

            results = self._parse_atom_xml(response.text, query)
            self.circuit_breaker.record_success()
            logger.info(f"[arXiv] Successfully retrieved {len(results)} papers")
            return results

        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[arXiv] Search error: {e}")
            return []

    def _parse_atom_xml(self, xml_text: str, original_query: str) -> List[SearchResult]:
        results: List[SearchResult] = []
        try:
            root = ET.fromstring(xml_text)
            entries = root.findall("atom:entry", ATOM_NS)

            for entry in entries:
                # Title
                title_elem = entry.find("atom:title", ATOM_NS)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled Paper"
                title = re.sub(r'\s+', ' ', title)

                # Abstract / Summary
                summary_elem = entry.find("atom:summary", ATOM_NS)
                abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                abstract = re.sub(r'\s+', ' ', abstract)

                # arXiv ID & URL
                id_elem = entry.find("atom:id", ATOM_NS)
                arxiv_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                
                arxiv_id = ""
                if "/abs/" in arxiv_url:
                    arxiv_id = arxiv_url.split("/abs/")[-1]

                # PDF link if present
                pdf_url = ""
                for link in entry.findall("atom:link", ATOM_NS):
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                        pdf_url = link.get("href", "")
                        break

                # Authors
                authors = []
                for author_elem in entry.findall("atom:author", ATOM_NS):
                    name_elem = author_elem.find("atom:name", ATOM_NS)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                # Published date
                pub_elem = entry.find("atom:published", ATOM_NS)
                pub_date = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else None

                # DOI if present
                doi = None
                doi_elem = entry.find("arxiv:doi", ATOM_NS)
                if doi_elem is not None and doi_elem.text:
                    doi = doi_elem.text.strip()

                snippet = abstract[:300] + ("..." if len(abstract) > 300 else "")

                results.append(SearchResult(
                    query=original_query,
                    title=title,
                    url=arxiv_url or pdf_url,
                    snippet=snippet,
                    content=abstract,  # Paper abstract is high-value primary text
                    source_type=SourceType.ARXIV.value,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    authors=authors,
                    published_date=pub_date,
                    journal_or_venue="arXiv Preprint Archive",
                    citation_count=None,
                    credibility_score=85.0,
                    credibility_level="high",
                    credibility_factors=["Academic preprint archive (arXiv.org)", "Verifiable authors and abstract"]
                ))

        except Exception as e:
            logger.error(f"[arXiv] Failed to parse XML response: {e}")

        return results

