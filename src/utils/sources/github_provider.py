"""GitHub repository and implementation search provider."""

import asyncio
import logging
from typing import List, Optional

from src.state import SearchResult, SourceType
from src.utils.sources.base import BaseSourceProvider
from src.utils.web_utils import HTTPClientManager
from src.config import config
from src.exceptions import SearchError, RateLimitError, CircuitOpenError

logger = logging.getLogger(__name__)


class GitHubProvider(BaseSourceProvider):
    """Searches GitHub for repositories, benchmarks, and reference implementations."""

    def __init__(self, token: Optional[str] = None, max_results: int = 5):
        super().__init__(
            name="github",
            source_type=SourceType.GITHUB,
            max_results=max_results,
            failure_threshold=3,
            reset_timeout=60.0
        )
        self.token = token or config.github_token
        self.base_url = "https://api.github.com/search/repositories"
        self.min_delay_seconds = 1.0

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("github", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        await self._enforce_rate_limit()

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DeepResearchAgent/2.0"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": limit
        }

        try:
            logger.info(f"[GitHub] Querying repositories for: '{query}' (limit={limit})")
            client = await HTTPClientManager.get_instance().get_client()

            response = await client.get(self.base_url, params=params, headers=headers, timeout=15.0)

            if response.status_code == 403 and "rate limit" in response.text.lower():
                self.circuit_breaker.record_failure()
                raise RateLimitError("GitHub API rate limit exceeded", retry_after=60, service="github")

            if response.status_code != 200:
                self.circuit_breaker.record_failure()
                logger.warning(f"[GitHub] API status {response.status_code}: {response.text[:120]}")
                return []

            data = response.json()
            items = data.get("items", [])
            results = []

            for repo in items:
                full_name = repo.get("full_name", "")
                description = repo.get("description") or "No description provided."
                html_url = repo.get("html_url", "")
                stars = repo.get("stargazers_count", 0)
                forks = repo.get("forks_count", 0)
                language = repo.get("language") or "Code"
                license_info = repo.get("license", {}) or {}
                license_name = license_info.get("name", "Unknown License")
                updated_at = (repo.get("updated_at") or "")[:10]

                snippet = (
                    f"⭐ {stars:,} stars | 🍴 {forks:,} forks | Language: {language} | "
                    f"License: {license_name} | Updated: {updated_at}\n{description}"
                )

                content_summary = (
                    f"# Repository: {full_name}\n"
                    f"- Description: {description}\n"
                    f"- Primary Language: {language}\n"
                    f"- Stargazers: {stars:,}\n"
                    f"- Forks: {forks:,}\n"
                    f"- License: {license_name}\n"
                    f"- Last Updated: {updated_at}\n"
                    f"- URL: {html_url}\n"
                )

                # Credibility calculation
                cred_score = 70.0
                factors = ["Verified GitHub open-source repository"]
                if stars >= 5000:
                    cred_score = 95.0
                    factors.append(f"Major open-source ecosystem ({stars:,} stars)")
                elif stars >= 1000:
                    cred_score = 90.0
                    factors.append(f"Widely adopted codebase ({stars:,} stars)")
                elif stars >= 100:
                    cred_score = 80.0
                    factors.append(f"Active community project ({stars:,} stars)")
                else:
                    factors.append(f"Emerging implementation ({stars} stars)")

                if license_name != "Unknown License":
                    factors.append(f"Permissive license: {license_name}")

                results.append(SearchResult(
                    query=query,
                    title=f"{full_name} ({language})",
                    url=html_url,
                    snippet=snippet,
                    content=content_summary,
                    source_type=SourceType.GITHUB.value,
                    repo_stars=stars,
                    repo_forks=forks,
                    primary_language=language,
                    published_date=updated_at,
                    journal_or_venue="GitHub",
                    credibility_score=cred_score,
                    credibility_level="high" if cred_score >= 80 else "medium",
                    credibility_factors=factors
                ))

            self.circuit_breaker.record_success()
            logger.info(f"[GitHub] Retrieved {len(results)} repositories")
            return results

        except (CircuitOpenError, RateLimitError):
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"[GitHub] Search error: {e}")
            return []

