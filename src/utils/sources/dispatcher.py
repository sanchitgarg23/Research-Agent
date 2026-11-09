"""Parallel Multi-Source Search Dispatcher with concurrency management and deduplication."""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Any

from src.state import SearchResult, SearchQuery, SourceType
from src.config import config
from src.utils.sources.base import BaseSourceProvider
from src.utils.sources.arxiv_provider import ArxivProvider
from src.utils.sources.semantic_scholar_provider import SemanticScholarProvider
from src.utils.sources.crossref_provider import CrossrefProvider
from src.utils.sources.github_provider import GitHubProvider
from src.utils.sources.news_provider import NewsProvider
from src.utils.sources.web_provider import WebProvider
from src.utils.dedup import ResultDeduplicator

logger = logging.getLogger(__name__)


class ParallelMultiSourceDispatcher:
    """Orchestrates parallel search execution across all enabled research sources."""

    def __init__(self, deduplicator: Optional[Any] = None):
        if deduplicator is not None and hasattr(deduplicator, "deduplicate"):
            self.deduplicator = deduplicator
        else:
            self.deduplicator = ResultDeduplicator()
        self.providers: Dict[str, BaseSourceProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Register active providers based on configuration toggles."""
        limit = config.max_sources_per_category

        if config.enable_web:
            self.providers["web"] = WebProvider(max_results=limit)
        if config.enable_arxiv:
            self.providers["arxiv"] = ArxivProvider(max_results=limit)
        if config.enable_semantic_scholar:
            self.providers["semantic_scholar"] = SemanticScholarProvider(max_results=limit)
        if config.enable_crossref:
            self.providers["crossref"] = CrossrefProvider(max_results=limit)
        if config.enable_github:
            self.providers["github"] = GitHubProvider(max_results=limit)
        if config.enable_news:
            self.providers["news"] = NewsProvider(max_results=limit)

        logger.info(f"Multi-Source Dispatcher initialized with {len(self.providers)} providers: {list(self.providers.keys())}")

    async def dispatch(
        self,
        queries: List[SearchQuery],
        max_results_per_source: Optional[int] = None
    ) -> Tuple[List[SearchResult], Dict[str, int]]:
        """Dispatch search queries in parallel across active research sources.
        
        Args:
            queries: List of planned search queries with target sources
            max_results_per_source: Optional override for results per source
            
        Returns:
            Tuple of (deduplicated_results, source_breakdown)
        """
        limit = max_results_per_source or config.max_sources_per_category
        semaphore = asyncio.Semaphore(config.max_parallel_searches * 2)

        tasks = []
        for sq in queries:
            target = getattr(sq, "target_source", "all").lower()
            query_text = sq.query

            # Determine which providers to query for this SearchQuery
            selected_providers = []
            if target == "all" or target not in self.providers:
                # Query all active providers
                selected_providers = list(self.providers.values())
            else:
                # Targeted source
                if target in self.providers:
                    selected_providers = [self.providers[target]]

            for provider in selected_providers:
                tasks.append(self._execute_single_source(semaphore, provider, query_text, limit))

        logger.info(f"Dispatching {len(tasks)} parallel search jobs across {len(self.providers)} research sources...")
        raw_nested_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_raw_results: List[SearchResult] = []
        for res in raw_nested_results:
            if isinstance(res, list):
                all_raw_results.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"Source query task failed with exception: {res}")

        # Deduplicate and resolve entities
        deduped_results, dedup_stats = self.deduplicator.deduplicate(all_raw_results)

        # Compute source breakdown
        source_breakdown: Dict[str, int] = {}
        for item in deduped_results:
            stype = getattr(item, "source_type", "web")
            source_breakdown[stype] = source_breakdown.get(stype, 0) + 1

        logger.info(f"Multi-Source search complete. Total: {len(deduped_results)} unique items. Breakdown: {source_breakdown}")
        return deduped_results, source_breakdown

    async def _execute_single_source(
        self,
        semaphore: asyncio.Semaphore,
        provider: BaseSourceProvider,
        query: str,
        limit: int
    ) -> List[SearchResult]:
        async with semaphore:
            try:
                return await provider.search(query, max_results=limit)
            except Exception as e:
                logger.warning(f"Provider '{provider.name}' failed query '{query}': {e}")
                return []

