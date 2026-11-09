"""Cross-source deduplication and entity resolution for multi-source research items."""

import re
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from src.state import SearchResult

logger = logging.getLogger(__name__)


class ResultDeduplicator:
    """Intelligently identifies and merges identical publications, repositories, and articles across different sources."""

    def __init__(self, title_similarity_threshold: float = 0.88):
        self.title_similarity_threshold = title_similarity_threshold

    def deduplicate(self, results: List[SearchResult]) -> Tuple[List[SearchResult], Dict[str, int]]:
        """Deduplicate a list of SearchResult objects using DOI, arXiv ID, URL, and title similarity.
        
        Returns:
            Tuple of (deduplicated_results, stats_dict)
        """
        if not results:
            return [], {"total_raw": 0, "deduplicated": 0, "duplicates_removed": 0}

        initial_count = len(results)
        unique_results: List[SearchResult] = []

        for item in results:
            matched_idx = self._find_duplicate_index(item, unique_results)
            if matched_idx is not None:
                # Merge into existing item
                self._merge_results(unique_results[matched_idx], item)
            else:
                unique_results.append(item)

        stats = {
            "total_raw": initial_count,
            "deduplicated": len(unique_results),
            "duplicates_removed": initial_count - len(unique_results)
        }
        logger.info(f"Deduplication complete: {initial_count} raw -> {len(unique_results)} unique ({stats['duplicates_removed']} duplicates merged)")
        return unique_results, stats

    def _find_duplicate_index(self, item: SearchResult, existing_list: List[SearchResult]) -> Optional[int]:
        """Find the index of an existing result that matches the candidate item."""
        norm_doi = self._normalize_doi(item.doi)
        norm_arxiv = self._normalize_arxiv_id(item.arxiv_id) or self._extract_arxiv_from_string(item.url) or self._extract_arxiv_from_string(item.doi)
        norm_url = self._normalize_url(item.url)
        norm_title = self._normalize_title(item.title)
        words_item = set(norm_title.split())

        for idx, existing in enumerate(existing_list):
            # 1. Match by DOI
            if norm_doi and norm_doi == self._normalize_doi(existing.doi):
                return idx

            # 2. Match by arXiv ID
            existing_arxiv = self._normalize_arxiv_id(existing.arxiv_id) or self._extract_arxiv_from_string(existing.url) or self._extract_arxiv_from_string(existing.doi)
            if norm_arxiv and existing_arxiv and norm_arxiv == existing_arxiv:
                return idx

            # 3. Match by Canonical URL
            if norm_url and norm_url == self._normalize_url(existing.url):
                return idx

            # 4. Match by Title (exact, token overlap, or fuzzy similarity)
            existing_title = self._normalize_title(existing.title)
            if norm_title and existing_title and len(norm_title) > 12 and len(existing_title) > 12:
                if norm_title == existing_title:
                    return idx

                words_existing = set(existing_title.split())
                if words_item and words_existing:
                    intersection = words_item.intersection(words_existing)
                    min_words = min(len(words_item), len(words_existing))
                    # If all words of one title are contained in the other and min_words >= 3
                    if min_words >= 3 and len(intersection) == min_words:
                        return idx
                    # Token overlap Jaccard
                    jaccard = len(intersection) / len(words_item.union(words_existing))
                    if jaccard >= 0.75:
                        return idx

                similarity = SequenceMatcher(None, norm_title, existing_title).ratio()
                if similarity >= self.title_similarity_threshold:
                    return idx

        return None

    @staticmethod
    def _extract_arxiv_from_string(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        match = re.search(r'(\d{4}\.\d{4,5})', text)
        return match.group(1) if match else None

    def _merge_results(self, target: SearchResult, source: SearchResult) -> None:
        """Merge metadata from source result into target result to create the richest possible item."""
        # Merge DOI
        if not target.doi and source.doi:
            target.doi = source.doi

        # Merge arXiv ID
        if not target.arxiv_id and source.arxiv_id:
            target.arxiv_id = source.arxiv_id

        # Merge Authors
        if not target.authors and source.authors:
            target.authors = source.authors
        elif source.authors and len(source.authors) > len(target.authors):
            target.authors = source.authors

        # Merge Publication Date
        if not target.published_date and source.published_date:
            target.published_date = source.published_date

        # Merge Venue
        if not target.journal_or_venue and source.journal_or_venue:
            target.journal_or_venue = source.journal_or_venue

        # Merge Citation count (take maximum)
        if source.citation_count is not None:
            if target.citation_count is None or source.citation_count > target.citation_count:
                target.citation_count = source.citation_count

        # Merge GitHub metrics
        if source.repo_stars is not None:
            target.repo_stars = max(target.repo_stars or 0, source.repo_stars)
        if source.repo_forks is not None:
            target.repo_forks = max(target.repo_forks or 0, source.repo_forks)
        if not target.primary_language and source.primary_language:
            target.primary_language = source.primary_language

        # Merge Content (prefer longer, richer text)
        if source.content and (not target.content or len(source.content) > len(target.content)):
            target.content = source.content

        # Credibility score (take highest score and combine factors)
        if source.credibility_score is not None:
            target.credibility_score = max(target.credibility_score or 50.0, source.credibility_score)
        
        for factor in source.credibility_factors:
            if factor not in target.credibility_factors:
                target.credibility_factors.append(factor)

        target.credibility_factors.append(f"Cross-verified across multiple sources ({source.source_type})")

    @staticmethod
    def _normalize_doi(doi: Optional[str]) -> Optional[str]:
        if not doi:
            return None
        doi = doi.lower().strip()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi

    @staticmethod
    def _normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
        if not arxiv_id:
            return None
        arxiv_id = arxiv_id.lower().strip()
        # Remove version suffix like v1, v2
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        arxiv_id = re.sub(r'^arxiv:', '', arxiv_id)
        return arxiv_id

    @staticmethod
    def _normalize_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            parsed = urlparse(url.lower().strip())
            netloc = re.sub(r'^www\.', '', parsed.netloc)
            path = parsed.path.rstrip('/')
            return f"{netloc}{path}"
        except Exception:
            return url.lower().strip().rstrip('/')

    @staticmethod
    def _normalize_title(title: Optional[str]) -> str:
        if not title:
            return ""
        # Lowercase, remove brackets, punctuation, multiple spaces
        clean = title.lower()
        clean = re.sub(r'\[.*?\]', '', clean)  # remove [pdf], [cs.AI], etc.
        clean = re.sub(r'[^\w\s]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
