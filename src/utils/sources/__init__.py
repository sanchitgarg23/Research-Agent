"""Multi-source research provider suite."""

from src.utils.sources.base import BaseSourceProvider
from src.utils.sources.arxiv_provider import ArxivProvider
from src.utils.sources.semantic_scholar_provider import SemanticScholarProvider
from src.utils.sources.crossref_provider import CrossrefProvider
from src.utils.sources.github_provider import GitHubProvider
from src.utils.sources.news_provider import NewsProvider
from src.utils.sources.web_provider import WebProvider

__all__ = [
    "BaseSourceProvider",
    "ArxivProvider",
    "SemanticScholarProvider",
    "CrossrefProvider",
    "GitHubProvider",
    "NewsProvider",
    "WebProvider"
]

