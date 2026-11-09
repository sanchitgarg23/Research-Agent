"""Multi-dimensional source credibility scoring based on domain authority, academic peer review, citations, and repository reputation."""

import re
from typing import List, Dict, Any, Union
from urllib.parse import urlparse
import logging

from src.state import SearchResult

logger = logging.getLogger(__name__)


class CredibilityScorer:
    """Score sources based on domain authority, academic validation, citations, and open-source reputation."""
    
    # Trusted domains
    TRUSTED_DOMAINS = {
        # Academic institutions (global)
        '.edu', '.ac.uk', '.ac.in', '.edu.in', '.edu.au', '.ac.jp',
        
        # Government (global)
        '.gov', '.gov.uk', '.gov.au', '.gov.ca', '.gov.in', '.europa.eu',
        
        # International news organizations
        'bbc.com', 'bbc.co.uk', 'reuters.com', 'ap.org', 'apnews.com', 'npr.org',
        'theguardian.com', 'nytimes.com', 'washingtonpost.com', 'wsj.com',
        'ft.com', 'economist.com', 'bloomberg.com', 'cnbc.com',
        'cnn.com', 'aljazeera.com', 'dw.com',
        
        # Academic & Research platforms
        'arxiv.org', 'scholar.google.com', 'researchgate.net', 'semanticscholar.org',
        'crossref.org', 'doi.org', 'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'nih.gov',
        'nature.com', 'sciencedirect.com', 'springer.com', 'wiley.com', 'ieee.org',
        'acm.org', 'jstor.org', 'plos.org', 'sciencemag.org', 'cell.com',
        
        # Open source & developer authorities
        'github.com', 'gitlab.com', 'apache.org', 'python.org', 'rust-lang.org',
        
        # Tech & Science publications
        'scientificamerican.com', 'newscientist.com', 'technologyreview.com',
        'spectrum.ieee.org', 'arstechnica.com', 'wired.com', 'techcrunch.com',
        
        # Wikipedia & educational resources
        'wikipedia.org', 'britannica.com'
    }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'\.(xyz|tk|ml|ga|cf|gq)$',  # Suspicious TLDs
        r'bit\.ly|tinyurl|t\.co',    # URL shorteners
        r'blogspot|wordpress\.com',  # Personal blogs (lower credibility)
    ]
    
    def score_url(self, url: str) -> Dict[str, Any]:
        """Score a URL's credibility based on domain authority and structure.
        
        Returns:
            Dict with 'score' (0-100), 'factors', and 'level' (low/medium/high)
        """
        if not url:
            return {'score': 0, 'factors': ['No URL provided'], 'level': 'low'}
        
        score = 50.0
        factors = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check for trusted domains
            is_trusted = False
            for trusted in self.TRUSTED_DOMAINS:
                if trusted in domain:
                    score += 30.0
                    factors.append(f'Trusted authority domain: {trusted}')
                    is_trusted = True
                    break
            
            # Check for suspicious patterns
            is_suspicious = False
            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, domain):
                    score -= 25.0
                    factors.append(f'Suspicious pattern: {pattern}')
                    is_suspicious = True
                    break
            
            # HTTPS check
            if parsed.scheme == 'https':
                score += 5.0
                factors.append('Secure HTTPS protocol')
            else:
                score -= 10.0
                factors.append('Insecure HTTP protocol')
            
            # Academic paths
            if any(path in parsed.path for path in ['/papers/', '/research/', '/publications/', '/abs/', '/article/']):
                score += 10.0
                factors.append('Scholarly publication path')
            
            score = max(0.0, min(100.0, score))
            level = 'high' if score >= 75 else 'medium' if score >= 45 else 'low'
            
            return {
                'score': round(score, 1),
                'factors': factors if factors else ['Standard domain verification'],
                'level': level,
                'domain': domain
            }
            
        except Exception as e:
            logger.warning(f"Error scoring URL {url}: {e}")
            return {'score': 35.0, 'factors': ['Scoring exception'], 'level': 'low'}

    def score_result(self, result: Union[SearchResult, Dict[str, Any]]) -> Dict[str, Any]:
        """Score an individual research item incorporating academic citations, DOIs, and repo metrics."""
        url = getattr(result, 'url', '') if hasattr(result, 'url') else result.get('url', '')
        source_type = getattr(result, 'source_type', 'web') if hasattr(result, 'source_type') else result.get('source_type', 'web')
        
        # Start with domain/URL baseline
        cred = self.score_url(url)
        score = cred['score']
        factors = list(cred['factors'])

        # Multi-dimensional boosts based on source type
        doi = getattr(result, 'doi', None) if hasattr(result, 'doi') else result.get('doi')
        if doi:
            score = max(score, 88.0)
            factors.append(f"Official DOI verified: {doi}")

        citations = getattr(result, 'citation_count', None) if hasattr(result, 'citation_count') else result.get('citation_count')
        if citations is not None:
            if citations >= 500:
                score = max(score, 98.0)
                factors.append(f"Landmark publication ({citations:,} citations)")
            elif citations >= 50:
                score = max(score, 92.0)
                factors.append(f"Highly cited literature ({citations:,} citations)")
            elif citations >= 10:
                score = max(score, 85.0)
                factors.append(f"Established academic citations ({citations} citations)")

        stars = getattr(result, 'repo_stars', None) if hasattr(result, 'repo_stars') else result.get('repo_stars')
        if stars is not None:
            if stars >= 5000:
                score = max(score, 96.0)
                factors.append(f"Premier open-source repository ({stars:,} stars)")
            elif stars >= 1000:
                score = max(score, 90.0)
                factors.append(f"Popular GitHub implementation ({stars:,} stars)")
            elif stars >= 100:
                score = max(score, 80.0)
                factors.append(f"Vetted community repository ({stars:,} stars)")

        venue = getattr(result, 'journal_or_venue', None) if hasattr(result, 'journal_or_venue') else result.get('journal_or_venue')
        if venue and any(v in venue.lower() for v in ['nature', 'ieee', 'acm', 'springer', 'science', 'pnas', 'cell']):
            score = max(score, 94.0)
            factors.append(f"Prestigious publication venue: {venue}")

        # Preserve any pre-existing factors attached during provider extraction
        existing_factors = getattr(result, 'credibility_factors', []) if hasattr(result, 'credibility_factors') else result.get('credibility_factors', [])
        for ef in existing_factors:
            if ef not in factors:
                factors.append(ef)

        score = max(0.0, min(100.0, score))
        level = 'high' if score >= 75 else 'medium' if score >= 45 else 'low'

        # Update the object if it is a SearchResult
        if isinstance(result, SearchResult):
            result.credibility_score = round(score, 1)
            result.credibility_level = level
            result.credibility_factors = factors

        return {
            'score': round(score, 1),
            'factors': factors,
            'level': level,
            'source_type': source_type
        }
    
    def score_search_results(self, results: List) -> List[Dict[str, Any]]:
        """Score a list of search results and rank them by credibility."""
        scored = []
        for result in results:
            credibility = self.score_result(result)
            scored.append({
                'result': result,
                'credibility': credibility
            })
        
        # Sort by credibility score descending
        scored.sort(key=lambda x: x['credibility']['score'], reverse=True)
        return scored
    
    def filter_by_credibility(self, results: List, min_score: int = 40) -> List:
        """Filter results by minimum credibility score."""
        scored = self.score_search_results(results)
        filtered = [
            item['result'] for item in scored
            if item['credibility']['score'] >= min_score
        ]
        logger.info(f"Filtered {len(results)} -> {len(filtered)} results (min_score={min_score})")
        return filtered

    def deduplicate(self, results: List[SearchResult]) -> tuple[List[SearchResult], Dict[str, int]]:
        """Safely handle deduplicate calls if invoked on CredibilityScorer."""
        from src.utils.dedup import ResultDeduplicator
        return ResultDeduplicator().deduplicate(results)
