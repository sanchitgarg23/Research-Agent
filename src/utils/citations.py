"""Citation formatting utilities for academic, preprint, and software citation styles."""

from typing import List, Dict, Optional, Any
from datetime import datetime
import re
import logging

from src.state import SearchResult

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Format citations in academic, preprint, and open-source software styles."""
    
    def __init__(self):
        self.styles = ['apa', 'mla', 'chicago', 'ieee', 'bibtex']
    
    def format_apa(
        self,
        url: str,
        title: str = "",
        author: str = "",
        date: str = "",
        venue: str = "",
        doi: str = ""
    ) -> str:
        """Format citation in APA 7th style."""
        author_str = author if author else "Unknown Author"
        year = date[:4] if date and len(date) >= 4 else "n.d."
        venue_str = f" *{venue}*." if venue else ""
        doi_str = f" https://doi.org/{doi}" if doi else (f" Retrieved from {url}" if url else "")
        
        return f"{author_str} ({year}). *{title}*.{venue_str}{doi_str}"
    
    def format_mla(
        self,
        url: str,
        title: str = "",
        author: str = "",
        date: str = "",
        venue: str = "",
        doi: str = ""
    ) -> str:
        """Format citation in MLA 9th style."""
        parts = []
        if author:
            parts.append(author)
        if title:
            parts.append(f'"{title}"')
        if venue:
            parts.append(f"*{venue}*")
        if date:
            parts.append(date)
        if doi:
            parts.append(f"doi:{doi}")
        elif url:
            parts.append(f"<{url}>")
        parts.append(f"Accessed {datetime.now().strftime('%d %b. %Y')}")
        return ". ".join(parts) + "."
    
    def format_chicago(
        self,
        url: str,
        title: str = "",
        author: str = "",
        date: str = "",
        venue: str = "",
        doi: str = ""
    ) -> str:
        """Format citation in Chicago style."""
        author_str = f"{author}. " if author else ""
        venue_str = f" *{venue}*" if venue else ""
        link_str = f" https://doi.org/{doi}" if doi else f" {url}"
        return f"{author_str}\"{title}.\"{venue_str} ({date or 'n.d.'}). Accessed {datetime.now().strftime('%B %d, %Y')}.{link_str}."
    
    def format_ieee(
        self,
        url: str,
        title: str = "",
        author: str = "",
        date: str = "",
        venue: str = "",
        doi: str = ""
    ) -> str:
        """Format citation in IEEE style."""
        author_str = f"{author}, " if author else ""
        venue_str = f", *{venue}*" if venue else ""
        date_str = f", {date}" if date else ""
        link_str = f", doi: {doi}" if doi else (f", [Online]. Available: {url}" if url else "")
        return f"{author_str}\"{title}\"{venue_str}{date_str}{link_str}."

    def format_bibtex(
        self,
        url: str,
        title: str = "",
        author: str = "",
        date: str = "",
        venue: str = "",
        doi: str = "",
        key: str = "ref"
    ) -> str:
        """Format citation as a BibTeX entry."""
        year = date[:4] if date and len(date) >= 4 else datetime.now().year
        entry_type = "article" if venue else "misc"
        fields = [
            f"  title = {{{title}}}",
            f"  author = {{{author or 'Anonymous'}}}",
            f"  year = {{{year}}}"
        ]
        if venue:
            fields.append(f"  journal = {{{venue}}}")
        if doi:
            fields.append(f"  doi = {{{doi}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        
        return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"
    
    def format_references_section(
        self,
        urls: List[str],
        style: str = 'apa',
        search_results: Optional[List[Any]] = None
    ) -> str:
        """Format a references section in the specified style using enriched metadata."""
        style = style.lower()
        if style not in self.styles:
            style = 'apa'
            logger.warning(f"Unknown style {style}, defaulting to APA")
        
        # Build lookup table from search results
        meta_lookup: Dict[str, Dict[str, Any]] = {}
        if search_results:
            for item in search_results:
                item_url = getattr(item, 'url', None) or (item.get('url') if isinstance(item, dict) else None)
                if item_url:
                    authors = getattr(item, 'authors', []) or (item.get('authors', []) if isinstance(item, dict) else [])
                    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else ""
                    
                    meta_lookup[item_url] = {
                        'title': getattr(item, 'title', '') or (item.get('title', '') if isinstance(item, dict) else ''),
                        'author': author_str,
                        'date': getattr(item, 'published_date', '') or (item.get('published_date', '') if isinstance(item, dict) else ''),
                        'venue': getattr(item, 'journal_or_venue', '') or (item.get('journal_or_venue', '') if isinstance(item, dict) else ''),
                        'doi': getattr(item, 'doi', '') or (item.get('doi', '') if isinstance(item, dict) else ''),
                        'source_type': getattr(item, 'source_type', 'web') or (item.get('source_type', 'web') if isinstance(item, dict) else 'web'),
                    }

        references = []
        for i, url in enumerate(urls, 1):
            m = meta_lookup.get(url, {})
            title = m.get('title') or ""
            author = m.get('author') or ""
            date = m.get('date') or ""
            venue = m.get('venue') or ""
            doi = m.get('doi') or ""
            source_type = m.get('source_type') or "web"

            # Tag badge for clarity in research reports
            badge = f"[{source_type.upper()}] " if source_type != "web" else ""

            if style == 'apa':
                citation = self.format_apa(url, title, author, date, venue, doi)
            elif style == 'mla':
                citation = self.format_mla(url, title, author, date, venue, doi)
            elif style == 'chicago':
                citation = self.format_chicago(url, title, author, date, venue, doi)
            elif style == 'ieee':
                citation = self.format_ieee(url, title, author, date, venue, doi)
            elif style == 'bibtex':
                citation = self.format_bibtex(url, title, author, date, venue, doi, key=f"ref_{i}")
            else:
                citation = self.format_apa(url, title, author, date, venue, doi)
            
            if style == 'bibtex':
                references.append(citation)
            else:
                references.append(f"{i}. {badge}{citation}")
        
        return "\n\n".join(references) if style == 'bibtex' else "\n".join(references)
    
    def update_report_citations(
        self,
        report_content: str,
        style: str = 'apa',
        search_results: Optional[List[Any]] = None
    ) -> str:
        """Update citations in a report to use specified style."""
        references_match = re.search(
            r'## References\n\n(.*?)(?=\n##|\Z)',
            report_content,
            re.DOTALL
        )
        if not references_match:
            return report_content
        
        url_pattern = r'https?://[^\s\)\>]+'
        existing_refs = references_match.group(1)
        urls = re.findall(url_pattern, existing_refs)
        
        if not urls:
            return report_content
        
        new_references = f"## References\n\n{self.format_references_section(urls, style, search_results)}"
        return re.sub(
            r'## References\n\n.*?(?=\n##|\Z)',
            new_references,
            report_content,
            flags=re.DOTALL
        )
