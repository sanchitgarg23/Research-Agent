"""Unit and integration test suite for the Multi-Source Research Engine."""

import unittest
from src.state import SearchResult, SourceType
from src.utils.dedup import ResultDeduplicator
from src.utils.credibility import CredibilityScorer
from src.utils.citations import CitationFormatter
from src.utils.sources.dispatcher import ParallelMultiSourceDispatcher


class TestMultiSourceEngine(unittest.TestCase):
    def test_search_result_schema(self):
        """Verify SearchResult accepts multi-source fields."""
        res = SearchResult(
            query="quantum algorithms",
            title="Shor's Algorithm for Quantum Computers",
            url="https://arxiv.org/abs/quant-ph/9508027",
            snippet="Fast quantum factoring...",
            source_type=SourceType.ARXIV.value,
            doi="10.1109/SFCS.1994.365700",
            arxiv_id="quant-ph/9508027",
            authors=["Peter W. Shor"],
            published_date="1995-08-01",
            journal_or_venue="IEEE FOCS",
            citation_count=8500
        )
        self.assertEqual(res.source_type, "arxiv")
        self.assertEqual(res.doi, "10.1109/SFCS.1994.365700")
        self.assertEqual(res.citation_count, 8500)
        self.assertEqual(len(res.authors), 1)

    def test_cross_source_deduplication(self):
        """Verify deduplication by DOI and title similarity merges metadata."""
        dedup = ResultDeduplicator(title_similarity_threshold=0.85)

        arxiv_item = SearchResult(
            query="deep residual learning",
            title="Deep Residual Learning for Image Recognition",
            url="https://arxiv.org/abs/1512.03385",
            snippet="Deeper neural networks are more difficult to train...",
            content="Full abstract on residual learning and shortcut connections.",
            source_type="arxiv",
            arxiv_id="1512.03385",
            authors=["Kaiming He", "Xiangyu Zhang"]
        )

        crossref_item = SearchResult(
            query="deep residual learning",
            title="Deep Residual Learning for Image Recognition (IEEE CVPR 2016)",
            url="https://doi.org/10.1109/CVPR.2016.90",
            snippet="CVPR presentation paper...",
            source_type="crossref",
            doi="10.1109/CVPR.2016.90",
            citation_count=210000,
            journal_or_venue="IEEE CVPR"
        )

        unique, stats = dedup.deduplicate([arxiv_item, crossref_item])
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].doi, "10.1109/CVPR.2016.90")
        self.assertEqual(unique[0].arxiv_id, "1512.03385")
        self.assertEqual(unique[0].citation_count, 210000)
        self.assertEqual(unique[0].journal_or_venue, "IEEE CVPR")
        self.assertTrue("crossref" in unique[0].credibility_factors[0])

    def test_credibility_scorer(self):
        """Verify multi-dimensional credibility calculation."""
        scorer = CredibilityScorer()
        
        academic_paper = SearchResult(
            query="ai",
            title="Landmark AI Study",
            url="https://nature.com/articles/s41586-020-2649-2",
            snippet="snippet",
            doi="10.1038/s41586-020-2649-2",
            citation_count=1500,
            journal_or_venue="Nature",
            source_type="crossref"
        )
        score_academic = scorer.score_result(academic_paper)
        self.assertEqual(score_academic["level"], "high")
        self.assertGreaterEqual(score_academic["score"], 90.0)

        repo = SearchResult(
            query="transformers",
            title="huggingface/transformers",
            url="https://github.com/huggingface/transformers",
            snippet="State of the art NLP",
            repo_stars=140000,
            primary_language="Python",
            source_type="github"
        )
        score_repo = scorer.score_result(repo)
        self.assertEqual(score_repo["level"], "high")
        self.assertGreaterEqual(score_repo["score"], 90.0)

    def test_citation_formatting(self):
        """Verify APA and BibTeX formatting with academic metadata."""
        cf = CitationFormatter()
        res = SearchResult(
            query="attention",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            snippet="Transformers...",
            doi="10.48550/arXiv.1706.03762",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            published_date="2017-06-12",
            journal_or_venue="NeurIPS",
            source_type="arxiv"
        )
        
        apa_ref = cf.format_references_section([res.url], style="apa", search_results=[res])
        self.assertIn("[ARXIV]", apa_ref)
        self.assertIn("Ashish Vaswani", apa_ref)
        self.assertIn("https://doi.org/10.48550/arXiv.1706.03762", apa_ref)

        bibtex_ref = cf.format_references_section([res.url], style="bibtex", search_results=[res])
        self.assertIn("@article{", bibtex_ref)
        self.assertIn("title = {Attention Is All You Need}", bibtex_ref)
        self.assertIn("doi = {10.48550/arXiv.1706.03762}", bibtex_ref)

    def test_dispatcher_initialization(self):
        """Verify dispatcher registers active providers."""
        dispatcher = ParallelMultiSourceDispatcher()
        self.assertIn("web", dispatcher.providers)
        self.assertIn("arxiv", dispatcher.providers)
        self.assertIn("semantic_scholar", dispatcher.providers)
        self.assertIn("crossref", dispatcher.providers)
        self.assertIn("github", dispatcher.providers)
        self.assertIn("news", dispatcher.providers)


if __name__ == "__main__":
    unittest.main()
