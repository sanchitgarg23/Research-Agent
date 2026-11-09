"""Autonomous agent responsible for executing multi-source research searches."""

import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

from src.state import ResearchState, SearchResult
from src.config import config
from .factory import get_llm
from src.utils.tools import get_research_tools
from src.utils.credibility import CredibilityScorer
from src.utils.sources.dispatcher import ParallelMultiSourceDispatcher
from src.utils.web_utils import ContentExtractor as ContentExtractorImpl
from src.llm_tracker import estimate_tokens
from src.exceptions import SearchError
from src.prompts import SEARCHER_SYSTEM_PROMPT, SEARCHER_USER_TEMPLATE
from src.callbacks import (
    emit_search_start,
    emit_multi_source_progress,
    emit_extraction_start,
    emit_extraction_complete,
    emit_error
)

logger = logging.getLogger(__name__)


class ResearchSearcher:
    """Autonomous agent responsible for executing research searches across multi-source engines."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        credibility_scorer: Optional[CredibilityScorer] = None,
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.3)
        self.tools = get_research_tools(agent_type="search")
        self.credibility_scorer = credibility_scorer or CredibilityScorer()
        self.max_retries = max_retries
        
    async def search(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously execute research searches using multi-source parallel engine or agent tools."""
        if not state.plan:
            await emit_error("No research plan available")
            return {"error": "No research plan available"}
        
        logger.info(f"Research Searcher investigating: {len(state.plan.search_queries)} planned queries")
        
        # Branch 1: Parallel Multi-Source Engine (arXiv, Semantic Scholar, Crossref, GitHub, News, Web)
        if config.enable_multi_source:
            logger.info("Executing Multi-Source Parallel Search across active research providers...")
            dispatcher = ParallelMultiSourceDispatcher()
            start_time = time.time()
            
            total_queries = len(state.plan.search_queries)
            for i, query in enumerate(state.plan.search_queries, 1):
                channel_tag = f"[{query.target_source.upper()}] " if hasattr(query, 'target_source') and query.target_source != "all" else ""
                await emit_search_start(f"{channel_tag}{query.query}", i, total_queries)
                
            raw_results, source_breakdown = await dispatcher.dispatch(state.plan.search_queries)
            
            # Emit live multi-source aggregation event
            await emit_multi_source_progress(source_breakdown, len(raw_results))
            
            # Deep content extraction for web and news sources lacking full text
            to_extract = [r for r in raw_results if (not r.content or len(r.content) < 200) and r.source_type in ["web", "news"]]
            top_to_extract = to_extract[:6]
            if top_to_extract:
                extractor = ContentExtractorImpl(timeout=10)
                for idx, r in enumerate(top_to_extract, 1):
                    await emit_extraction_start(r.url, idx, len(top_to_extract))
                    try:
                        extracted = await extractor.extract_content_async(r.url)
                        if extracted:
                            r.content = extracted
                    except Exception as ex:
                        logger.warning(f"Content extraction error for {r.url}: {ex}")
                        
            total_extracted_chars = sum(len(r.content or '') for r in raw_results)
            extracted_count = sum(1 for r in raw_results if r.content)
            await emit_extraction_complete(extracted_count, total_extracted_chars)
            
            # Score results with multi-dimensional credibility
            scored_results = self.credibility_scorer.score_search_results(raw_results)
            filtered_scored = [
                item for item in scored_results
                if item['credibility']['score'] >= config.min_credibility_score
            ]
            
            credibility_scores = [item['credibility'] for item in filtered_scored]
            sorted_results = [item['result'] for item in filtered_scored]
            
            for q in state.plan.search_queries:
                q.completed = True
                
            duration = time.time() - start_time
            call_detail = {
                'agent': 'ResearchSearcher',
                'operation': 'multi_source_parallel_search',
                'model': 'multi_source_dispatcher',
                'input_tokens': 0,
                'output_tokens': 0,
                'duration': round(duration, 2),
                'results_count': len(sorted_results),
                'original_results_count': len(raw_results),
                'source_breakdown': source_breakdown,
                'min_credibility_score': config.min_credibility_score,
                'attempt': 1
            }
            
            return {
                "search_results": sorted_results,
                "credibility_scores": credibility_scores,
                "source_breakdown": source_breakdown,
                "current_stage": "synthesizing",
                "iterations": state.iterations + 1,
                "llm_calls": state.llm_calls,
                "total_input_tokens": state.total_input_tokens,
                "total_output_tokens": state.total_output_tokens,
                "llm_call_details": state.llm_call_details + [call_detail]
            }
        
        # Branch 2: Legacy single-channel agent search fallback
        total_queries = len(state.plan.search_queries)
        for i, query in enumerate(state.plan.search_queries, 1):
            await emit_search_start(query.query, i, total_queries)
        
        max_searches = config.max_search_queries
        max_results_per_search = config.max_search_results_per_query
        expected_total_results = max_searches * max_results_per_search
        
        system_prompt = SEARCHER_SYSTEM_PROMPT.format(
            max_searches=max_searches,
            max_results_per_search=max_results_per_search,
            expected_total_results=expected_total_results
        )
        
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                objectives_text = "\n".join(f"- {obj}" for obj in state.plan.objectives)
                queries_text = "\n".join(
                    f"- {q.query} (Purpose: {q.purpose})" 
                    for q in state.plan.search_queries
                )
                
                input_message = SEARCHER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    objectives=objectives_text,
                    queries=queries_text,
                    min_sources=expected_total_results
                )
                
                input_tokens = estimate_tokens(input_message)
                
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                duration = time.time() - start_time
                
                messages = result.get('messages', [])
                output_text = ""
                if messages:
                    output_text = str(messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]))
                
                output_tokens = estimate_tokens(output_text)
                
                search_results = self._extract_results_from_messages(messages)
                
                logger.info(f"Autonomous agent collected {len(search_results)} results")
                
                total_extracted_chars = sum(
                    len(r.content) if r.content else 0 
                    for r in search_results
                )
                extracted_count = sum(1 for r in search_results if r.content)
                
                await emit_extraction_complete(extracted_count, total_extracted_chars)
                
                if not search_results:
                    await emit_error("Agent did not collect any search results")
                    raise SearchError("Agent did not collect any search results")
            
                scored_results = self.credibility_scorer.score_search_results(search_results)
                
                filtered_scored = [
                    item for item in scored_results
                    if item['credibility']['score'] >= config.min_credibility_score
                ]
                
                credibility_scores = [item['credibility'] for item in filtered_scored]
                sorted_results = [item['result'] for item in filtered_scored]
                
                logger.info(f"Filtered {len(search_results)} -> {len(sorted_results)} results (min_credibility={config.min_credibility_score})")
                
                for q in state.plan.search_queries:
                    q.completed = True
                
                call_detail = {
                    'agent': 'ResearchSearcher',
                    'operation': 'autonomous_search',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'results_count': len(sorted_results),
                    'original_results_count': len(search_results),
                    'min_credibility_score': config.min_credibility_score,
                    'attempt': attempt + 1
                }
                
                return {
                    "search_results": sorted_results,
                    "credibility_scores": credibility_scores,
                    "current_stage": "synthesizing",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Search failed after {self.max_retries} attempts")
                    await emit_error(f"Search failed: {str(e)}")
                    return {
                        "error": f"Search failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Search failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _extract_results_from_messages(self, messages: list) -> List[SearchResult]:
        """Extract search results from agent messages."""
        search_results = []
        
        for msg in messages:
            if hasattr(msg, 'name') and msg.name in ['web_search', 'multi_source_search']:
                try:
                    content = msg.content
                    if isinstance(content, str):
                        tool_results = json.loads(content)
                    else:
                        tool_results = content
                    
                    if isinstance(tool_results, list):
                        for item in tool_results:
                            if isinstance(item, dict):
                                search_results.append(SearchResult(
                                    query=item.get('query', ''),
                                    title=item.get('title', ''),
                                    url=item.get('url', ''),
                                    snippet=item.get('snippet', ''),
                                    content=item.get('content', None),
                                    source_type=item.get('source_type', 'web'),
                                    doi=item.get('doi'),
                                    arxiv_id=item.get('arxiv_id'),
                                    authors=item.get('authors', []),
                                    published_date=item.get('published_date'),
                                    journal_or_venue=item.get('journal_or_venue'),
                                    citation_count=item.get('citation_count'),
                                    repo_stars=item.get('repo_stars')
                                ))
                except Exception as e:
                    logger.warning(f"Error parsing tool result: {e}")
            
            if hasattr(msg, 'name') and msg.name == 'extract_webpage_content':
                try:
                    content = msg.content
                    if search_results and content:
                        for sr in reversed(search_results):
                            if not sr.content:
                                sr.content = content
                                break
                except Exception as e:
                    logger.warning(f"Error updating content: {e}")
        
        return search_results
