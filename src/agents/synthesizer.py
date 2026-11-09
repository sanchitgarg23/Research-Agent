"""Autonomous agent responsible for synthesizing research findings."""

import asyncio
import json
import logging
import re
import time
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.state import ResearchState
from src.config import config
from .factory import get_llm
from src.utils.tools import get_research_tools
from src.llm_tracker import estimate_tokens
from src.exceptions import SynthesisError
from src.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE
from src.callbacks import emit_synthesis_start, emit_synthesis_complete, emit_error

logger = logging.getLogger(__name__)


class ResearchSynthesizer:
    """Autonomous agent responsible for synthesizing research findings across diverse sources."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.3, model_override=config.summarization_model)
        self.tools = get_research_tools(agent_type="synthesis")
        self.max_retries = max_retries
        
    async def synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously synthesize key findings using direct factual extraction."""
        logger.info(f"Synthesizing findings from {len(state.search_results)} results")
        
        if not state.search_results:
            await emit_error("No search results to synthesize")
            return {"error": "No search results to synthesize"}
        
        await emit_synthesis_start(len(state.search_results))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIZER_SYSTEM_PROMPT),
            ("human", SYNTHESIZER_USER_TEMPLATE)
        ])
        chain = prompt | self.llm | StrOutputParser()
        
        max_results = 20
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                current_max = max(5, max_results - (attempt * 5))
                
                results_to_use = state.search_results[:current_max]
                credibility_scores_to_use = state.credibility_scores[:current_max] if state.credibility_scores else []
                
                results_text = self._format_results_text(results_to_use, credibility_scores_to_use)
                
                input_tokens = estimate_tokens(results_text)
                
                output_text = await chain.ainvoke({
                    "topic": state.research_topic,
                    "results": results_text
                })
                
                duration = time.time() - start_time
                output_tokens = estimate_tokens(output_text)
                
                call_detail = {
                    'agent': 'ResearchSynthesizer',
                    'operation': 'direct_synthesis',
                    'model': config.summarization_model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                key_findings = self._extract_findings(output_text, state.search_results)
                
                logger.info(f"Extracted {len(key_findings)} key findings")
                await emit_synthesis_complete(len(key_findings))
                
                return {
                    "key_findings": key_findings,
                    "current_stage": "reporting",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Synthesis failed after {self.max_retries} attempts")
                    await emit_error(f"Synthesis failed: {str(e)}")
                    return {
                        "error": f"Synthesis failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Synthesis failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _format_results_text(self, results: list, credibility_scores: list) -> str:
        """Format search results with multi-source metadata and credibility information."""
        formatted_items = []
        for i, r in enumerate(results):
            cred = credibility_scores[i] if i < len(credibility_scores) else {}
            stype = getattr(r, 'source_type', 'web').upper()
            title = getattr(r, 'title', 'Untitled')
            url = getattr(r, 'url', '')
            snippet = getattr(r, 'snippet', '')
            content = getattr(r, 'content', '') or ''
            
            meta_parts = [f"Source Channel: {stype}"]
            if getattr(r, 'doi', None):
                meta_parts.append(f"DOI: {r.doi}")
            if getattr(r, 'arxiv_id', None):
                meta_parts.append(f"arXiv ID: {r.arxiv_id}")
            if getattr(r, 'citation_count', None) is not None:
                meta_parts.append(f"Citations: {r.citation_count:,}")
            if getattr(r, 'repo_stars', None) is not None:
                meta_parts.append(f"GitHub Stars: {r.repo_stars:,} (Language: {r.primary_language or 'Code'})")
            if getattr(r, 'journal_or_venue', None):
                meta_parts.append(f"Venue: {r.journal_or_venue}")
            if getattr(r, 'published_date', None):
                meta_parts.append(f"Date: {r.published_date}")
            if getattr(r, 'authors', None):
                meta_parts.append(f"Authors: {', '.join(r.authors[:3])}")

            meta_str = " | ".join(meta_parts)
            cred_str = f"Credibility: {cred.get('level', 'unknown').upper()} (Score: {cred.get('score', 'N/A')}/100)"
            factors_str = f"Factors: {', '.join(cred.get('factors', []))}" if cred.get('factors') else ""

            item_text = (
                f"[{i+1}] {title}\n"
                f"URL: {url}\n"
                f"{meta_str}\n"
                f"{cred_str}{' - ' + factors_str if factors_str else ''}\n"
                f"Snippet: {snippet}\n" +
                (f"Abstract / Content: {content[:350]}..." if content else "")
            )
            formatted_items.append(item_text)
            
        return "\n\n".join(formatted_items)
    
    def _extract_findings(self, output_text: str, search_results: list) -> List[str]:
        """Extract key findings from synthesis output with negative refusal and citation filtering."""
        def is_valid_finding(text: str) -> bool:
            clean = text.strip()
            if len(clean) < 20:
                return False
            t = clean.lower()
            negative_signals = [
                "no specific, actionable insights",
                "no specific actionable insights",
                "no findings can be provided",
                "no json array",
                "unable to provide",
                "no relevant information",
                "i cannot provide",
                "as an ai",
                "not found in the provided sources",
                "means something in",
                "refer to the descriptions",
                "refer to the detailed",
                "refer to this page",
                "refer to descriptions",
                "on this page",
                "descriptions on this page"
            ]
            return not any(neg in t for neg in negative_signals)
        
        key_findings = []
        
        # Try finding JSON string arrays
        json_matches = re.findall(r'\[[\s\S]*?\]', output_text)
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    items = [str(x) for x in parsed if is_valid_finding(str(x))]
                    if items:
                        key_findings.extend(items)
                        break
            except Exception:
                continue
        
        # Fallback to line-by-line extraction
        if len(key_findings) < 3:
            lines = output_text.split('\n')
            for line in lines:
                line = line.strip().lstrip('-').lstrip('*').lstrip('>').strip()
                line = re.sub(r'^\d+\.\s*', '', line)
                if is_valid_finding(line) and not line.startswith('[') and not line.startswith(']'):
                    if line not in key_findings:
                        key_findings.append(line)
        
        # Fallback to search results extraction if agent output was sparse
        if len(key_findings) < 3 and search_results:
            logger.warning("Agent produced few findings, extracting substantive ones from search results")
            for r in search_results:
                text = r.snippet or r.content or ""
                if text and len(text.strip()) > 30:
                    clean_text = text.strip().replace('\n', ' ')[:250]
                    finding_cand = f"{r.title}: {clean_text}"
                    if is_valid_finding(finding_cand) and finding_cand not in key_findings:
                        key_findings.append(finding_cand)
                if len(key_findings) >= 10:
                    break
        
        return key_findings[:15]
