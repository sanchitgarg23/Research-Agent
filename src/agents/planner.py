"""Autonomous agent responsible for planning multi-source research strategy."""

import asyncio
import logging
import re
import time
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel

from src.state import ResearchState, ResearchPlan, SearchQuery
from src.config import config
from .factory import get_llm
from src.llm_tracker import estimate_tokens
from src.exceptions import PlanningError
from src.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from src.callbacks import emit_planning_start, emit_planning_complete, emit_error

logger = logging.getLogger(__name__)


def clean_topic_str(topic: str) -> str:
    """Strip conversational fluff and return a clean, capitalized topic."""
    cleaned = topic.strip()
    lower = cleaned.lower()
    prefixes = [
        "hey can u tell me whats ", "hey can u tell me what is ", "hey can u tell me ",
        "hey can you tell me whats ", "hey can you tell me what is ", "hey can you tell me ",
        "can u tell me whats ", "can u tell me what is ", "can u tell me ",
        "can you tell me whats ", "can you tell me what is ", "can you tell me ",
        "tell me about whats ", "tell me about what is ", "tell me about ",
        "tell me whats ", "tell me what is ", "tell me ",
        "whats ", "what is ", "what are ", "who is ", "who was "
    ]
    for p in prefixes:
        if lower.startswith(p):
            cleaned = cleaned[len(p):].strip()
            break
    cleaned = cleaned.rstrip('?.! ')
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or topic


class ResearchPlanner:
    """Autonomous agent responsible for formulating multi-source research plans."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.7)
        self.max_retries = max_retries
        
    async def plan(self, state: ResearchState) -> Dict[str, Any]:
        """Create a research plan with structured LLM output.
        
        Returns dict with updates that LangGraph will merge into state.
        """
        logger.info(f"Planning research for: {state.research_topic}")
        
        await emit_planning_start(state.research_topic)
        
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_queries=config.max_search_queries,
            max_sections=config.max_report_sections
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", PLANNER_USER_TEMPLATE)
        ])
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                chain = prompt | self.llm | JsonOutputParser()
                
                input_text = f"{state.research_topic} {config.max_search_queries} {config.max_report_sections}"
                input_tokens = estimate_tokens(input_text)
                
                result = await chain.ainvoke({
                    "topic": state.research_topic,
                    "max_queries": config.max_search_queries,
                    "max_sections": config.max_report_sections
                })
                
                duration = time.time() - start_time
                output_tokens = estimate_tokens(str(result))
                
                call_detail = {
                    'agent': 'ResearchPlanner',
                    'operation': 'plan',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                if not all(key in result for key in ["topic", "objectives", "search_queries", "report_outline"]):
                    raise PlanningError("Invalid plan structure returned")
                
                if not result["search_queries"]:
                    raise PlanningError("No search queries generated")
                
                # Clean refined topic
                raw_refined = result.get("topic", state.research_topic)
                cleaned_topic = clean_topic_str(raw_refined)
                
                # Clean outline to remove duplicated numbers and handle dicts or strings
                cleaned_outline = []
                for sec in result["report_outline"][:config.max_report_sections]:
                    sec_name = sec.get("section_title") or sec.get("title") or str(sec) if isinstance(sec, dict) else str(sec)
                    sec_clean = re.sub(r'^\d+[\.\)]\s*', '', sec_name).strip()
                    if sec_clean:
                        cleaned_outline.append(sec_clean)
                if not cleaned_outline:
                    cleaned_outline = [
                        "Foundational Overview & Direct Answer",
                        "Key Analysis & Insights"
                    ][:config.max_report_sections]
                
                # Search queries with guaranteed web presence
                search_queries = [
                    SearchQuery(
                        query=sq["query"],
                        purpose=sq.get("purpose", ""),
                        target_source=sq.get("target_source", "all")
                    )
                    for sq in result["search_queries"][:config.max_search_queries]
                ]
                
                has_web = any(getattr(sq, 'target_source', '').lower() in ['web', 'all'] for sq in search_queries)
                if not has_web and search_queries:
                    search_queries[0].target_source = 'web'
                
                plan = ResearchPlan(
                    topic=cleaned_topic,
                    objectives=result["objectives"][:5],
                    search_queries=search_queries,
                    report_outline=cleaned_outline
                )
                
                logger.info(f"Created plan with {len(plan.search_queries)} queries (max: {config.max_search_queries})")
                logger.info(f"Report outline has {len(plan.report_outline)} sections (max: {config.max_report_sections})")
                
                await emit_planning_complete(len(plan.search_queries), len(plan.report_outline))
                
                return {
                    "plan": plan,
                    "current_stage": "searching",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Planning failed after {self.max_retries} attempts")
                    await emit_error(f"Planning failed: {str(e)}")
                    return {
                        "error": f"Planning failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Planning failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
