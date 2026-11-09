"""Autonomous agent responsible for writing research reports."""

import asyncio
import logging
import re
import time
from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from src.state import ResearchState, ReportSection
from src.config import config
from .factory import get_llm
from src.utils.tools import get_research_tools
from src.utils.citations import CitationFormatter
from src.llm_tracker import estimate_tokens
from src.exceptions import ReportGenerationError
from src.prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
from src.callbacks import emit_writing_start, emit_writing_section, emit_writing_complete, emit_error

logger = logging.getLogger(__name__)


class ReportWriter:
    """Autonomous agent responsible for compiling structured, publication-grade research reports."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        citation_style: str = 'apa',
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.7)
        self.tools = get_research_tools(agent_type="writing")
        self.max_retries = max_retries
        self.citation_style = citation_style
        self.citation_formatter = citation_formatter or CitationFormatter()
        
    async def write_report(self, state: ResearchState) -> Dict[str, Any]:
        """Write the final research report with direct Google-style answer and non-repetitive deep dive."""
        logger.info("Writing final report")
        
        if not state.plan or not state.key_findings:
            await emit_error("Insufficient data for report generation")
            return {"error": "Insufficient data for report generation"}
        
        total_sections = len(state.plan.report_outline)
        await emit_writing_start(total_sections)
        
        sources_context = ""
        if state.search_results:
            sources_context = "\nAvailable Sources for Citation:\n" + "\n".join(
                f"[{i+1}] {r.title} ({r.url})"
                for i, r in enumerate(state.search_results[:20])
            )
        
        sections_outline = "\n".join(f"- {sec}" for sec in state.plan.report_outline)
        findings_text = "\n".join(f"- {f}" for f in state.key_findings)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM_PROMPT),
            ("human", WRITER_USER_TEMPLATE)
        ])
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                input_message = WRITER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    sections_outline=sections_outline,
                    findings=findings_text,
                    sources_context=sources_context
                )
                
                input_tokens = estimate_tokens(input_message)
                chain = prompt | self.llm | StrOutputParser()
                raw_content = await chain.ainvoke({
                    "topic": state.research_topic,
                    "sections_outline": sections_outline,
                    "findings": findings_text,
                    "sources_context": sources_context
                })
                
                duration = time.time() - start_time
                output_tokens = estimate_tokens(raw_content)
                
                # Sanitize fences
                raw_content = raw_content.strip()
                if raw_content.startswith("```markdown"):
                    raw_content = raw_content[len("```markdown"):].strip()
                elif raw_content.startswith("```md"):
                    raw_content = raw_content[len("```md"):].strip()
                elif raw_content.startswith("```"):
                    raw_content = raw_content[3:].strip()
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3].strip()
                
                # Split into Quick Answer and Sections
                parts = re.split(r'\n(?=##\s+)', raw_content.strip())
                quick_answer = ""
                section_blocks = []
                
                if parts and not parts[0].strip().startswith("##"):
                    quick_answer = parts[0].strip()
                    section_blocks = parts[1:]
                else:
                    section_blocks = parts
                
                report_sections = []
                for b in section_blocks:
                    lines = b.strip().split("\n", 1)
                    title = re.sub(r'^##\s*', '', lines[0]).strip()
                    title = re.sub(r'^\d+[\.\)]\s*', '', title).strip()
                    body = lines[1].strip() if len(lines) > 1 else ""
                    
                    # Extract citations
                    citations = re.findall(r'\[(\d+)\]', body)
                    source_urls = []
                    for cite_num in set(citations):
                        idx = int(cite_num) - 1
                        if 0 <= idx < len(state.search_results):
                            source_urls.append(state.search_results[idx].url)
                    
                    report_sections.append(ReportSection(
                        title=title,
                        content=body,
                        sources=source_urls
                    ))
                
                if not report_sections:
                    report_sections = [
                        ReportSection(
                            title=state.plan.report_outline[0] if state.plan.report_outline else "Detailed Analysis",
                            content=raw_content,
                            sources=[r.url for r in state.search_results[:5]]
                        )
                    ]
                
                temp_state = ResearchState(
                    research_topic=state.research_topic,
                    plan=state.plan,
                    report_sections=report_sections,
                    search_results=state.search_results,
                    source_breakdown=getattr(state, 'source_breakdown', {})
                )
                
                final_report = self._compile_report(temp_state, quick_answer=quick_answer)
                
                if state.search_results:
                    final_report = self.citation_formatter.update_report_citations(
                        final_report,
                        style=self.citation_style,
                        search_results=state.search_results
                    )
                
                if len(final_report) < 150:
                    raise ReportGenerationError("Report too short - insufficient content")
                
                logger.info(f"Report generation complete: {len(final_report)} chars")
                await emit_writing_complete(len(final_report))
                
                call_detail = {
                    'agent': 'ReportWriter',
                    'operation': 'unified_report_generation',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                return {
                    "report_sections": report_sections,
                    "final_report": final_report,
                    "current_stage": "complete",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Report attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Report generation failed after {self.max_retries} attempts")
                    await emit_error(f"Report generation failed: {str(e)}")
                    return {
                        "error": f"Report writing failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Report generation failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _compile_report(self, state: ResearchState, quick_answer: str = "") -> str:
        """Compile final report with Google-style direct answer and non-repetitive deep dive."""
        search_results = getattr(state, 'search_results', []) or []
        report_sections = getattr(state, 'report_sections', []) or []
        
        # Determine clean, professional display title
        raw_topic = state.research_topic.strip()
        plan_topic = state.plan.topic.strip() if state.plan and hasattr(state.plan, 'topic') and state.plan.topic else raw_topic
        
        from src.agents.planner import clean_topic_str
        display_title = clean_topic_str(plan_topic)

        report_parts = [
            f"# {display_title}\n\n",
        ]
        
        # Google-Style Direct Answer Box
        if not quick_answer or len(quick_answer.strip()) < 15:
            if hasattr(state, 'key_findings') and state.key_findings:
                quick_answer = state.key_findings[0]
            elif search_results and hasattr(search_results[0], 'snippet') and search_results[0].snippet:
                quick_answer = search_results[0].snippet

        if quick_answer:
            clean_qa = quick_answer.strip()
            clean_qa = re.sub(r'^(#+\s*)*(>+\s*)*(🔍\s*)*(\*\*)?Quick\s*Answer:?(\*\*)?\s*', '', clean_qa, flags=re.IGNORECASE).strip()
            clean_qa = re.sub(r'^(#+\s*)*(>+\s*)*(🔍\s*)*(\*\*)?Quick\s*Answer:?(\*\*)?\s*', '', clean_qa, flags=re.IGNORECASE).strip()
            clean_qa = f"> 🔍 **Quick Answer**: {clean_qa}"
            report_parts.append(f"{clean_qa}\n\n---\n\n")

        # Compile Sections without duplicate titles or meta-boilerplate
        has_references_section = False
        for section_idx, section in enumerate(report_sections, 1):
            content = section.content.strip()
            
            # Strip code block fences if present
            if content.startswith("```markdown"):
                content = content[len("```markdown"):].strip()
            elif content.startswith("```md"):
                content = content[len("```md"):].strip()
            elif content.startswith("```"):
                content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            
            # Strip meta boilerplate artifacts
            content = re.sub(r'(?i)#+\s*Executive\s*Summary.*?(?=(#+|\Z))', '', content, flags=re.DOTALL)
            content = re.sub(r'(?i)#+\s*Research\s*Objectives.*?(?=(#+|\Z))', '', content, flags=re.DOTALL)
            content = re.sub(r'(?i)^\s*###?\s*Subtopics?[:\s]*.*?\n+', '', content)
            content = re.sub(r'(?i)^\s*(\*\*|__)?Subtopics?[:\s]*.*?(\*\*|__)?\n+', '', content)
            content = re.sub(r'(?i)^\s*###?\s*Key\s*Findings.*?[:\s]*\n+', '', content)
            content = re.sub(r'(?i)^\s*###?\s*Verified\s*Facts.*?[:\s]*\n+', '', content)
            content = re.sub(r'(?i)This section provides a (structured|readable|comprehensive).*?$', '', content)
            
            if "## References" in content or section.title.lower() == "references":
                has_references_section = True
            
            unnumber_title = re.sub(r'^\d+[\.\)]\s*', '', section.title).strip()
            if unnumber_title.lower() in ["executive summary", "executive overview", "executive summary & context"]:
                unnumber_title = "Overview & Core Concept"
            
            # Remove any leading repetition of section title
            clean_title = re.escape(section.title)
            content = re.sub(r'^(#+\s*)?' + clean_title + r'[:\s]*\n+', '', content, flags=re.IGNORECASE)
            clean_unnumber = re.escape(unnumber_title)
            content = re.sub(r'^(#+\s*)?(\d+[\.\)]\s*)?' + clean_unnumber + r'[:\s]*\n+', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^(\*\*|__)?Section\s*\d+[:\.]\s*.*?\n+', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^\d+[\.\)]\s*' + clean_unnumber + r'[:\s]*\n+', '', content, flags=re.IGNORECASE)
            content = content.strip()
            
            report_parts.append(f"## {section_idx}. {unnumber_title}\n\n")
            report_parts.append(f"{content}\n\n")
            report_parts.append("---\n\n")

        # References Section
        if not has_references_section:
            report_parts.append("## References\n\n")
            source_info = []
            seen_urls = set()
            
            for result in search_results:
                if hasattr(result, 'url') and result.url and result.url not in seen_urls:
                    seen_urls.add(result.url)
                    title = getattr(result, 'title', '')
                    source_info.append((result.url, title))
            
            for section in report_sections:
                if hasattr(section, 'sources'):
                    for url in section.sources:
                        if url not in seen_urls:
                            seen_urls.add(url)
                            source_info.append((url, ''))
            
            if source_info:
                for i, (url, title) in enumerate(source_info[:30], 1):
                    citation = self.citation_formatter.format_apa(url, title)
                    report_parts.append(f"{i}. {citation}\n")
            else:
                report_parts.append("*No sources were available for this research.*\n")
        
        return "".join(report_parts)
