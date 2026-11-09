"""Interactive Chainlit interface for Deep Research Agent with enhanced UX."""

import asyncio
import html
import chainlit as cl
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.config import config
from src.state import ResearchState
from src.graph import create_research_graph
from src.utils.exports import ReportExporter
from src.utils.history import ResearchHistory
from src.utils.database import get_data_layer
from chainlit.data import get_data_layer as get_active_data_layer
import logging

logger = logging.getLogger(__name__)
from src.callbacks import (
    progress_callback, 
    ProgressUpdate, 
    ResearchStage,
    emit_complete
)
from src.agents.planner import clean_topic_str


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_EMOJI = {
    ResearchStage.INITIALIZING: "",
    ResearchStage.PLANNING: "",
    ResearchStage.SEARCHING: "",
    ResearchStage.EXTRACTING: "",
    ResearchStage.SYNTHESIZING: "",
    ResearchStage.WRITING: "",
    ResearchStage.COMPLETE: "",
    ResearchStage.ERROR: ""
}

STAGE_COLORS = {
    ResearchStage.INITIALIZING: "#6b7280",
    ResearchStage.PLANNING: "#8b5cf6",
    ResearchStage.SEARCHING: "#3b82f6",
    ResearchStage.EXTRACTING: "#06b6d4",
    ResearchStage.SYNTHESIZING: "#10b981",
    ResearchStage.WRITING: "#f59e0b",
    ResearchStage.COMPLETE: "#22c55e",
    ResearchStage.ERROR: "#ef4444"
}

STAGE_NAMES = {
    ResearchStage.INITIALIZING: "Initializing",
    ResearchStage.PLANNING: "Planning Strategy",
    ResearchStage.SEARCHING: "Web Search",
    ResearchStage.EXTRACTING: "Content Extraction",
    ResearchStage.SYNTHESIZING: "AI Synthesis",
    ResearchStage.WRITING: "Report Writing",
    ResearchStage.COMPLETE: "Complete",
    ResearchStage.ERROR: "Error"
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedProgressDisplay:
    """Modern progress display with animated visuals and glassmorphism styling."""
    
    def __init__(self):
        self.message: cl.Message = None
        self.step_messages: dict = {}
        self.updates: list[ProgressUpdate] = []
        self.current_stage: ResearchStage = ResearchStage.INITIALIZING
        self.start_time: datetime = None
        self.topic: str = ""
        
    async def initialize(self, topic: str):
        """Initialize with a sleek progress card."""
        self.start_time = datetime.now()
        self.updates = []
        self.topic = topic
        self.current_stage = ResearchStage.INITIALIZING
        
        content = self._render()
        self.message = cl.Message(content=content)
        await self.message.send()
    
    async def update(self, progress_update: ProgressUpdate):
        """Update with smooth transitions."""
        self.updates.append(progress_update)
        self.current_stage = progress_update.stage
        
        if self.message:
            self.message.content = self._render()
            await self.message.update()
    
    def _get_elapsed(self) -> str:
        """Get formatted elapsed time."""
        if not self.start_time:
            return "0s"
        delta = datetime.now() - self.start_time
        minutes, seconds = divmod(delta.seconds, 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    def _render(self) -> str:
        """Render the complete modern progress display with glassmorphic cards."""
        elapsed = self._get_elapsed()
        
        pct = 0
        if self.updates:
            for update in reversed(self.updates):
                if update.progress_pct is not None:
                    pct = update.progress_pct
                    break
        
        stages = [
            (ResearchStage.PLANNING, "Strategic Planning", "🎯"),
            (ResearchStage.SEARCHING, "Deep Web Search", "🌐"),
            (ResearchStage.EXTRACTING, "Content Extraction", "📑"),
            (ResearchStage.SYNTHESIZING, "Multi-Agent Synthesis", "🧠"),
            (ResearchStage.WRITING, "Report Compilation", "✍️"),
        ]
        
        stage_keys = [s[0] for s in stages]
        current_idx = -1
        if self.current_stage in stage_keys:
            current_idx = stage_keys.index(self.current_stage)
        elif self.current_stage == ResearchStage.COMPLETE:
            current_idx = 99
        
        stepper_html = '<div class="stepper-pipeline">'
        for idx, (stg, name, icon) in enumerate(stages):
            if idx < current_idx:
                status_class = "step-node completed"
                state_label = "✓ Completed"
            elif idx == current_idx:
                status_class = "step-node active"
                state_label = '<span class="pulsing-dot"></span> In Progress'
            else:
                status_class = "step-node"
                state_label = "Queued"
            
            stepper_html += f'<div class="{status_class}"><span>{icon} {name}</span><small>{state_label}</small></div>'
        stepper_html += '</div>'
        
        activity_items_html = ""
        recent = self.updates[-6:] if self.updates else []
        if not recent:
            activity_items_html = '<div class="activity-item"><span class="activity-text"><em>Orchestrating autonomous agents...</em></span></div>'
        else:
            for update in reversed(recent):
                t_str = update.timestamp.strftime("%H:%M:%S")
                safe_msg = html.escape(str(update.message))
                if update.details:
                    truncated = update.details[:75] + '...' if len(update.details) > 75 else update.details
                    safe_detail = html.escape(str(truncated))
                    detail_str = f"<br><small style='color:#94a3b8;'>{safe_detail}</small>"
                else:
                    detail_str = ""
                activity_items_html += f'<div class="activity-item"><span class="activity-time">{t_str}</span><span class="activity-text"><strong>{safe_msg}</strong>{detail_str}</span></div>\n'
        
        safe_topic = html.escape(str(self.topic))
        return f"""<div class="research-container">
<div class="research-header">
<div class="research-title">
<span class="pulsing-dot"></span>
Apex Autonomous Research Studio
</div>
<span class="topic-badge">{safe_topic}</span>
</div>
<div class="progress-meta">
<span>Pipeline Progress: <strong>{pct:.0f}%</strong></span>
<span>Elapsed: <strong>{elapsed}</strong></span>
</div>
<div class="progress-track">
<div class="progress-fill" style="width: {pct}%;"></div>
</div>
{stepper_html}
<div style="margin-top: 14px;">
<div style="font-size: 0.75rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Live Agent Telemetry</div>
<div class="activity-box">
{activity_items_html}
</div>
</div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_research_with_updates(topic: str, progress_display: EnhancedProgressDisplay):
    """Run research with real-time updates."""
    progress_callback.reset()
    
    async def on_progress(update: ProgressUpdate):
        await progress_display.update(update)
    
    progress_callback.register_async(on_progress)
    
    try:
        initial_state = ResearchState(research_topic=topic)
        graph = create_research_graph()
        final_state = await graph.ainvoke(initial_state)
        
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        await emit_complete(topic, len(search_results), len(key_findings))
        
        return final_state
    finally:
        progress_callback.unregister(on_progress)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE & DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@cl.data_layer
def setup_data_layer():
    """Configure SQLite persistence for chat history and threads."""
    return get_data_layer()


@cl.header_auth_callback
async def header_auth(headers: dict) -> Optional[cl.User]:
    """Auto-authenticate user session so chat history is permanently persisted."""
    return cl.User(identifier="sanchitgarg", metadata={"role": "admin", "name": "Sanchit Garg"})


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Resume a previous research conversation when selected from the sidebar."""
    cl.user_session.set("research_count", len(thread.get("steps", [])))
    logger.info(f"Resuming thread: {thread.get('name')} (id={thread.get('id')})")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@cl.on_chat_start
async def start():
    """Initialize with a bespoke, polished research studio experience."""
    cl.user_session.set("research_count", 0)
    
    actions = [
        cl.Action(
            name="example_quantum",
            payload={"topic": "Quantum computing fault tolerance and 2026 outlook"},
            label="🔬 Quantum Fault Tolerance"
        ),
        cl.Action(
            name="example_ai",
            payload={"topic": "Multi-agent orchestration frameworks and production memory architectures"},
            label="🤖 Multi-Agent Frameworks"
        ),
        cl.Action(
            name="example_climate",
            payload={"topic": "Quantitative alpha formulation and backtesting risk models"},
            label="📈 Quantitative Finance"
        ),
        cl.Action(
            name="show_history",
            payload={},
            label="📁 History"
        ),
        cl.Action(
            name="show_settings",
            payload={},
            label="⚙️ System Config"
        )
    ]
    
    welcome_content = """
<div class="research-container" style="margin-top: 10px;">
  <div class="research-header">
    <div class="research-title">
      <span style="font-size: 1.3rem;">⚡</span> Apex Deep Research Studio
    </div>
    <span class="badge-high">Engine v2.0 • Sanchit Garg</span>
  </div>
  
  <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-top: 4px;">
    Autonomous intelligence system orchestrated with <strong>LangGraph</strong> and a <strong>Multi-Source Parallel Engine</strong>. Integrates <strong>arXiv</strong> preprints, <strong>Semantic Scholar</strong> scientific graphs, <strong>Crossref</strong> DOIs, <strong>GitHub</strong> codebases, <strong>News</strong> wires, and authoritative <strong>Web</strong> analysis with cross-source deduplication and verified citations.
  </p>

  <div class="metrics-grid">
    <div class="metric-tile">
      <div class="metric-tile-title">Multi-Source Retrieval</div>
      <div class="metric-tile-sub">arXiv, Semantic Scholar, Crossref, GitHub, News, Web</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Entity Resolution</div>
      <div class="metric-tile-sub">DOI, arXiv ID & fuzzy title deduplication</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Multidimensional Credibility</div>
      <div class="metric-tile-sub">Citations, peer review, repo stars & domain ranking</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Academic Citations</div>
      <div class="metric-tile-sub">APA, MLA, IEEE, Chicago & BibTeX export</div>
    </div>
  </div>

  <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 14px;">
    <strong>Quick Launch:</strong> Select a featured research track below, or enter any query directly in the console.
  </p>
</div>
"""
    await cl.Message(
        content=welcome_content,
        actions=actions
    ).send()


@cl.action_callback("example_quantum")
async def on_example_quantum(action: cl.Action):
    """Handle quantum computing example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("example_ai")
async def on_example_ai(action: cl.Action):
    """Handle AI agents example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("example_climate")
async def on_example_climate(action: cl.Action):
    """Handle climate tech example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("show_history")
async def on_show_history(action: cl.Action):
    """Show research history."""
    history = ResearchHistory()
    entries = history.get_recent(limit=10)
    
    if not entries:
        await cl.Message(
            content="""
<div class="research-container">
  <div class="research-header">
    <div class="research-title">📁 Investigation History</div>
    <span class="badge-med">No Prior Runs</span>
  </div>
  <p style="color: #94a3b8; font-size: 0.9rem;">
    No prior research investigations recorded yet. Start your first research above! All conversations are automatically preserved in the left sidebar.
  </p>
</div>
"""
        ).send()
        return
    
    table_rows = ""
    for entry in entries:
        date = entry.get('timestamp', 'N/A')
        if isinstance(date, str) and len(date) > 10:
            date = date[:10]
        topic = entry.get('topic', 'Unknown')
        if len(topic) > 35:
            topic = topic[:32] + "..."
        sources = entry.get('metadata', {}).get('sources', 'N/A')
        findings = entry.get('metadata', {}).get('findings', 'N/A')
        table_rows += f"| {date} | **{topic}** | {sources} | {findings} |\n"
    
    content = f"""
<div class="research-container">
  <div class="research-header">
    <div class="research-title">📁 Investigation Archives</div>
    <span class="badge-high">Stored in SQLite</span>
  </div>
  <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">
    💡 <strong>Sidebar History:</strong> All your past sessions are permanently stored in the left sidebar. You can refresh or switch between conversations anytime!
  </p>
</div>

| Date | Research Topic | Sources | Insights |
|------|----------------|---------|----------|
{table_rows}
"""
    await cl.Message(content=content).send()


@cl.action_callback("show_settings")
async def on_show_settings(action: cl.Action):
    """Show current configuration."""
    content = f"""# Current Settings

## Model Configuration
| Setting | Value |
|---------|-------|
| Provider | `{config.model_provider}` |
| Model | `{config.model_name}` |
| Summarization Model | `{config.summarization_model}` |

## Search Configuration
| Setting | Value |
|---------|-------|
| Max Search Queries | `{config.max_search_queries}` |
| Results per Query | `{config.max_search_results_per_query}` |
| Min Credibility Score | `{config.min_credibility_score}` |

## Report Configuration
| Setting | Value |
|---------|-------|
| Max Sections | `{config.max_report_sections}` |
| Min Words/Section | `{config.min_section_words}` |
| Citation Style | `{config.citation_style.upper()}` |

---

*To change settings, modify your `.env` file and restart the app.*
"""
    await cl.Message(content=content).send()


@cl.action_callback("download_md")
async def on_download_md(action: cl.Action):
    """Handle markdown download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**Markdown Report:**", elements=elements).send()


@cl.action_callback("download_html")
async def on_download_html(action: cl.Action):
    """Handle HTML download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**HTML Report:**", elements=elements).send()


@cl.action_callback("download_txt")
async def on_download_txt(action: cl.Action):
    """Handle TXT download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**Plain Text Report:**", elements=elements).send()


@cl.action_callback("view_sources")
async def on_view_sources(action: cl.Action):
    """Show detailed source list across all research channels."""
    sources = action.payload.get("sources", [])
    credibility = action.payload.get("credibility", [])
    
    if not sources:
        await cl.Message(content="*No sources available.*").send()
        return
    
    content = "# Multi-Source Verification Analysis\n\n"
    content += "| # | Channel | Credibility | Title | Metrics / Metadata | Link |\n"
    content += "|---|---------|-------------|-------|--------------------|------|\n"
    
    for i, source in enumerate(sources[:35]):
        title = source.get('title', 'Unknown')[:35]
        if len(source.get('title', '')) > 35:
            title += "..."
        url = source.get('url', 'N/A')
        stype = source.get('source_type', 'web').upper()
        
        evidence_parts = []
        if source.get('stars') is not None:
            evidence_parts.append(f"⭐ {source['stars']:,}")
        if source.get('citations') is not None:
            evidence_parts.append(f"📄 {source['citations']:,} cites")
        if source.get('doi'):
            evidence_parts.append(f"DOI: {source['doi'][:16]}")
        if source.get('venue') and not evidence_parts:
            evidence_parts.append(source['venue'][:20])
            
        evidence_str = " | ".join(evidence_parts) if evidence_parts else "Web Content"
        
        # Get credibility badge
        cred = credibility[i] if i < len(credibility) else {}
        level = cred.get('level', 'unknown')
        score = cred.get('score', 'N/A')
        
        badge = "[HIGH]" if level == 'high' else "[MED]" if level == 'medium' else "[LOW]"
        content += f"| {i+1} | `{stype}` | {badge} {score} | {title} | {evidence_str} | [Open]({url}) |\n"
    
    await cl.Message(content=content).send()


@cl.action_callback("view_findings")
async def on_view_findings(action: cl.Action):
    """Show key findings in detail."""
    findings = action.payload.get("findings", [])
    
    if not findings:
        await cl.Message(content="*No findings available.*").send()
        return
    
    content = "# Key Findings\n\n"
    for i, finding in enumerate(findings, 1):
        content += f"**{i}.** {finding}\n\n"
    
    await cl.Message(content=content).send()


async def start_research(topic: str):
    """Start research with the given topic."""
    
    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        await cl.Message(
            content=f"""## Configuration Error

**Error:** {str(e)}

Please ensure your `.env` file is properly configured with the required API keys.

```
# Example .env configuration
GEMINI_API_KEY=your-api-key-here
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
```
"""
        ).send()
        return
    
    # Increment research count
    count = cl.user_session.get("research_count", 0) + 1
    cl.user_session.set("research_count", count)

    clean_topic = clean_topic_str(topic)

    # Name the conversation thread in the sidebar
    try:
        active_dl = get_active_data_layer()
        if active_dl and hasattr(cl.context, 'session') and cl.context.session:
            thread_id = getattr(cl.context.session, 'thread_id', None)
            if thread_id:
                clean_name = clean_topic
                if len(clean_name) > 40:
                    clean_name = clean_name[:37] + "..."
                await active_dl.update_thread(thread_id=thread_id, name=clean_name)
    except Exception as e:
        logger.warning(f"Could not update thread title: {e}")
    
    # Show configuration summary
    await cl.Message(
        content=f"""
<div class="research-container">
  <div class="research-header">
    <div class="research-title"><span>🚀</span> Launching Investigation #{count}</div>
    <span class="topic-badge">{clean_topic}</span>
  </div>
  <div class="metrics-grid">
    <div class="metric-tile">
      <div class="metric-tile-title">Inference Engine</div>
      <div class="metric-tile-sub"><code>{config.model_name}</code></div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Search Breadth</div>
      <div class="metric-tile-sub">{config.max_search_queries} strategic queries</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Analysis Depth</div>
      <div class="metric-tile-sub">{config.max_report_sections} structured sections</div>
    </div>
  </div>
  <p style="color: #64748b; font-size: 0.8rem; margin: 8px 0 0 0;">
    Autonomous orchestration in progress. Standby for real-time telemetry...
  </p>
</div>
"""
    ).send()
    
    # Initialize progress display
    progress_display = EnhancedProgressDisplay()
    await progress_display.initialize(topic)
    
    try:
        # Run research
        final_state = await run_research_with_updates(topic, progress_display)
        
        # Check for errors
        if final_state.get("error"):
            await cl.Message(
                content=f"""## Research Failed

**Error:** {final_state.get('error')}

Please try again or simplify your research topic.
"""
            ).send()
            return
        
        # Extract results
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        report_sections = final_state.get('report_sections', [])
        credibility_scores = final_state.get('credibility_scores', [])
        source_breakdown = final_state.get('source_breakdown', {})
        
        # Count metrics
        unique_sources = set()
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)
        
        high_cred = sum(1 for s in credibility_scores if s.get('level') == 'high')
        medium_cred = sum(1 for s in credibility_scores if s.get('level') == 'medium')
        
        # Source breakdown string
        breakdown_badges = []
        for st, count in source_breakdown.items():
            breakdown_badges.append(f"<span class='badge-high'>{st.title()}: {count}</span>")
        breakdown_html = " ".join(breakdown_badges) if breakdown_badges else f"<span class='badge-high'>{high_cred} High</span> <span class='badge-med'>{medium_cred} Med</span>"

        # LLM metrics
        llm_calls = final_state.get('llm_calls', 0)
        total_input = final_state.get('total_input_tokens', 0)
        total_output = final_state.get('total_output_tokens', 0)
        total_tokens = total_input + total_output
        
        # Elapsed time
        elapsed = 0
        if progress_display.start_time:
            elapsed = (datetime.now() - progress_display.start_time).seconds
        
        # Create interactive summary with action buttons
        summary_actions = [
            cl.Action(
                name="view_sources",
                payload={
                    "sources": [
                        {
                            "title": getattr(r, 'title', ''),
                            "url": getattr(r, 'url', ''),
                            "source_type": getattr(r, 'source_type', 'web'),
                            "doi": getattr(r, 'doi', None),
                            "stars": getattr(r, 'repo_stars', None),
                            "citations": getattr(r, 'citation_count', None),
                            "venue": getattr(r, 'journal_or_venue', None)
                        }
                        for r in search_results[:35]
                    ],
                    "credibility": credibility_scores[:35]
                },
                label="View Multi-Source Dossier"
            ),
            cl.Action(
                name="view_findings",
                payload={"findings": key_findings[:20]},
                label="View Findings"
            )
        ]
        
        summary_content = f"""
<div class="research-container">
  <div class="research-header">
    <div class="research-title">
      <span>✨</span> Multi-Source Investigation Complete
    </div>
    <span class="badge-high">Status: Verified Dossier</span>
  </div>

  <div class="metrics-grid">
    <div class="metric-tile">
      <div class="metric-tile-title">Verified Sources</div>
      <div class="metric-tile-value">{len(unique_sources)}</div>
      <div class="metric-tile-sub">{breakdown_html}</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Synthesized Findings</div>
      <div class="metric-tile-value">{len(key_findings)}</div>
      <div class="metric-tile-sub">{len(report_sections)} Core Sections</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Execution Latency</div>
      <div class="metric-tile-value">{elapsed}s</div>
      <div class="metric-tile-sub">Parallel Multi-Channel</div>
    </div>
    <div class="metric-tile">
      <div class="metric-tile-title">Token Utilization</div>
      <div class="metric-tile-value">{total_tokens:,}</div>
      <div class="metric-tile-sub">{llm_calls} Agent Invocations</div>
    </div>
  </div>
</div>
"""
        
        await cl.Message(
            content=summary_content,
            actions=summary_actions
        ).send()
        
        # Save and display report
        if final_state.get("final_report"):
            report = final_state["final_report"]
            
            # Save files
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_topic = safe_topic[:30].strip()
            filename = f"{safe_topic}_{timestamp}.md"
            output_file = output_dir / filename
            output_file.write_text(report, encoding='utf-8')
            
            # Export formats
            exporter = ReportExporter()
            base_path = output_file.with_suffix('')
            html_file = exporter.export(report, base_path, format='html')
            txt_file = exporter.export(report, base_path, format='txt')
            
            # Add to history
            history = ResearchHistory()
            history.add_research(
                topic=topic,
                output_file=output_file,
                metadata={
                    'sources': len(unique_sources),
                    'sections': len(report_sections),
                    'findings': len(key_findings),
                    'elapsed_seconds': elapsed,
                    'total_tokens': total_tokens
                }
            )
            
            # Download actions
            download_actions = [
                cl.Action(
                    name="download_md",
                    payload={"path": str(output_file)},
                    label="Markdown"
                ),
                cl.Action(
                    name="download_html",
                    payload={"path": str(html_file)},
                    label="HTML"
                ),
                cl.Action(
                    name="download_txt",
                    payload={"path": str(txt_file)},
                    label="Text"
                )
            ]
            
            await cl.Message(
                content=f"""
<div class="research-container">
  <div class="research-header">
    <div class="research-title"><span>📥</span> Export Dossier Artifacts</div>
    <span class="badge-high">Ready for Download</span>
  </div>
  <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
    The generated report has been saved to <code>outputs/</code>. Select your preferred export format:
  </p>
</div>
""",
                actions=download_actions
            ).send()
            
            # Display the report
            await cl.Message(content=report).send()
            
            # Prompt for next research
            next_actions = [
                cl.Action(
                    name="example_quantum",
                    payload={"topic": f"Latest developments in {topic.split()[0] if topic.split() else 'technology'}"},
                    label="Related Topic"
                )
            ]
            
            await cl.Message(
                content="""---

## Ready for More?

Type another research topic or click a suggested action above!

*Tip: You can ask follow-up questions about specific aspects of your research.*
""",
                actions=next_actions
            ).send()
        
        else:
            await cl.Message(
                content="**Warning:** No report was generated. Please try again with a different topic."
            ).send()
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        await cl.Message(
            content=f"""## Unexpected Error

**Error:** {str(e)}

<details>
<summary>Technical Details (click to expand)</summary>

```python
{error_details}
```

</details>

Please check the logs and try again.
"""
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    topic = message.content.strip()
    
    if not topic:
        await cl.Message(
            content="Please provide a research topic."
        ).send()
        return
    
    # Check for casual conversational greetings
    casual_greetings = ["yo", "hi", "hello", "hey", "hola", "sup", "hey there", "greetings"]
    if topic.lower().strip("!.? ") in casual_greetings:
        await cl.Message(
            content="""### 👋 Welcome to Apex Research OS!

I am your autonomous **Multi-Source Research Agent**, equipped to conduct comprehensive investigations across:
- 🌐 **Web**: Encyclopedic definitions, industry overviews, and official documentation
- 📚 **arXiv & Semantic Scholar**: Academic publications, scientific breakthroughs, and preprints
- 🔗 **Crossref**: Peer-reviewed journals and metadata
- 💻 **GitHub**: Open-source implementations, code repositories, and benchmarks
- 📰 **News**: Real-time market events and industry announcements

**What topic would you like to investigate?**
Try asking:
- *"What is the meaning and philosophy of science?"*
- *"State of quantum computing and error correction in 2025"*
- *"Explain transformer architectures in vision models"*
"""
        ).send()
        return

    # Check for special commands
    if topic.lower() in ["/history", "history", "show history"]:
        await on_show_history(cl.Action(name="show_history", payload={}))
        return
    
    if topic.lower() in ["/settings", "settings", "show settings"]:
        await on_show_settings(cl.Action(name="show_settings", payload={}))
        return
    
    if topic.lower() in ["/help", "help"]:
        await cl.Message(
            content="""## Help

### Commands
- `/history` - View your research history
- `/settings` - View current configuration
- `/help` - Show this help message

### Tips
- Be specific with your research topics for better results
- Use questions like "What are the latest trends in..."
- Include time context like "in 2025" for current information

### Example Topics
- "Future of quantum computing in 2025"
- "How do transformer models work?"
- "Best practices for microservices architecture"
"""
        ).send()
        return
    
    await start_research(topic)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
