"""Final report generation node — synthesizes research into a markdown report."""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith.run_helpers import get_current_run_tree

from deep_research.configuration import Configuration
from deep_research.graph.model import build_model_config, configurable_model
from deep_research.helpers.source_store import (
    build_source_map,
    format_source_map_for_prompt,
    get_sources_dir,
    resolve_citations,
)
from deep_research.prompts import final_report_prompt
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    """Generate a comprehensive markdown report from research findings.

    Takes the research brief and accumulated notes, produces a final
    structured report. No token-limit retry yet (Increment 5).
    """
    configurable = Configuration.from_runnable_config(config)

    model_config = build_model_config(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        temperature=configurable.research_model_temperature,
        thinking_budget=configurable.research_model_thinking_budget,
    )

    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    # Build source map for citation resolution
    sources_dir = get_sources_dir()
    source_map = build_source_map(sources_dir)
    source_map_text = format_source_map_for_prompt(source_map)

    prompt = final_report_prompt.format(
        brief=state["research_brief"],
        notes=state["notes"],
        report_metadata=state.get("report_metadata", ""),
        source_map=source_map_text,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    logger.info("Generating final report (brief: %d chars, notes: %d chars, sources: %d)",
                len(state["research_brief"]), len(state["notes"]), len(source_map))
    response = await model.ainvoke([HumanMessage(content=prompt)])
    report = response.text
    logger.info("Report generated: %d chars", len(report))

    if not report:
        logger.warning("Report generation returned empty text")
        rt = get_current_run_tree()
        if rt:
            rt.metadata["fallback"] = "empty_report"
        # Fall back to notes as raw report so the run still produces output
        report = f"# Research Notes\n\n{state['notes']}"

    # Post-process: resolve [source_id] → [N] + append Sources section
    if source_map:
        report, warnings = resolve_citations(report, source_map)
        for w in warnings:
            logger.warning("Citation warning: %s", w)

    return {"final_report": report}
