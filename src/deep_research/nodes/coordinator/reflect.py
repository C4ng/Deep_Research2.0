"""Coordinator reflection node — assess cross-topic completeness and route.

Evaluates all research results against the original brief, identifies
cross-topic gaps and contradictions, and decides whether to continue
with follow-up research or exit.
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.graph.model import build_model_config, configurable_model
from deep_research.models import CoordinatorReflection, ResearchResult
from deep_research.nodes.coordinator.coordinator import _format_research_results
from deep_research.prompts import coordinator_reflection_prompt
from deep_research.state import CoordinatorState

logger = logging.getLogger(__name__)


def _merge_notes(results: list[ResearchResult]) -> str:
    """Merge all researcher notes with topic headers for the final report."""
    parts = []
    for r in results:
        if r.notes:
            parts.append(f"## Topic: {r.topic}\n\n{r.notes}")
    return "\n\n---\n\n".join(parts)


def _format_report_metadata(
    results: list[ResearchResult],
    reflection: CoordinatorReflection,
) -> str:
    """Format research metadata for the report node.

    Combines per-researcher signals (contradictions, gaps, knowledge_state)
    with coordinator-level signals (cross-topic contradictions, coverage gaps,
    overall assessment). The report prompt uses this to surface contradictions
    and gaps honestly.
    """
    parts = []

    # Per-topic signals
    for r in results:
        section = [f"### {r.topic}"]
        section.append(f"Coverage: {r.knowledge_state}")
        if r.contradictions:
            section.append("Contradictions:")
            section.extend(f"- {c}" for c in r.contradictions)
        if r.missing_info:
            section.append("Persistent gaps (searched but not found):")
            section.extend(f"- {g}" for g in r.missing_info)
        parts.append("\n".join(section))

    # Cross-topic signals from coordinator reflection
    if reflection.cross_topic_contradictions:
        cross = ["### Cross-Topic Contradictions"]
        cross.extend(f"- {c}" for c in reflection.cross_topic_contradictions)
        parts.append("\n".join(cross))

    if reflection.coverage_gaps:
        gaps = ["### Coverage Gaps (not investigated)"]
        gaps.extend(f"- {g}" for g in reflection.coverage_gaps)
        parts.append("\n".join(gaps))

    parts.append(f"### Overall Assessment\n{reflection.overall_assessment}")

    return "\n\n".join(parts)


def _format_reflection_guidance(reflection: CoordinatorReflection) -> str:
    """Format reflection output as guidance for the next coordinator round."""
    parts = [reflection.overall_assessment, ""]

    if reflection.coverage_gaps:
        parts.append("Coverage gaps to address:")
        parts.extend(f"- {g}" for g in reflection.coverage_gaps)
        parts.append("")

    if reflection.cross_topic_contradictions:
        parts.append("Cross-topic contradictions to resolve:")
        parts.extend(f"- {c}" for c in reflection.cross_topic_contradictions)

    return "\n".join(parts)


async def coordinator_reflect(
    state: CoordinatorState, config: RunnableConfig
) -> Command[Literal["coordinator", "__end__"]]:
    """Assess cross-topic research completeness and route.

    On exit: merges all researcher notes into combined notes for the report.
    On continue: formats reflection as guidance for the next coordinator round.
    """
    configurable = Configuration.from_runnable_config(config)
    iteration = state.get("coordinator_iterations", 0) + 1
    logger.info("Coordinator reflection round %d", iteration)

    results = state.get("research_results", [])

    # Fail-fast: if this round produced zero successful results,
    # skip the reflection LLM call and exit with whatever we have.
    latest_count = state.get("latest_round_result_count", -1)
    if latest_count == 0:
        logger.warning(
            "All researchers failed in round %d — skipping coordinator "
            "reflection, exiting with %d prior results", iteration, len(results),
        )
        combined_notes = _merge_notes(results)
        return Command(
            goto="__end__",
            update={
                "notes": combined_notes,
                "report_metadata": "",
                "coordinator_iterations": iteration,
                "last_coordinator_reflection": "",
            },
        )

    # Build reflection prompt
    formatted_results = _format_research_results(results)

    model_config = build_model_config(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        temperature=configurable.research_model_temperature,
        thinking_budget=configurable.reflection_thinking_budget,
    )

    model = (
        configurable_model
        .with_structured_output(CoordinatorReflection)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    prompt = coordinator_reflection_prompt.format(
        research_brief=state["research_brief"],
        research_results=formatted_results,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    reflection = await model.ainvoke([HumanMessage(content=prompt)])

    logger.info(
        "Coordinator reflection: knowledge_state=%s, should_continue=%s, "
        "gaps=%d, contradictions=%d",
        reflection.knowledge_state,
        reflection.should_continue,
        len(reflection.coverage_gaps),
        len(reflection.cross_topic_contradictions),
        extra={"event_type": "coordinator_reflection", "event_data": {
            "round": iteration,
            "knowledge_state": reflection.knowledge_state,
            "overall_assessment": reflection.overall_assessment,
            "cross_topic_contradictions": reflection.cross_topic_contradictions,
            "coverage_gaps": reflection.coverage_gaps,
            "should_continue": reflection.should_continue,
        }},
    )

    should_stop = (
        not reflection.should_continue
        or reflection.knowledge_state in ("sufficient", "unavailable")
        or iteration >= configurable.max_coordinator_iterations
    )

    if should_stop:
        if iteration >= configurable.max_coordinator_iterations:
            logger.warning(
                "Forcing exit — max coordinator iterations (%d) reached",
                iteration,
            )
        combined_notes = _merge_notes(results)
        report_metadata = _format_report_metadata(results, reflection)
        logger.info("Coordinator routing to END, merged %d researcher notes", len(results))
        return Command(
            goto="__end__",
            update={
                "notes": combined_notes,
                "report_metadata": report_metadata,
                "coordinator_iterations": iteration,
                "last_coordinator_reflection": "",
            },
        )

    guidance = _format_reflection_guidance(reflection)
    logger.info("Coordinator routing back for round %d", iteration + 1)
    return Command(
        goto="coordinator",
        update={
            "coordinator_iterations": iteration,
            "last_coordinator_reflection": guidance,
            "messages": [],
        },
    )
