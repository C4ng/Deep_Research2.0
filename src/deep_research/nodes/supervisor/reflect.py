"""Supervisor reflection node — assess cross-topic completeness and route.

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
from deep_research.graph.model import configurable_model
from deep_research.models import ResearchResult, SupervisorReflection
from deep_research.nodes.supervisor.supervisor import _format_research_results
from deep_research.prompts import supervisor_reflection_prompt
from deep_research.state import SupervisorState

logger = logging.getLogger(__name__)


def _merge_notes(results: list[ResearchResult]) -> str:
    """Merge all researcher notes with topic headers for the final report."""
    parts = []
    for r in results:
        if r.notes:
            parts.append(f"## Topic: {r.topic}\n\n{r.notes}")
    return "\n\n---\n\n".join(parts)


def _format_reflection_guidance(reflection: SupervisorReflection) -> str:
    """Format reflection output as guidance for the next supervisor round."""
    parts = [reflection.overall_assessment, ""]

    if reflection.coverage_gaps:
        parts.append("Coverage gaps to address:")
        parts.extend(f"- {g}" for g in reflection.coverage_gaps)
        parts.append("")

    if reflection.cross_topic_contradictions:
        parts.append("Cross-topic contradictions to resolve:")
        parts.extend(f"- {c}" for c in reflection.cross_topic_contradictions)

    return "\n".join(parts)


async def supervisor_reflect(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Assess cross-topic research completeness and route.

    On exit: merges all researcher notes into combined notes for the report.
    On continue: formats reflection as guidance for the next supervisor round.
    """
    configurable = Configuration.from_runnable_config(config)
    iteration = state.get("supervisor_iterations", 0) + 1
    logger.info("Supervisor reflection round %d", iteration)

    # Build reflection prompt
    results = state.get("research_results", [])
    formatted_results = _format_research_results(results)

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    model = (
        configurable_model
        .with_structured_output(SupervisorReflection)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    prompt = supervisor_reflection_prompt.format(
        research_brief=state["research_brief"],
        research_results=formatted_results,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    reflection = await model.ainvoke([HumanMessage(content=prompt)])

    logger.info(
        "Supervisor reflection: knowledge_state=%s, should_continue=%s, "
        "gaps=%d, contradictions=%d",
        reflection.knowledge_state,
        reflection.should_continue,
        len(reflection.coverage_gaps),
        len(reflection.cross_topic_contradictions),
    )

    max_iterations = getattr(configurable, "max_supervisor_iterations", 3)
    should_stop = (
        not reflection.should_continue
        or reflection.knowledge_state == "sufficient"
        or iteration >= max_iterations
    )

    if should_stop:
        if iteration >= max_iterations:
            logger.warning(
                "Forcing exit — max supervisor iterations (%d) reached",
                iteration,
            )
        combined_notes = _merge_notes(results)
        logger.info("Supervisor routing to END, merged %d researcher notes", len(results))
        return Command(
            goto="__end__",
            update={
                "notes": combined_notes,
                "supervisor_iterations": iteration,
                "last_supervisor_reflection": "",
            },
        )

    guidance = _format_reflection_guidance(reflection)
    logger.info("Supervisor routing back for round %d", iteration + 1)
    return Command(
        goto="supervisor",
        update={
            "supervisor_iterations": iteration,
            "last_supervisor_reflection": guidance,
            "messages": [],
        },
    )
