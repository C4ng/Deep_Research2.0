"""Reflect node — structured reflection after tool execution.

Assesses research progress, accumulates structured knowledge,
and routes to either continue researching or summarize and exit.
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.graph.model import build_model_config, configurable_model
from deep_research.models import ResearchReflection
from deep_research.prompts import researcher_reflection_prompt
from deep_research.state import ResearcherState

logger = logging.getLogger(__name__)


def _extract_tool_results(state: ResearcherState) -> str:
    """Extract all ToolMessage content from messages.

    TODO(context): On round 2+, this includes prior rounds' tool results which
    are redundant with accumulated_context. Filtering to only this round's results
    would reduce context size but requires tracking message boundaries per round.
    """
    return "\n\n".join(
        m.content for m in state.get("messages", [])
        if isinstance(m, ToolMessage) and m.content
    )


def _format_accumulated_context(state: ResearcherState) -> str:
    """Format accumulated findings, contradictions, and gaps for the reflection prompt."""
    prior_findings = state.get("accumulated_findings", [])
    prior_contradictions = state.get("accumulated_contradictions", [])
    prior_gaps = state.get("current_gaps", [])

    if not (prior_findings or prior_contradictions or prior_gaps):
        return ""

    parts = []
    if prior_findings:
        parts.append("Previously identified findings:")
        parts.extend(f"- {f}" for f in prior_findings)
    if prior_contradictions:
        parts.append("\nPreviously identified contradictions:")
        parts.extend(f"- {c}" for c in prior_contradictions)
    if prior_gaps:
        parts.append("\nGaps identified last round (assess if now filled):")
        parts.extend(f"- {g}" for g in prior_gaps)
    return "\n".join(parts)


def _format_reflection(
    reflection: ResearchReflection,
    accumulated_findings: list[str],
) -> str:
    """Format reflection + accumulated context for the researcher.

    Includes both the current round's assessment and accumulated
    findings from all prior rounds so the researcher knows what's
    already covered.
    """
    parts = []

    if accumulated_findings:
        parts.append("Already covered across all rounds:")
        parts.extend(f"- {f}" for f in accumulated_findings)
        parts.append("")

    if reflection.key_findings:
        parts.append("New findings this round:")
        parts.extend(f"- {f}" for f in reflection.key_findings)
        parts.append("")

    parts.append("Missing information:")
    parts.extend(f"- {info}" for info in reflection.missing_info)

    if reflection.contradictions:
        parts.append("\nContradictions to resolve:")
        parts.extend(f"- {c}" for c in reflection.contradictions)

    if reflection.next_queries:
        parts.append("\nSuggested next queries:")
        parts.extend(f"- {q}" for q in reflection.next_queries)

    return "\n".join(parts)


async def _run_reflection(
    state: ResearcherState, config: RunnableConfig
) -> ResearchReflection:
    """Call the model for structured reflection assessment."""
    configurable = Configuration.from_runnable_config(config)

    model_config = build_model_config(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        temperature=configurable.research_model_temperature,
        thinking_budget=configurable.reflection_thinking_budget,
    )

    model = (
        configurable_model
        .with_structured_output(ResearchReflection)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    prompt = researcher_reflection_prompt.format(
        research_topic=state["research_topic"],
        findings=_extract_tool_results(state),
        accumulated_context=_format_accumulated_context(state),
        date=datetime.now().strftime("%B %d, %Y"),
    )

    return await model.ainvoke([HumanMessage(content=prompt)])


async def reflect(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "summarize"]]:
    """Assess research progress and decide whether to continue or summarize.

    Accumulates key_findings and contradictions into state fields.
    Routes to 'researcher' if more searching is needed,
    or to 'summarize' when research is complete or max iterations reached.
    """
    configurable = Configuration.from_runnable_config(config)
    iteration = state.get("research_iterations", 0) + 1
    logger.info("Reflection round %d — assessing research progress", iteration)

    reflection = await _run_reflection(state, config)

    logger.info(
        "Reflection result: knowledge_state=%s, should_continue=%s, "
        "gaps=%d, contradictions=%d, prior_gaps_filled=%d",
        reflection.knowledge_state,
        reflection.should_continue,
        len(reflection.missing_info),
        len(reflection.contradictions),
        reflection.prior_gaps_filled,
    )

    # Accumulate structured knowledge
    # - findings: append reducer (grows across rounds)
    # - contradictions: overwrite (LLM outputs full canonical list each round)
    # - gaps: overwrite (latest assessment only)
    accumulation_update = {
        "accumulated_findings": reflection.key_findings,
        "accumulated_contradictions": reflection.contradictions,
        "current_gaps": reflection.missing_info,
        "research_iterations": iteration,
    }

    # Routing decision
    should_stop = (
        not reflection.should_continue
        or reflection.knowledge_state == "sufficient"
        or iteration >= configurable.max_research_iterations
    )

    # Dead-end detection: gaps persist unfilled across rounds
    prior_gap_count = len(state.get("current_gaps", []))
    dead_end = (
        prior_gap_count > 0
        and reflection.prior_gaps_filled == 0
        and iteration >= 2
    )

    if dead_end and not should_stop:
        if iteration >= 3:
            # Reformulation already attempted — force exit
            logger.warning(
                "Dead end persists after reformulation — forcing exit "
                "(round %d, %d unfilled gaps)", iteration, prior_gap_count,
            )
            should_stop = True
        else:
            # First dead end — inject reformulation guidance, continue
            logger.info(
                "Dead end detected (round %d, %d gaps unfilled) — "
                "injecting reformulation guidance", iteration, prior_gap_count,
            )

    if should_stop:
        if iteration >= configurable.max_research_iterations:
            logger.warning("Forcing exit — max research iterations (%d) reached", iteration)
        logger.info("Routing to summarize")
        return Command(
            goto="summarize",
            update={
                **accumulation_update,
                "last_reflection": "",
                "final_knowledge_state": reflection.knowledge_state,
            },
        )

    all_findings = list(state.get("accumulated_findings", [])) + reflection.key_findings
    reflection_text = _format_reflection(reflection, all_findings)

    # Append reformulation guidance on dead end
    if dead_end:
        reflection_text += (
            "\n\nDEAD END: The gaps above persisted despite targeted searches. "
            "Do NOT repeat similar queries. Instead:\n"
            "- Try synonyms or alternative terminology\n"
            "- Approach from a different angle or adjacent topic\n"
            "- Search for the information in different source types "
            "(academic papers, government reports, industry analyses)\n"
            "- Broaden or narrow the scope to find related information"
        )

    logger.info("Routing back to researcher for round %d", iteration + 1)
    return Command(
        goto="researcher",
        update={
            **accumulation_update,
            "last_reflection": reflection_text,
        },
    )
