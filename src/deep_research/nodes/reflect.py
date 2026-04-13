"""Reflect node — structured reflection after tool execution.

Assesses research progress and routes to either continue
researching or compress findings and exit.
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.models import Reflection
from deep_research.prompts import reflection_prompt
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


def _extract_tool_results(state: AgentState) -> str:
    """Extract all ToolMessage content from messages."""
    return "\n\n".join(
        m.content for m in state.get("messages", [])
        if isinstance(m, ToolMessage) and m.content
    )


def _format_reflection(reflection: Reflection) -> str:
    """Format a Reflection into a readable string for the researcher."""
    parts = [
        "Missing information:",
        *[f"- {info}" for info in reflection.missing_info],
    ]
    if reflection.contradictions:
        parts.append("\nContradictions to resolve:")
        parts.extend(f"- {c}" for c in reflection.contradictions)
    if reflection.next_queries:
        parts.append("\nSuggested next queries:")
        parts.extend(f"- {q}" for q in reflection.next_queries)
    return "\n".join(parts)


async def reflect(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["researcher", "compress"]]:
    """Assess research progress and decide whether to continue or compress.

    Routes to 'researcher' if more searching is needed,
    or to 'compress' when research is complete or max iterations reached.
    """
    configurable = Configuration.from_runnable_config(config)

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    model = (
        configurable_model
        .with_structured_output(Reflection)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    findings = _extract_tool_results(state)
    prompt = reflection_prompt.format(
        research_brief=state["research_brief"],
        findings=findings,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    iteration = state.get("research_iterations", 0) + 1
    logger.info("Reflection round %d — assessing research progress", iteration)

    reflection: Reflection = await model.ainvoke([HumanMessage(content=prompt)])

    logger.info(
        "Reflection result: knowledge_state=%s, should_continue=%s, "
        "gaps=%d, contradictions=%d",
        reflection.knowledge_state,
        reflection.should_continue,
        len(reflection.missing_info),
        len(reflection.contradictions),
    )

    # Routing decision
    should_stop = (
        not reflection.should_continue
        or reflection.knowledge_state == "sufficient"
        or iteration >= configurable.max_research_iterations
    )

    if should_stop:
        if iteration >= configurable.max_research_iterations:
            logger.warning("Forcing exit — max research iterations (%d) reached", iteration)
        logger.info("Routing to compress")
        return Command(
            goto="compress",
            update={
                "research_iterations": iteration,
                "last_reflection": "",
            },
        )

    logger.info("Routing back to researcher for round %d", iteration + 1)
    return Command(
        goto="researcher",
        update={
            "research_iterations": iteration,
            "last_reflection": _format_reflection(reflection),
        },
    )
