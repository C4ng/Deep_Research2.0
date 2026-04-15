"""Brief generation node — transforms user query into a structured research brief.

Follows the same graph-exit pattern as the clarify node:
- Reads messages, calls LLM, routes via Command
- If human review is enabled and the user hasn't approved yet, exits
  to __end__ with the draft brief for the user to review
- On re-invocation, the LLM reads the prior brief + user feedback and
  either revises (ready_to_proceed=False) or proceeds (ready_to_proceed=True)
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.graph.model import build_model_config, configurable_model
from deep_research.models import ResearchBrief
from deep_research.prompts import research_brief_prompt
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


def _format_brief(brief: ResearchBrief) -> str:
    """Format a ResearchBrief as a string for state and display."""
    return f"**{brief.title}**\n\n{brief.research_question}\n\n**Approach:**\n\n{brief.approach}"


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__", "coordinator"]]:
    """Generate or revise a research brief, then route.

    Reads messages and any existing brief/feedback context. The LLM
    determines whether the user approved or requested changes via
    ready_to_proceed. The node routes accordingly:
    - Not approved + review enabled → exit to __end__ for user review
    - Approved or review disabled → proceed to researcher/coordinator
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
        .with_structured_output(ResearchBrief)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    # Prior brief and last user message serve as revision context.
    # When empty, the LLM generates a fresh brief.
    prior_brief = state.get("research_brief", "")
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    feedback = last_msg.content if isinstance(last_msg, HumanMessage) and prior_brief else ""

    messages_str = get_buffer_string(messages)
    prompt = research_brief_prompt.format(
        messages=messages_str,
        date=datetime.now().strftime("%B %d, %Y"),
        prior_brief=prior_brief,
        feedback=feedback,
    )

    logger.info("Generating research brief from %d messages", len(messages))
    brief: ResearchBrief = await model.ainvoke([HumanMessage(content=prompt)])
    brief_str = _format_brief(brief)
    logger.info("Brief generated: %s (ready=%s)", brief.title, brief.ready_to_proceed)

    # First draft (no prior brief) must always be shown for review,
    # regardless of what the LLM sets for ready_to_proceed.
    is_first_draft = not prior_brief
    if configurable.allow_human_review and (is_first_draft or not brief.ready_to_proceed):
        # Show to user for review (first draft or revision with changes)
        logger.info("Showing brief to user for review")
        return Command(
            goto="__end__",
            update={
                "research_brief": brief_str,
                "messages": [AIMessage(content=(
                    "Here's the research plan I've drafted. "
                    "Let me know if you'd like to adjust the scope, "
                    "angle, or focus — or proceed directly.\n\n"
                    + brief_str
                ))],
            },
        )

    # User approved or review disabled — proceed to research.
    # On approval, preserve the prior brief (re-generation is non-deterministic).
    if prior_brief and brief.ready_to_proceed:
        brief_str = prior_brief
    logger.info("Proceeding to coordinator")
    return Command(
        goto="coordinator",
        update={
            "research_brief": brief_str,
        },
    )
