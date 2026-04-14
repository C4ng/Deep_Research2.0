"""Clarification node — ask the user for clarification if the query is ambiguous.

Optional (config-gated). When clarification is needed, the graph exits
and the user re-invokes with their answer appended to messages.
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.models import ClarifyOutput
from deep_research.prompts import clarify_prompt
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if needed.

    If clarification is disabled or not needed, routes to write_brief.
    If clarification is needed, routes to __end__ with the question
    as an AIMessage — the caller re-invokes with the user's answer.
    """
    configurable = Configuration.from_runnable_config(config)

    if not configurable.allow_clarification:
        logger.info("Clarification disabled, proceeding to write_brief")
        return Command(goto="write_brief")

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    model = (
        configurable_model
        .with_structured_output(ClarifyOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    messages_str = get_buffer_string(state["messages"])
    prompt = clarify_prompt.format(
        messages=messages_str,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    logger.info("Assessing whether clarification is needed")
    result: ClarifyOutput = await model.ainvoke([HumanMessage(content=prompt)])

    if result.need_clarification:
        logger.info("Clarification needed, returning question to user")
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=result.question)]},
        )

    logger.info("No clarification needed, proceeding to write_brief")
    return Command(
        goto="write_brief",
        update={"messages": [AIMessage(content=result.verification)]},
    )
