"""Researcher node functions — LLM invocation and tool execution.

Single responsibility per function:
- `researcher`: one LLM call with tools bound
- `researcher_tools`: execute tool calls in parallel
"""

import asyncio
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.prompts import research_system_prompt
from deep_research.state import AgentState
from deep_research.tools.registry import get_all_tools

logger = logging.getLogger(__name__)


async def _execute_tool_safely(tool, args, config):
    """Execute a single tool call with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        logger.warning("Tool %s failed: %s", getattr(tool, "name", "unknown"), e)
        return f"Error executing tool: {e}"


async def researcher(state: AgentState, config: RunnableConfig) -> dict:
    """Invoke the research model with tools bound.

    Single responsibility: one LLM call. Tool execution is handled
    by researcher_tools, routing by reflect.
    """
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)

    if not tools:
        raise ValueError(
            "No tools available for research. Check search API configuration."
        )

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    # Build messages for the LLM call.
    # System/Human messages are constructed locally (not stored in state),
    # so we always build them here. On subsequent rounds (after reflection),
    # drop prior AI/Tool messages — reflection already captured key findings
    # and the full history stays in state for compress.
    messages = state.get("messages", [])
    is_subsequent_round = state.get("research_iterations", 0) > 0

    if is_subsequent_round:
        messages = [
            m for m in messages if isinstance(m, HumanMessage)
        ]

    system_parts = [
        research_system_prompt.format(
            topic=state["research_brief"],
            date=datetime.now().strftime("%B %d, %Y"),
            max_searches_per_round=configurable.max_searches_per_round,
        )
    ]

    last_reflection = state.get("last_reflection", "")
    if last_reflection:
        system_parts.append(
            f"\n<prior_reflection>\n{last_reflection}\n</prior_reflection>"
        )

    messages = [
        SystemMessage(content="\n".join(system_parts)),
        HumanMessage(content=f"Please research the following topic:\n\n{state['research_brief']}"),
    ] + messages

    logger.info("Researcher invoking LLM (%d messages in context)", len(messages))
    response = await model.ainvoke(messages)
    tool_count = len(response.tool_calls) if response.tool_calls else 0
    logger.info("Researcher LLM responded: %d tool calls, %d chars text",
                tool_count, len(response.text) if response.text else 0)

    return {"messages": [response]}


async def researcher_tools(state: AgentState, config: RunnableConfig) -> dict:
    """Execute tool calls in parallel, then route to reflect.

    Always routes to reflect via edge — no inner loop.
    """
    messages = state.get("messages", [])
    most_recent = messages[-1] if messages else None

    # No tool calls — model produced a text response, pass through to reflect
    if not most_recent or not most_recent.tool_calls:
        return {}

    # Execute all tool calls in parallel
    tools = await get_all_tools(config)
    tools_by_name = {t.name: t for t in tools}

    tool_names = [tc["name"] for tc in most_recent.tool_calls if tc["name"] in tools_by_name]
    logger.info("Executing %d tool calls in parallel: %s", len(tool_names), tool_names)

    tool_tasks = [
        _execute_tool_safely(
            tools_by_name[tc["name"]], tc["args"], config
        )
        for tc in most_recent.tool_calls
        if tc["name"] in tools_by_name
    ]
    results = await asyncio.gather(*tool_tasks)

    tool_messages = [
        ToolMessage(
            content=str(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for result, tc in zip(results, most_recent.tool_calls)
    ]

    return {"messages": tool_messages}
