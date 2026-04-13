"""Researcher subgraph — searches for information on a given topic.

Split into two nodes:
- `researcher`: calls the model (one LLM invocation)
- `researcher_tools`: executes tool calls in parallel, decides routing

These alternate in a loop: researcher → researcher_tools → researcher → ...
until the model stops calling tools or max rounds are reached.

Compiled as a subgraph so the main graph treats it as a single node.
Increment 2 adds structured Reflection and system-controlled routing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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


async def researcher(state: AgentState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Invoke the research model with tools bound.

    Single responsibility: one LLM call. Tool execution and routing
    are handled by researcher_tools.
    """
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)

    if not tools:
        raise ValueError(
            "No tools available for research. Check search API configuration."
        )

    model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable={
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "temperature": configurable.research_model_temperature,
        })
    )

    # On first entry, set up system prompt + topic
    messages = state.get("messages", [])
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_prompt = research_system_prompt.format(
            topic=state["research_brief"],
            date=datetime.now().strftime("%B %d, %Y"),
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please research the following topic:\n\n{state['research_brief']}"),
        ] + messages

    response = await model.ainvoke(messages)

    return Command(
        goto="researcher_tools",
        update={"messages": [response]},
    )


async def researcher_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["researcher", "__end__"]]:
    """Execute tool calls in parallel and decide whether to continue.

    Routes back to `researcher` if more searching is needed,
    or to `__end__` when the model is done or max rounds are reached.
    """
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent = messages[-1] if messages else None

    # No tool calls — model produced a final text response
    if not most_recent or not most_recent.tool_calls:
        notes = most_recent.content if most_recent else ""
        return Command(goto="__end__", update={"notes": notes})

    # Execute all tool calls in parallel
    tools = await get_all_tools(config)
    tools_by_name = {t.name: t for t in tools}

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

    # Check if we've hit the max rounds
    tool_call_rounds = sum(
        1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls
    )
    if tool_call_rounds >= configurable.max_tool_call_rounds:
        logger.warning("Researcher hit max tool rounds (%d)", configurable.max_tool_call_rounds)
        return Command(goto="__end__", update={"messages": tool_messages, "notes": most_recent.content or ""})

    return Command(goto="researcher", update={"messages": tool_messages})


# Build the researcher subgraph
researcher_builder = StateGraph(AgentState)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_edge(START, "researcher")
researcher_subgraph = researcher_builder.compile()
