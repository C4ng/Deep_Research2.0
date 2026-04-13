"""Researcher subgraph — searches for information on a given topic.

Four nodes in a loop:
- `researcher`: calls the model (one LLM invocation with tools)
- `researcher_tools`: executes tool calls in parallel
- `reflect`: structured reflection — decides continue or stop
- `compress`: compresses raw tool results into concise notes on exit

Flow: researcher → researcher_tools → reflect → (researcher | compress → END)

Compiled as a subgraph so the main graph treats it as a single node.
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langsmith.run_helpers import get_current_run_tree

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.nodes.compress import compress_research
from deep_research.nodes.reflect import reflect
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
    are handled by researcher_tools and reflect.
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

    # On first entry, set up system prompt + topic
    messages = state.get("messages", [])
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_parts = [
            research_system_prompt.format(
                topic=state["research_brief"],
                date=datetime.now().strftime("%B %d, %Y"),
            )
        ]

        # Include reflection guidance if available (from prior reflect round)
        last_reflection = state.get("last_reflection", "")
        if last_reflection:
            system_parts.append(
                f"\n<prior_reflection>\n{last_reflection}\n</prior_reflection>"
            )

        messages = [
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=f"Please research the following topic:\n\n{state['research_brief']}"),
        ] + messages
    else:
        # Subsequent rounds — append reflection guidance if available
        last_reflection = state.get("last_reflection", "")
        if last_reflection:
            messages = messages + [
                HumanMessage(content=f"Based on reflection, here is guidance for this round:\n\n{last_reflection}"),
            ]

    logger.info("Researcher invoking LLM (%d messages in context)", len(messages))
    response = await model.ainvoke(messages)
    tool_count = len(response.tool_calls) if response.tool_calls else 0
    logger.info("Researcher LLM responded: %d tool calls, %d chars text",
                tool_count, len(response.text) if response.text else 0)

    return Command(
        goto="researcher_tools",
        update={"messages": [response]},
    )


async def researcher_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["researcher", "reflect"]]:
    """Execute tool calls in parallel, then route to reflect.

    Routes back to researcher only if the model called tools and
    max_tool_call_rounds has not been reached (inner tool-calling loop).
    Otherwise routes to reflect for assessment.
    """
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])
    most_recent = messages[-1] if messages else None

    # No tool calls — model produced a final text response, let reflect assess
    if not most_recent or not most_recent.tool_calls:
        return Command(goto="reflect")

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

    # Check if we've hit the max tool call rounds (inner loop safety net)
    tool_call_rounds = sum(
        1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls
    )
    if tool_call_rounds >= configurable.max_tool_call_rounds:
        logger.warning("Hit max tool call rounds (%d), routing to reflect",
                       configurable.max_tool_call_rounds)
        return Command(goto="reflect", update={"messages": tool_messages})

    # More tool calls possible — loop back to researcher for another LLM call
    return Command(goto="researcher", update={"messages": tool_messages})


# Build the researcher subgraph
researcher_builder = StateGraph(AgentState)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("reflect", reflect)
researcher_builder.add_node("compress", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress", END)
researcher_subgraph = researcher_builder.compile()
