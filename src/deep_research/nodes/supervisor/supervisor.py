"""Supervisor node functions — LLM invocation and tool execution.

- `supervisor`: one LLM call with conduct_research tool bound
- `supervisor_tools`: execute tool calls, parse ResearchResult from responses
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.models import ResearchResult
from deep_research.nodes.supervisor.tools import conduct_research
from deep_research.prompts import supervisor_system_prompt
from deep_research.state import SupervisorState

logger = logging.getLogger(__name__)


def _format_research_results(results: list[ResearchResult]) -> str:
    """Format research results as metadata summary for the supervisor prompt.

    Includes topic, knowledge_state, key_findings, missing_info,
    contradictions — but NOT full notes (context engineering).
    """
    if not results:
        return "No research has been conducted yet."

    parts = []
    for i, r in enumerate(results, 1):
        section = [f"### Researcher {i}: {r.topic}"]
        section.append(f"Knowledge state: {r.knowledge_state}")

        if r.key_findings:
            section.append("Key findings:")
            section.extend(f"  - {f}" for f in r.key_findings)

        if r.missing_info:
            section.append("Remaining gaps:")
            section.extend(f"  - {g}" for g in r.missing_info)

        if r.contradictions:
            section.append("Contradictions:")
            section.extend(f"  - {c}" for c in r.contradictions)

        parts.append("\n".join(section))

    return "\n\n".join(parts)


async def supervisor(state: SupervisorState, config: RunnableConfig) -> dict:
    """Invoke the supervisor model with conduct_research tool bound.

    Builds context from research_brief, prior research results (metadata
    only), and reflection guidance. Each round gets fresh messages —
    prior tool-calling messages are not carried over.
    """
    configurable = Configuration.from_runnable_config(config)

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    tools = [conduct_research]
    model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    # Build system prompt with brief, prior results, and reflection
    prior_research = _format_research_results(
        state.get("research_results", [])
    )

    system_parts = [
        supervisor_system_prompt.format(
            research_brief=state["research_brief"],
            prior_research=prior_research,
            max_research_topics=getattr(configurable, "max_research_topics", 5),
            date=datetime.now().strftime("%B %d, %Y"),
        )
    ]

    last_reflection = state.get("last_supervisor_reflection", "")
    if last_reflection:
        system_parts.append(
            f"\n<prior_reflection>\n{last_reflection}\n</prior_reflection>"
        )

    is_follow_up = bool(state.get("research_results"))
    trigger = (
        "Review the prior research results and reflection, then dispatch follow-up researchers."
        if is_follow_up
        else "Decompose the research brief and dispatch researchers."
    )

    messages = [
        SystemMessage(content="\n".join(system_parts)),
        HumanMessage(content=trigger),
    ]

    logger.info("Supervisor invoking LLM (%d prior results)", len(state.get("research_results", [])))
    response = await model.ainvoke(messages)
    tool_count = len(response.tool_calls) if response.tool_calls else 0
    logger.info("Supervisor LLM responded: %d tool calls", tool_count)

    return {"messages": [response]}


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> dict:
    """Execute supervisor tool calls and parse ResearchResults.

    Each tool call runs a researcher subgraph via conduct_research.
    Parses the JSON response back into ResearchResult for state accumulation.
    """
    messages = state.get("messages", [])
    most_recent = messages[-1] if messages else None

    if not most_recent or not most_recent.tool_calls:
        return {}

    tools = [conduct_research]
    tools_by_name = {t.name: t for t in tools}

    tool_names = [tc["name"] for tc in most_recent.tool_calls if tc["name"] in tools_by_name]
    logger.info("Supervisor executing %d tool calls: %s", len(tool_names), tool_names)

    # Execute tool calls sequentially (each spawns a full researcher subgraph)
    # TODO(parallel): Switch to asyncio.gather for parallel execution in Increment 5
    tool_messages = []
    new_results = []

    for tc in most_recent.tool_calls:
        tool = tools_by_name.get(tc["name"])
        if not tool:
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool: {tc['name']}",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )
            continue

        try:
            result_json = await tool.ainvoke(tc["args"], config)
            research_result = ResearchResult.model_validate_json(result_json)
            new_results.append(research_result)

            tool_messages.append(
                ToolMessage(
                    content=result_json,
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )
        except Exception as e:
            logger.warning("Tool %s failed: %s", tc["name"], e)
            tool_messages.append(
                ToolMessage(
                    content=f"Error executing tool: {e}",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )

    logger.info("Supervisor tools completed: %d results collected", len(new_results))
    return {"messages": tool_messages, "research_results": new_results}
