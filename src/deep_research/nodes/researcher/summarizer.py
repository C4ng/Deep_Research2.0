"""Summarizer node — compresses raw tool results into concise research notes.

Extracts all ToolMessage content from messages and compresses it
using the summarization model (cheap, no reasoning needed).
Skips LLM compression if content is short enough.
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import build_model_config, configurable_model
from deep_research.prompts import compress_research_prompt
from deep_research.state import ResearcherState

logger = logging.getLogger(__name__)

# Below this threshold, skip LLM compression — not worth the API call
COMPRESSION_THRESHOLD = 2000


async def summarize_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Compress accumulated tool results into concise research notes.

    Extracts raw ToolMessage content from messages, compresses via
    the summarization model, and writes to notes for downstream use.
    Passes accumulated_findings to the prompt so the summarizer knows
    what reflection identified as important — helps prioritize.
    """
    # Extract raw tool results from messages
    tool_results = "\n\n".join(
        m.content for m in state.get("messages", [])
        if isinstance(m, ToolMessage) and m.content
    )

    if not tool_results:
        logger.warning("No tool results to compress")
        return {"notes": ""}

    # Skip compression for short content
    if len(tool_results) < COMPRESSION_THRESHOLD:
        logger.info("Tool results under threshold (%d chars), skipping compression",
                     len(tool_results))
        return {"notes": tool_results}

    configurable = Configuration.from_runnable_config(config)

    model_config = build_model_config(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        temperature=configurable.summarization_model_temperature,
        thinking_budget=configurable.summarization_model_thinking_budget,
    )

    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    # Format accumulated findings as prioritization hints
    accumulated = state.get("accumulated_findings", [])
    findings_hint = ""
    if accumulated:
        findings_hint = (
            "\n\n<key_findings>\n"
            "The following findings were identified as important during research. "
            "Prioritize preserving information related to these:\n"
            + "\n".join(f"- {f}" for f in accumulated)
            + "\n</key_findings>"
        )

    prompt = compress_research_prompt.format(
        research_topic=state["research_topic"],
        tool_results=tool_results,
        date=datetime.now().strftime("%B %d, %Y"),
    ) + findings_hint

    logger.info("Compressing %d chars of tool results", len(tool_results))
    response = await model.ainvoke([HumanMessage(content=prompt)])
    compressed = response.text or ""

    if not compressed:
        # TODO(fallback): LLM returned empty — fall back to raw tool results
        logger.warning("Compression returned empty, falling back to raw tool results")
        return {"notes": tool_results}

    ratio = len(compressed) / len(tool_results) * 100
    logger.info("Compressed to %d chars (%.0f%% of original)", len(compressed), ratio)

    return {"notes": compressed}
