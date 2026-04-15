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
from deep_research.helpers.errors import NO_RESULTS_PREFIX, TOOL_ERROR_PREFIX
from deep_research.prompts import compress_research_prompt, llm_knowledge_fallback_prompt
from deep_research.state import ResearcherState

logger = logging.getLogger(__name__)

# Below this threshold, skip LLM compression — not worth the API call
COMPRESSION_THRESHOLD = 2000


def _is_junk(content: str) -> bool:
    """Check if a ToolMessage is an error or empty-result placeholder."""
    return content.startswith(TOOL_ERROR_PREFIX) or content.startswith(NO_RESULTS_PREFIX)


async def summarize_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Compress accumulated tool results into concise research notes.

    Extracts raw ToolMessage content from messages, compresses via
    the summarization model, and writes to notes for downstream use.
    Falls back to LLM knowledge generation when no real search data exists.
    """
    messages = state.get("messages", [])

    # Extract real tool results — skip errors and "no results found" messages
    tool_results = "\n\n".join(
        m.content for m in messages
        if isinstance(m, ToolMessage) and m.content and not _is_junk(m.content)
    )

    if not tool_results:
        logger.info("No usable search results — falling back to LLM knowledge")
        return await _generate_from_llm_knowledge(state, config)

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

    # TODO(compression): With 3 rounds × 3 searches, tool_results can reach ~90K chars.
    # The summarization model's max_tokens (4096) caps output at ~16K chars, so actual
    # compression ratio is dictated by output limit, not prompt instructions (observed 7%
    # instead of target 30%). Options: (1) compress per-round instead of all-at-once,
    # (2) raise max_tokens, (3) chunk-and-merge. Per-round is cleanest — avoids
    # re-compressing earlier rounds and keeps each pass within output budget.
    logger.info("Compressing %d chars of tool results", len(tool_results))
    response = await model.ainvoke([HumanMessage(content=prompt)])
    compressed = response.text or ""

    if not compressed:
        logger.warning("Compression returned empty, falling back to raw tool results")
        return {"notes": tool_results}

    ratio = len(compressed) / len(tool_results) * 100
    logger.info("Compressed to %d chars (%.0f%% of original)", len(compressed), ratio)

    return {"notes": compressed}


async def _generate_from_llm_knowledge(
    state: ResearcherState, config: RunnableConfig
) -> dict:
    """Generate research notes from LLM training knowledge when search is unavailable."""
    configurable = Configuration.from_runnable_config(config)

    model_config = build_model_config(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        temperature=configurable.research_model_temperature,
        thinking_budget=configurable.research_model_thinking_budget,
    )

    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    prompt = llm_knowledge_fallback_prompt.format(
        research_topic=state["research_topic"],
        date=datetime.now().strftime("%B %d, %Y"),
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])
    notes = response.text or ""
    logger.info("LLM knowledge fallback produced %d chars", len(notes))
    return {"notes": notes}
