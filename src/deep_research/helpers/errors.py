"""Error detection helpers for fail-fast routing.

Provides shared constants and detection functions used by both
researcher and coordinator nodes to short-circuit on persistent
tool failures (e.g., API quota exhaustion, empty search results).
"""

from langchain_core.messages import AIMessage, ToolMessage

# Prefix used when wrapping tool exceptions as string results.
# Must match the format in researcher._execute_tool_safely().
TOOL_ERROR_PREFIX = "Error executing tool:"

# Prefix returned by BaseSearchTool.search_and_summarize() when search yields nothing.
NO_RESULTS_PREFIX = "No search results found"


def all_tools_failed(messages: list) -> bool:
    """Check if ALL tool results from the current round are errors.

    Finds the last AIMessage with tool_calls (marking the current round's
    LLM call) and checks ToolMessages after that boundary. Returns True
    only if there are ToolMessages AND every one starts with the error prefix.
    """
    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and getattr(messages[i], "tool_calls", None):
            last_ai_idx = i
            break

    current_round = messages[last_ai_idx + 1:] if last_ai_idx >= 0 else messages
    tool_messages = [
        m for m in current_round
        if isinstance(m, ToolMessage) and m.content
    ]

    if not tool_messages:
        return False

    return all(m.content.startswith(TOOL_ERROR_PREFIX) for m in tool_messages)


def no_search_results(messages: list) -> bool:
    """Check if the current round produced no usable search data.

    Returns True if all ToolMessages are either errors or "No search results
    found" messages. Covers both total tool failure and successful calls that
    returned empty result sets (e.g., DuckDuckGo fallback with no matches).
    """
    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and getattr(messages[i], "tool_calls", None):
            last_ai_idx = i
            break

    current_round = messages[last_ai_idx + 1:] if last_ai_idx >= 0 else messages
    tool_messages = [
        m for m in current_round
        if isinstance(m, ToolMessage) and m.content
    ]

    if not tool_messages:
        return False

    return all(
        m.content.startswith(TOOL_ERROR_PREFIX) or m.content.startswith(NO_RESULTS_PREFIX)
        for m in tool_messages
    )
