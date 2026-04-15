"""State definitions for the Deep Research graph.

AgentState: top-level state for the main graph.
ResearcherState: isolated state for the researcher subgraph.
CoordinatorState: state for the coordinator subgraph.
"""

import operator
from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict

from deep_research.models import ResearchResult


class AgentState(TypedDict):
    """Top-level state flowing through the main research graph."""

    messages: Annotated[list, add_messages]
    """Conversation history (user messages + internal messages)."""

    research_brief: str
    """Structured research brief generated from the user query."""

    is_simple: bool
    """Whether the question is simple enough for a single researcher (no coordinator)."""

    notes: str
    """Accumulated research findings (combined from coordinator or single researcher)."""

    report_metadata: str
    """Formatted research metadata for the report: contradictions, gaps, knowledge states."""

    final_report: str
    """The final markdown report output."""


class ResearcherState(TypedDict):
    """State for the researcher subgraph — isolated per researcher instance.

    Each researcher gets its own state with accumulator fields that persist
    structured knowledge across reflection rounds.
    """

    messages: Annotated[list, add_messages]
    """Tool-calling messages (AIMessage + ToolMessage pairs)."""

    research_topic: str
    """Assigned subtopic to research (not the full brief)."""

    research_iterations: int
    """Reflection cycle count, incremented by the reflect node."""

    last_reflection: str
    """Formatted reflection guidance for the next round (overwrite)."""

    # Accumulated across reflection rounds
    accumulated_findings: Annotated[list[str], operator.add]
    """Key findings accumulated across all reflection rounds (append reducer)."""

    accumulated_contradictions: list[str]
    """Contradictions — full canonical list, overwritten each round by the LLM."""

    current_gaps: list[str]
    """Current gaps remaining — overwritten each round (latest assessment)."""

    final_knowledge_state: str
    """Final knowledge assessment set by reflect when routing to summarizer."""

    notes: str
    """Summarizer output — compressed research notes."""


def create_researcher_state(topic: str) -> dict:
    """Create initial state dict for a researcher subgraph invocation."""
    return {
        "messages": [],
        "research_topic": topic,
        "research_iterations": 0,
        "last_reflection": "",
        "accumulated_findings": [],
        "accumulated_contradictions": [],
        "current_gaps": [],
        "final_knowledge_state": "",
        "notes": "",
    }


class CoordinatorState(TypedDict):
    """State for the coordinator subgraph.

    The coordinator decomposes a research brief into subtopics, dispatches
    researchers, collects results, reflects on cross-topic completeness,
    and decides whether follow-up research is needed.
    """

    messages: Annotated[list, add_messages]
    """Tool-calling messages (coordinator LLM + dispatch_research tool results)."""

    research_brief: str
    """The full research brief from write_brief."""

    research_results: Annotated[list[ResearchResult], operator.add]
    """Results from all dispatched researchers (append reducer)."""

    last_coordinator_reflection: str
    """Formatted reflection guidance for the next coordinator round (overwrite)."""

    coordinator_iterations: int
    """Number of coordinator reflection cycles completed."""

    latest_round_result_count: int
    """Number of successful researcher results from the most recent dispatch round."""

    notes: str
    """Combined notes from all researchers for the final report."""

    report_metadata: str
    """Formatted research metadata for the report: contradictions, gaps, knowledge states."""
