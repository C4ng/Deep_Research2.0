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

    notes: str
    """Accumulated research findings (combined from coordinator or single researcher)."""

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

    # Accumulated across reflection rounds (append reducers)
    accumulated_findings: Annotated[list[str], operator.add]
    """Key findings accumulated across all reflection rounds."""

    accumulated_contradictions: Annotated[list[str], operator.add]
    """Contradictions discovered across all reflection rounds."""

    current_gaps: list[str]
    """Current gaps remaining — overwritten each round (latest assessment)."""

    final_knowledge_state: str
    """Final knowledge assessment set by reflect when routing to summarizer."""

    notes: str
    """Summarizer output — compressed research notes."""


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

    notes: str
    """Combined notes from all researchers for the final report."""
