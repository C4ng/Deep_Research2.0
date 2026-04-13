"""State definitions for the Deep Research graph.

Grows incrementally — this is the minimal version for Increment 1.
Increment 3 adds ResearcherState and SupervisorState.
"""

from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Top-level state flowing through the main research graph."""

    messages: Annotated[list, add_messages]
    """Conversation history (user messages + internal messages)."""

    # Fields below use last-write-wins (no reducer) — fine for Increment 1
    # where each field is written by exactly one node. Increment 3 adds
    # custom reducers when multiple researchers write to shared state.

    research_brief: str
    """Structured research brief generated from the user query."""

    notes: str
    """Accumulated research findings from the researcher."""

    final_report: str
    """The final markdown report output."""
