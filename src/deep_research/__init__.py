"""Deep Research — an agentic deep research system built on LangGraph."""

__version__ = "0.1.0"

from deep_research.logging_config import setup_logging

setup_logging()

# Public API
from deep_research.configuration import Configuration
from deep_research.graph.graph import build_graph, deep_researcher
from deep_research.models import (
    ClarifyOutput,
    CoordinatorReflection,
    ResearchBrief,
    ResearchReflection,
    ResearchResult,
)
from deep_research.state import AgentState, CoordinatorState, ResearcherState

__all__ = [
    "AgentState",
    "ClarifyOutput",
    "Configuration",
    "CoordinatorReflection",
    "CoordinatorState",
    "ResearchBrief",
    "ResearchReflection",
    "ResearchResult",
    "ResearcherState",
    "build_graph",
    "deep_researcher",
]
