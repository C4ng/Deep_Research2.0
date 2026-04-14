"""Deep Research — an agentic deep research system built on LangGraph."""

__version__ = "0.1.0"

from deep_research.logging_config import setup_logging

setup_logging()

# Public API
# TODO: Consider adding a utils.py re-export facade
# if the number of internal helpers grows. For now, direct imports suffice.
from deep_research.configuration import Configuration
from deep_research.graph.graph import build_graph, deep_researcher
from deep_research.models import (
    ResearchReflection,
    ResearchResult,
    SupervisorReflection,
)
from deep_research.state import AgentState, ResearcherState, SupervisorState

__all__ = [
    "AgentState",
    "Configuration",
    "ResearchReflection",
    "ResearchResult",
    "ResearcherState",
    "SupervisorReflection",
    "SupervisorState",
    "build_graph",
    "deep_researcher",
]
