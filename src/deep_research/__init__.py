"""Deep Research — an agentic deep research system built on LangGraph."""

__version__ = "0.1.0"

from deep_research.logging_config import setup_logging

setup_logging()

# Public API
from deep_research.configuration import Configuration
from deep_research.graph.graph import build_graph, deep_researcher
from deep_research.state import AgentState

__all__ = [
    "AgentState",
    "Configuration",
    "build_graph",
    "deep_researcher",
]
