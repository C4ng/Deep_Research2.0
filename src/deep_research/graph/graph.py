"""Main research graph — wires all nodes into the pipeline.

START → clarify → write_brief → coordinator → final_report → END

Both clarify and write_brief are user interaction nodes that route
themselves via Command — exiting to __end__ when user input is needed,
or proceeding to the next node when ready.

On resume (user responds after HITL pause), _route_start skips clarify
if a research brief already exists in state.
"""

import logging

from langgraph.graph import END, START, StateGraph

from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.clarify import clarify_with_user
from deep_research.nodes.coordinator import coordinator_subgraph
from deep_research.nodes.report import final_report_generation
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


def _route_start(state: AgentState) -> str:
    """Route on resume — skip clarify if we already have a brief draft."""
    if state.get("research_brief"):
        return "write_brief"
    return "clarify"


def build_graph(checkpointer=None):
    """Build and compile the main research graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
            Required for human-in-the-loop features (interrupt/resume).
            Use MemorySaver for dev/test, a persistent store for production.
    """
    graph = StateGraph(AgentState)

    graph.add_node("clarify", clarify_with_user)
    graph.add_node("write_brief", write_research_brief)
    graph.add_node("coordinator", coordinator_subgraph)
    graph.add_node("final_report", final_report_generation)

    graph.add_conditional_edges(START, _route_start, ["clarify", "write_brief"])
    # clarify routes itself via Command (to write_brief or __end__)
    # write_brief routes itself via Command (to coordinator or __end__)
    graph.add_edge("coordinator", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer)


# Pre-compiled graph instance (no checkpointer — add one for HITL use)
deep_researcher = build_graph()
