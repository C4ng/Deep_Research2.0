"""Main research graph — wires all nodes into the pipeline.

write_brief → coordinator_subgraph → final_report

The coordinator decomposes the brief into subtopics, dispatches researchers,
reflects on cross-topic completeness, and merges notes for the report.
"""

from langgraph.graph import END, START, StateGraph

from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.coordinator import coordinator_subgraph
from deep_research.nodes.report import final_report_generation
from deep_research.state import AgentState


def build_graph() -> StateGraph:
    """Build and compile the main research graph."""
    graph = StateGraph(AgentState)

    graph.add_node("write_brief", write_research_brief)
    graph.add_node("coordinator", coordinator_subgraph)
    graph.add_node("final_report", final_report_generation)

    graph.add_edge(START, "write_brief")
    graph.add_edge("write_brief", "coordinator")
    graph.add_edge("coordinator", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()


# Pre-compiled graph instance
deep_researcher = build_graph()
