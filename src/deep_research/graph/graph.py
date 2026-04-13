"""Main research graph — wires all nodes into the pipeline.

Increment 1: write_brief → researcher_subgraph → final_report (linear)
Increment 3: adds supervisor subgraph between brief and report.
Increment 4: adds clarify_with_user before brief.
"""

from langgraph.graph import END, START, StateGraph

from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.report import final_report_generation
from deep_research.nodes.researcher import researcher_subgraph
from deep_research.state import AgentState


def build_graph() -> StateGraph:
    """Build and compile the main research graph."""
    graph = StateGraph(AgentState)

    graph.add_node("write_brief", write_research_brief)
    graph.add_node("researcher", researcher_subgraph)
    graph.add_node("final_report", final_report_generation)

    graph.add_edge(START, "write_brief")
    graph.add_edge("write_brief", "researcher")
    graph.add_edge("researcher", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()


# Pre-compiled graph instance
deep_researcher = build_graph()
