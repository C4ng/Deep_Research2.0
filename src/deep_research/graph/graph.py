"""Main research graph — wires all nodes into the pipeline.

clarify → write_brief → [simple?] → researcher / coordinator → final_report

The clarify node may exit the graph early (returning a question to the user).
Simple questions bypass the coordinator and go directly to a single researcher.
Complex questions go through the coordinator for multi-topic decomposition.
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.clarify import clarify_with_user
from deep_research.nodes.coordinator import coordinator_subgraph
from deep_research.nodes.report import final_report_generation
from deep_research.nodes.researcher import researcher_subgraph
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


async def run_single_researcher(state: AgentState, config: RunnableConfig) -> dict:
    """Adapter for simple questions — runs one researcher directly.

    Maps AgentState → ResearcherState, invokes the researcher subgraph,
    and maps the result back.
    """
    logger.info("Simple question — running single researcher (no coordinator)")
    initial_state = {
        "messages": [],
        "research_topic": state["research_brief"],
        "research_iterations": 0,
        "last_reflection": "",
        "accumulated_findings": [],
        "accumulated_contradictions": [],
        "current_gaps": [],
        "final_knowledge_state": "",
        "notes": "",
    }
    result = await researcher_subgraph.ainvoke(initial_state, config)
    return {"notes": result["notes"]}


def route_by_complexity(state: AgentState) -> str:
    """Route simple questions to a single researcher, complex ones to the coordinator."""
    if state.get("is_simple", False):
        return "researcher"
    return "coordinator"


def build_graph() -> StateGraph:
    """Build and compile the main research graph."""
    graph = StateGraph(AgentState)

    graph.add_node("clarify", clarify_with_user)
    graph.add_node("write_brief", write_research_brief)
    graph.add_node("researcher", run_single_researcher)
    graph.add_node("coordinator", coordinator_subgraph)
    graph.add_node("final_report", final_report_generation)

    graph.add_edge(START, "clarify")
    # clarify uses Command for routing (to write_brief or __end__)
    graph.add_conditional_edges("write_brief", route_by_complexity)
    graph.add_edge("researcher", "final_report")
    graph.add_edge("coordinator", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()


# Pre-compiled graph instance
deep_researcher = build_graph()
