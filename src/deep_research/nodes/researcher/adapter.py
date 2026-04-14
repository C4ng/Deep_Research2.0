"""Adapter for running a single researcher from the main graph.

Maps AgentState → ResearcherState, invokes the researcher subgraph,
and maps the result back. Used for simple questions that bypass the
coordinator.
"""

import logging

from langchain_core.runnables import RunnableConfig

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
