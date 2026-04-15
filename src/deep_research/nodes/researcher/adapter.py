"""Adapter for running a single researcher from the main graph.

Maps AgentState → ResearcherState, invokes the researcher subgraph,
and maps the result back. Used for simple questions that bypass the
coordinator.
"""

import logging

from langchain_core.runnables import RunnableConfig

from deep_research.nodes.researcher import researcher_subgraph
from deep_research.state import AgentState, create_researcher_state

logger = logging.getLogger(__name__)


async def run_single_researcher(state: AgentState, config: RunnableConfig) -> dict:
    """Adapter for simple questions — runs one researcher directly.

    Maps AgentState → ResearcherState, invokes the researcher subgraph,
    and maps the result back.
    """
    logger.info("Simple question — running single researcher (no coordinator)")
    initial_state = create_researcher_state(state["research_brief"])
    result = await researcher_subgraph.ainvoke(initial_state, config)

    # Format metadata for the report node
    metadata_parts = []
    knowledge = result.get("final_knowledge_state", "")
    if knowledge:
        metadata_parts.append(f"Coverage: {knowledge}")
    if result.get("accumulated_contradictions"):
        metadata_parts.append("### Contradictions")
        metadata_parts.extend(f"- {c}" for c in result["accumulated_contradictions"])
    if result.get("current_gaps"):
        metadata_parts.append("### Persistent Gaps (searched but not found)")
        metadata_parts.extend(f"- {g}" for g in result["current_gaps"])

    return {
        "notes": result["notes"],
        "report_metadata": "\n".join(metadata_parts),
    }
