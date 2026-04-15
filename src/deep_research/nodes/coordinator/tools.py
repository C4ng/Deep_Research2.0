"""Coordinator tools — dispatch_research dispatches a researcher subgraph.

The tool is called by the coordinator LLM to research a specific subtopic.
Each invocation runs a full researcher subgraph (search → reflect loop →
summarize) and returns a structured ResearchResult as JSON.
"""

import logging
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from deep_research.models import ResearchResult
from deep_research.nodes.researcher import researcher_subgraph
from deep_research.state import create_researcher_state

logger = logging.getLogger(__name__)


@tool
async def dispatch_research(
    topic: str,
    context: str,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Research a specific topic. Each call spawns a focused researcher.

    Args:
        topic: Focused research topic to investigate.
        context: Why this topic matters and what angle to investigate.
    """
    logger.info("dispatch_research dispatching researcher for: %s", topic)

    initial_state = create_researcher_state(f"{topic}\n\nContext: {context}")

    result = await researcher_subgraph.ainvoke(initial_state, config)

    research_result = ResearchResult(
        topic=topic,
        notes=result["notes"],
        key_findings=result["accumulated_findings"],
        knowledge_state=result.get("final_knowledge_state") or "partial",
        missing_info=result["current_gaps"],
        contradictions=result["accumulated_contradictions"],
    )

    logger.info(
        "dispatch_research completed: topic=%s, knowledge_state=%s, "
        "findings=%d, gaps=%d",
        topic,
        research_result.knowledge_state,
        len(research_result.key_findings),
        len(research_result.missing_info),
    )

    return research_result.model_dump_json()
