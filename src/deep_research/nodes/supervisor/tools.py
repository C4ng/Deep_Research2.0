"""Supervisor tools — conduct_research dispatches a researcher subgraph.

The tool is called by the supervisor LLM to research a specific subtopic.
Each invocation runs a full researcher subgraph (search → reflect loop →
summarize) and returns a structured ResearchResult as JSON.
"""

import logging
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from deep_research.models import ResearchResult
from deep_research.nodes.researcher import researcher_subgraph

logger = logging.getLogger(__name__)


@tool
async def conduct_research(
    topic: str,
    context: str,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Research a specific topic. Each call spawns a focused researcher.

    Args:
        topic: Focused research topic to investigate.
        context: Why this topic matters and what angle to investigate.
    """
    logger.info("conduct_research dispatching researcher for: %s", topic)

    initial_state = {
        "messages": [],
        "research_topic": f"{topic}\n\nContext: {context}",
        "research_iterations": 0,
        "last_reflection": "",
        "accumulated_findings": [],
        "accumulated_contradictions": [],
        "current_gaps": [],
        "final_knowledge_state": "",
        "notes": "",
    }

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
        "conduct_research completed: topic=%s, knowledge_state=%s, "
        "findings=%d, gaps=%d",
        topic,
        research_result.knowledge_state,
        len(research_result.key_findings),
        len(research_result.missing_info),
    )

    return research_result.model_dump_json()
