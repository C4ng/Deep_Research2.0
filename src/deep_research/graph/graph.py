"""Main research graph — wires all nodes into the pipeline.

clarify → write_brief → human_review → [simple?] → researcher / coordinator → final_report

The clarify node may exit the graph early (returning a question to the user).
The human_review node pauses for user review of the research plan.
Simple questions bypass the coordinator and go directly to a single researcher.
Complex questions go through the coordinator for multi-topic decomposition.
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.clarify import clarify_with_user
from deep_research.nodes.coordinator import coordinator_subgraph
from deep_research.nodes.report import final_report_generation
from deep_research.nodes.researcher.adapter import run_single_researcher
from deep_research.state import AgentState

logger = logging.getLogger(__name__)


async def human_review(state: AgentState, config: RunnableConfig) -> dict:
    """Pause for user review of the research plan.

    Interrupts the graph with the generated brief. The user can:
    - Approve as-is: resume with empty/approval response
    - Give feedback: resume with natural language feedback, the agent
      revises the brief accordingly

    Skipped when allow_human_review is false (programmatic use).
    """
    configurable = Configuration.from_runnable_config(config)

    if not configurable.allow_human_review:
        logger.info("Human review disabled, proceeding")
        return {}

    brief = state.get("research_brief", "")
    logger.info("Pausing for human review of research plan")

    # interrupt() pauses the graph and surfaces the brief to the caller.
    # On resume, it returns the user's feedback.
    feedback = interrupt({"brief": brief})

    # Empty or approval response — proceed with the brief as-is
    if not feedback or (isinstance(feedback, str) and feedback.strip() == ""):
        logger.info("User approved the research plan")
        return {}

    # User provided feedback — revise the brief using LLM
    feedback_str = feedback if isinstance(feedback, str) else str(feedback)
    logger.info("User provided feedback, revising research plan")

    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "temperature": configurable.research_model_temperature,
    }
    if configurable.research_model_thinking_budget is not None:
        model_config["thinking_budget"] = configurable.research_model_thinking_budget

    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable=model_config)
    )

    revision_prompt = (
        f"Here is a draft research plan:\n\n{brief}\n\n"
        f"The user reviewed it and gave this feedback:\n{feedback_str}\n\n"
        "Revise the research plan to incorporate the feedback. "
        "Keep the same format (Title, research question, Approach). "
        "Output only the revised plan, nothing else."
    )

    response = await model.ainvoke([HumanMessage(content=revision_prompt)])
    revised_brief = response.content
    logger.info("Research plan revised based on user feedback")
    return {"research_brief": revised_brief}


def route_by_complexity(state: AgentState) -> str:
    """Route simple questions to a single researcher, complex ones to the coordinator."""
    if state.get("is_simple", False):
        return "researcher"
    return "coordinator"


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
    graph.add_node("human_review", human_review)
    graph.add_node("researcher", run_single_researcher)
    graph.add_node("coordinator", coordinator_subgraph)
    graph.add_node("final_report", final_report_generation)

    graph.add_edge(START, "clarify")
    # clarify uses Command for routing (to write_brief or __end__)
    graph.add_edge("write_brief", "human_review")
    graph.add_conditional_edges("human_review", route_by_complexity)
    graph.add_edge("researcher", "final_report")
    graph.add_edge("coordinator", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer)


# Pre-compiled graph instance (no checkpointer — add one for HITL use)
deep_researcher = build_graph()
