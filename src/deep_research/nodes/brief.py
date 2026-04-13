"""Brief generation node — transforms user query into a structured research brief."""

from datetime import datetime

from langchain_core.messages import HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.models import ResearchBrief
from deep_research.prompts import research_brief_prompt
from deep_research.state import AgentState


async def write_research_brief(state: AgentState, config: RunnableConfig) -> dict:
    """Generate a structured research brief from the user's messages.

    Reads the conversation history, calls the research model with
    structured output to produce a ResearchBrief, and returns it
    as a string for downstream nodes.
    """
    configurable = Configuration.from_runnable_config(config)

    model = (
        configurable_model
        .with_structured_output(ResearchBrief)
        .with_retry(stop_after_attempt=3)
        .with_config(configurable={
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "temperature": configurable.research_model_temperature,
        })
    )

    messages_str = get_buffer_string(state["messages"])
    prompt = research_brief_prompt.format(
        messages=messages_str,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    brief: ResearchBrief = await model.ainvoke([HumanMessage(content=prompt)])

    # Store as formatted string for downstream consumption
    brief_str = (
        f"Title: {brief.title}\n\n"
        f"Research Questions:\n"
        + "\n".join(f"- {q}" for q in brief.research_questions)
        + f"\n\nKey Topics:\n"
        + "\n".join(f"- {t}" for t in brief.key_topics)
    )

    # TODO: Consider whether downstream nodes (researcher, report) should also
    # receive the original user messages alongside the brief. The brief is the
    # model's structured interpretation — it may lose nuance, tone, or implicit
    # constraints from the original question (e.g., audience, perspective, depth).
    # For now, researcher uses only the brief for context isolation (Increment 3).
    return {"research_brief": brief_str}
