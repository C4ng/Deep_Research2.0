"""Final report generation node — synthesizes research into a markdown report."""

from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.prompts import final_report_prompt
from deep_research.state import AgentState


async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    """Generate a comprehensive markdown report from research findings.

    Takes the research brief and accumulated notes, produces a final
    structured report. No token-limit retry yet (Increment 5).
    """
    configurable = Configuration.from_runnable_config(config)

    model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(configurable={
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "temperature": configurable.research_model_temperature,
        })
    )

    prompt = final_report_prompt.format(
        brief=state["research_brief"],
        notes=state["notes"],
        date=datetime.now().strftime("%B %d, %Y"),
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])

    return {"final_report": response.content}
