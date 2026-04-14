"""Researcher subgraph — searches for information on a given topic.

Four nodes in a loop:
- `researcher`: calls the model (one LLM invocation with tools)
- `researcher_tools`: executes tool calls in parallel
- `reflect`: structured reflection — decides continue or stop
- `summarize`: compresses raw tool results into concise notes on exit

Flow: researcher → researcher_tools → reflect → (researcher | summarize → END)

No inner loop — each research round gets one LLM call, then reflects.
The model can call multiple tools in parallel within that one response.

Compiled as a subgraph so the main graph treats it as a single node.
"""

from langgraph.graph import END, START, StateGraph

from deep_research.nodes.researcher.reflect import reflect
from deep_research.nodes.researcher.researcher import researcher, researcher_tools
from deep_research.nodes.researcher.summarizer import summarize_research
from deep_research.state import ResearcherState

# Build the researcher subgraph
researcher_builder = StateGraph(ResearcherState)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("reflect", reflect)
researcher_builder.add_node("summarize", summarize_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("researcher", "researcher_tools")
researcher_builder.add_edge("researcher_tools", "reflect")
researcher_builder.add_edge("summarize", END)
researcher_subgraph = researcher_builder.compile()

__all__ = ["researcher_subgraph"]
