"""Supervisor subgraph — decomposes research brief into subtopics.

Three nodes:
- `supervisor`: calls the model to decompose brief and dispatch researchers
- `supervisor_tools`: executes conduct_research tool calls
- `supervisor_reflect`: assesses cross-topic completeness, routes continue or exit

Flow: supervisor → supervisor_tools → supervisor_reflect → (supervisor | END)

On exit, merges all researcher notes with topic headers into `notes`.
"""

from langgraph.graph import END, START, StateGraph

from deep_research.nodes.supervisor.reflect import supervisor_reflect
from deep_research.nodes.supervisor.supervisor import supervisor, supervisor_tools
from deep_research.state import SupervisorState

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("supervisor_reflect", supervisor_reflect)
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("supervisor_tools", "supervisor_reflect")
# supervisor_reflect uses Command for conditional routing (supervisor | __end__)
supervisor_subgraph = supervisor_builder.compile()

__all__ = ["supervisor_subgraph"]
