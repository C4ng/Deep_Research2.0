"""Coordinator subgraph — decomposes research brief into subtopics.

Three nodes:
- `coordinator`: calls the model to decompose brief and dispatch researchers
- `coordinator_tools`: executes dispatch_research tool calls
- `coordinator_reflect`: assesses cross-topic completeness, routes continue or exit

Flow: coordinator → coordinator_tools → coordinator_reflect → (coordinator | END)

On exit, merges all researcher notes with topic headers into `notes`.
"""

from langgraph.graph import START, StateGraph

from deep_research.nodes.coordinator.coordinator import coordinator, coordinator_tools
from deep_research.nodes.coordinator.reflect import coordinator_reflect
from deep_research.state import CoordinatorState

coordinator_builder = StateGraph(CoordinatorState)
coordinator_builder.add_node("coordinator", coordinator)
coordinator_builder.add_node("coordinator_tools", coordinator_tools)
coordinator_builder.add_node("coordinator_reflect", coordinator_reflect)
coordinator_builder.add_edge(START, "coordinator")
coordinator_builder.add_edge("coordinator", "coordinator_tools")
coordinator_builder.add_edge("coordinator_tools", "coordinator_reflect")
# coordinator_reflect uses Command for conditional routing (coordinator | __end__)
coordinator_subgraph = coordinator_builder.compile()

__all__ = ["coordinator_subgraph"]
