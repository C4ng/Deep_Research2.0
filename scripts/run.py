"""Simple interactive runner for the deep research pipeline.

Usage: python scripts/run.py "your research question here"
       python scripts/run.py  (prompts for input)
"""

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deep_research.graph.graph import build_graph


async def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    if not query.strip():
        print("No question provided.")
        return

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "interactive"}}

    # Initial invocation
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=thread,
    )

    # HITL loop — graph exits to __end__ when it needs user input
    while True:
        # Show latest AI messages
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n{'='*60}")
                print(f"[Assistant]\n{msg.content}")
                print(f"{'='*60}")

        # Check if the final report is ready
        if state.get("final_report"):
            print(f"\n{'='*60}")
            print("[Final Report]")
            print(f"{'='*60}")
            print(state["final_report"])
            break

        # Get user input to continue
        user_input = input("\nYou: ").strip()
        if not user_input:
            break

        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread,
        )


if __name__ == "__main__":
    asyncio.run(main())
