"""Map LangGraph stream updates to UI events.

Each node's state update is mapped to zero or more UI events
categorized as 'chat' or 'activity' channel events.
"""

from langchain_core.messages import AIMessage, ToolMessage


def map_stream_event(node_name: str, update: dict) -> list[dict]:
    """Map a graph node's state update to UI events.

    Args:
        node_name: The graph node that produced the update.
        update: The state update dict from that node.

    Returns:
        List of UI events, each with channel/type/data.
    """
    events = []

    # --- Chat events: AI messages from user-facing nodes ---
    if node_name in ("clarify", "write_brief"):
        messages = update.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                events.append({
                    "channel": "chat",
                    "type": "ai_message",
                    "data": {"content": msg.content, "node": node_name},
                })

    # --- Activity events from coordinator ---
    if node_name == "coordinator":
        messages = update.get("messages", [])
        # Coordinator's AI message contains tool_calls = researcher dispatches
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                topics = []
                for tc in msg.tool_calls:
                    if tc["name"] == "dispatch_research":
                        topic = tc["args"].get("topic", "")
                        if topic:
                            topics.append(topic)
                if topics:
                    events.append({
                        "channel": "activity",
                        "type": "dispatch",
                        "data": {"topics": topics, "count": len(topics)},
                    })

            # ToolMessages = researcher results returning
            if isinstance(msg, ToolMessage) and msg.content:
                events.append({
                    "channel": "activity",
                    "type": "researcher_complete",
                    "data": {"name": msg.name or "researcher"},
                })

    # --- Activity: coordinator tools collected results ---
    if node_name == "coordinator_tools":
        results = update.get("research_results", [])
        if results:
            summaries = []
            for r in results:
                summaries.append({
                    "topic": r.topic,
                    "knowledge_state": r.knowledge_state,
                    "findings_count": len(r.key_findings),
                    "gaps_count": len(r.missing_info),
                })
            events.append({
                "channel": "activity",
                "type": "results_collected",
                "data": {"results": summaries, "count": len(results)},
            })

    # --- Activity: coordinator reflection ---
    if node_name == "coordinator_reflect":
        notes = update.get("notes", "")
        iterations = update.get("coordinator_iterations", 0)
        if notes:
            events.append({
                "channel": "activity",
                "type": "coordinator_done",
                "data": {"iteration": iterations},
            })

    # --- Activity: final report generation ---
    if node_name == "final_report":
        report = update.get("final_report", "")
        if report:
            events.append({
                "channel": "activity",
                "type": "report_generated",
                "data": {"length": len(report)},
            })

    return events
