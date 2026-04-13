"""End-to-end integration test — runs the full pipeline.

Hits real APIs (Gemini + Tavily). Requires valid API keys in .env.
This is slow (~30-60s) so it's separated from unit/node tests.
"""

import pytest
from langchain_core.messages import HumanMessage

from deep_research.graph.graph import build_graph


@pytest.mark.asyncio
async def test_full_pipeline_produces_report():
    """Full pipeline: question in → markdown report out."""
    graph = build_graph()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is the current state of quantum computing in 2025?")],
            "research_brief": "",
            "notes": "",
            "final_report": "",
        },
        config={"configurable": {}},
    )

    # All state fields should be populated
    assert result["research_brief"], "research_brief should not be empty"
    assert result["notes"], "notes should not be empty"
    assert result["final_report"], "final_report should not be empty"

    # Report should be substantial markdown
    report = result["final_report"]
    assert len(report) > 500, f"Report too short ({len(report)} chars)"
    assert "#" in report, "Report should contain markdown headings"

    # Brief should have expected structure
    brief = result["research_brief"]
    assert "Title:" in brief
    assert "Research Questions:" in brief
