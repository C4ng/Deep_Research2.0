"""Tests for graph nodes.

These tests hit real APIs — requires valid API keys in .env.
"""

import pytest
from langchain_core.messages import HumanMessage

from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.report import final_report_generation
from deep_research.tools.registry import get_all_tools


@pytest.fixture
def sample_state():
    return {
        "messages": [HumanMessage(content="What are the main causes and effects of coral reef bleaching?")],
        "research_brief": "",
        "notes": "",
        "final_report": "",
    }


@pytest.mark.asyncio
async def test_write_research_brief_returns_brief(sample_state):
    """Brief node produces a non-empty research_brief string."""
    result = await write_research_brief(sample_state, config={"configurable": {}})
    assert "research_brief" in result
    assert len(result["research_brief"]) > 0


@pytest.mark.asyncio
async def test_write_research_brief_has_structure(sample_state):
    """Brief output contains title, questions, and topics."""
    result = await write_research_brief(sample_state, config={"configurable": {}})
    brief = result["research_brief"]
    assert "Title:" in brief
    assert "Research Questions:" in brief
    assert "Key Topics:" in brief


@pytest.mark.asyncio
async def test_final_report_generation():
    """Report node produces a non-empty markdown report from notes."""
    state = {
        "messages": [],
        "research_brief": "Title: Coral Reef Bleaching\n\nResearch Questions:\n- What causes bleaching?",
        "notes": (
            "Coral bleaching occurs when water temperatures rise above 1°C "
            "over the summer maximum for 4+ weeks. The primary cause is climate "
            "change driving ocean warming. The Great Barrier Reef experienced "
            "mass bleaching in 2016, 2017, 2020, and 2022.\n"
            "Source: https://www.gbrmpa.gov.au/the-reef/coral-bleaching"
        ),
        "final_report": "",
    }
    result = await final_report_generation(state, config={"configurable": {}})
    assert "final_report" in result
    assert len(result["final_report"]) > 100
    assert "#" in result["final_report"]  # has markdown headings


@pytest.mark.asyncio
async def test_tool_registry_returns_tools():
    """Tool registry returns at least one tool with default config."""
    tools = await get_all_tools(config={"configurable": {}})
    assert len(tools) > 0
    assert tools[0].name == "tavily_search"
