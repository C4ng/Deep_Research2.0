"""Tests for graph nodes.

These tests hit real APIs — requires valid API keys in .env.
"""

import pytest
from langchain_core.messages import HumanMessage

from deep_research.nodes.brief import write_research_brief
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
async def test_tool_registry_returns_tools():
    """Tool registry returns at least one tool with default config."""
    tools = await get_all_tools(config={"configurable": {}})
    assert len(tools) > 0
    assert tools[0].name == "tavily_search"
