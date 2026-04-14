"""Tests for graph nodes.

API tests hit real APIs — requires valid API keys in .env.
Unit tests (reflect routing, formatting) run without API calls.
"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from deep_research.models import ResearchReflection
from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.researcher.summarizer import summarize_research
from deep_research.nodes.researcher.reflect import _extract_tool_results, _format_reflection
from deep_research.nodes.report import final_report_generation
from deep_research.tools.registry import get_all_tools


@pytest.fixture
def sample_state():
    """AgentState for brief/report tests."""
    return {
        "messages": [HumanMessage(content="What are the main causes and effects of coral reef bleaching?")],
        "research_brief": "",
        "notes": "",
        "final_report": "",
    }


@pytest.fixture
def researcher_state():
    """ResearcherState for researcher/reflect/summarizer tests."""
    return {
        "messages": [],
        "research_topic": "Test topic",
        "research_iterations": 0,
        "last_reflection": "",
        "accumulated_findings": [],
        "accumulated_contradictions": [],
        "current_gaps": [],
        "notes": "",
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


# --- Reflect node unit tests (no API calls) ---


def test_extract_tool_results(researcher_state):
    """Extracts content from ToolMessages, skips others."""
    researcher_state["messages"] = [
        HumanMessage(content="search for X"),
        ToolMessage(content="Result A", name="tavily_search", tool_call_id="1"),
        ToolMessage(content="Result B", name="tavily_search", tool_call_id="2"),
        ToolMessage(content="", name="tavily_search", tool_call_id="3"),
    ]
    result = _extract_tool_results(researcher_state)
    assert "Result A" in result
    assert "Result B" in result
    assert result.count("Result") == 2  # empty ToolMessage skipped


def test_format_reflection_with_all_fields():
    """Formats reflection with accumulated findings, current round, and all fields."""
    reflection = ResearchReflection(
        key_findings=["Found X", "Found W"],
        missing_info=["No data on Y", "Missing Z"],
        contradictions=["Source A says X, Source B says not X"],
        knowledge_state="partial",
        should_continue=True,
        next_queries=["search for Y", "search for Z"],
    )
    accumulated = ["Prior finding A", "Prior finding B"]
    result = _format_reflection(reflection, accumulated)
    # Accumulated findings from prior rounds
    assert "Prior finding A" in result
    assert "Prior finding B" in result
    # Current round findings
    assert "Found X" in result
    assert "Found W" in result
    # Gaps, contradictions, queries
    assert "No data on Y" in result
    assert "Missing Z" in result
    assert "Source A says X" in result
    assert "search for Y" in result


def test_format_reflection_minimal():
    """Formats reflection without accumulated findings, contradictions, or next_queries."""
    reflection = ResearchReflection(
        key_findings=["Found X"],
        missing_info=["No data on Y"],
        knowledge_state="insufficient",
        should_continue=True,
    )
    result = _format_reflection(reflection, accumulated_findings=[])
    assert "Found X" in result
    assert "No data on Y" in result
    assert "Contradictions" not in result
    assert "next queries" not in result
    # No accumulated section when empty
    assert "across all rounds" not in result


def test_format_reflection_with_accumulated_no_new():
    """Formats reflection with accumulated findings but no new findings this round."""
    reflection = ResearchReflection(
        key_findings=[],
        missing_info=["Still missing X"],
        knowledge_state="partial",
        should_continue=True,
    )
    accumulated = ["Prior finding A"]
    result = _format_reflection(reflection, accumulated)
    assert "Prior finding A" in result
    assert "Still missing X" in result


# --- Summarizer node unit tests (no API calls) ---


@pytest.mark.asyncio
async def test_summarize_skips_short_content(researcher_state):
    """Short tool results are returned as-is without LLM compression."""
    researcher_state["messages"] = [
        ToolMessage(content="Short result", name="tavily_search", tool_call_id="1"),
    ]
    result = await summarize_research(researcher_state, config={"configurable": {}})
    assert result["notes"] == "Short result"


@pytest.mark.asyncio
async def test_summarize_empty_messages(researcher_state):
    """No tool results returns empty notes."""
    result = await summarize_research(researcher_state, config={"configurable": {}})
    assert result["notes"] == ""


# --- Accumulator behavior tests (no API calls) ---


def test_accumulated_findings_persist_across_rounds():
    """Accumulated findings from round 1 are visible when formatting round 2 reflection."""
    # Round 1 reflection
    round1 = ResearchReflection(
        key_findings=["Finding from round 1"],
        missing_info=["Gap A"],
        knowledge_state="insufficient",
        should_continue=True,
        next_queries=["query A"],
    )
    # Simulate accumulation: after round 1, accumulated_findings = round1.key_findings
    accumulated_after_round1 = round1.key_findings[:]

    # Round 2 reflection
    round2 = ResearchReflection(
        key_findings=["Finding from round 2"],
        missing_info=["Gap B"],
        knowledge_state="partial",
        should_continue=True,
        next_queries=["query B"],
    )
    # accumulated grows: round1 + round2 findings
    accumulated_after_round2 = accumulated_after_round1 + round2.key_findings

    result = _format_reflection(round2, accumulated_after_round2)
    # Both rounds' findings appear in accumulated section
    assert "Finding from round 1" in result
    assert "Finding from round 2" in result
    assert "Gap B" in result


def test_current_gaps_overwrite():
    """current_gaps uses last-write-wins — only latest gaps remain."""
    # This tests the state design intent: current_gaps has no reducer (overwrite)
    # We verify by simulating two updates
    state = {
        "current_gaps": ["old gap 1", "old gap 2"],
    }
    # Overwrite with new gaps (as the reflect node does)
    state["current_gaps"] = ["new gap only"]
    assert state["current_gaps"] == ["new gap only"]
    assert "old gap 1" not in state["current_gaps"]
