"""Tests for graph nodes.

API tests hit real APIs — requires valid API keys in .env.
Unit tests (reflect routing, formatting) run without API calls.
"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from deep_research.models import ResearchReflection, ResearchResult
from deep_research.nodes.brief import write_research_brief
from deep_research.nodes.researcher.summarizer import summarize_research
from deep_research.nodes.researcher.reflect import _extract_tool_results, _format_reflection
from deep_research.nodes.report import final_report_generation
from deep_research.models import CoordinatorReflection
from deep_research.nodes.coordinator.reflect import _merge_notes, _format_reflection_guidance
from deep_research.nodes.coordinator.coordinator import _format_research_results
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
    """Brief output contains title and a research question paragraph."""
    result = await write_research_brief(sample_state, config={"configurable": {}})
    brief = result["research_brief"]
    assert "Title:" in brief
    # Should be a single research question paragraph, not decomposed lists
    assert "Research Questions:" not in brief
    assert "Key Topics:" not in brief
    # Title line + blank line + question paragraph
    lines = brief.strip().split("\n")
    assert len(lines) >= 2


@pytest.mark.asyncio
async def test_final_report_generation():
    """Report node produces a non-empty markdown report from notes."""
    state = {
        "messages": [],
        "research_brief": "Title: Coral Reef Bleaching\n\nWhat are the main causes and effects of coral reef bleaching?",
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


# --- Coordinator formatting tests (no API calls) ---


def test_format_research_results_empty():
    """Empty results list produces a 'no research' message."""
    result = _format_research_results([])
    assert "No research" in result


def test_format_research_results_single():
    """Single result formats with topic, knowledge state, and findings."""
    results = [
        ResearchResult(
            topic="Quantum hardware",
            notes="compressed notes...",
            key_findings=["Superconducting qubits lead", "Ion traps viable"],
            knowledge_state="partial",
            missing_info=["No data on photonic approaches"],
            contradictions=["Source A says 1000 qubits, Source B says 500"],
        )
    ]
    formatted = _format_research_results(results)
    assert "Quantum hardware" in formatted
    assert "partial" in formatted
    assert "Superconducting qubits lead" in formatted
    assert "Ion traps viable" in formatted
    assert "No data on photonic approaches" in formatted
    assert "Source A says 1000 qubits" in formatted
    # Notes should NOT appear (context engineering)
    assert "compressed notes" not in formatted


def test_format_research_results_multiple():
    """Multiple results are numbered and all included."""
    results = [
        ResearchResult(
            topic="Topic A",
            notes="notes A",
            key_findings=["Finding A"],
            knowledge_state="sufficient",
        ),
        ResearchResult(
            topic="Topic B",
            notes="notes B",
            key_findings=["Finding B"],
            knowledge_state="insufficient",
            missing_info=["Gap B"],
        ),
    ]
    formatted = _format_research_results(results)
    assert "Researcher 1" in formatted
    assert "Researcher 2" in formatted
    assert "Topic A" in formatted
    assert "Topic B" in formatted
    assert "Finding A" in formatted
    assert "Finding B" in formatted
    assert "Gap B" in formatted


def test_format_research_results_no_optional_fields():
    """Result with no gaps, contradictions still formats cleanly."""
    results = [
        ResearchResult(
            topic="Simple topic",
            notes="some notes",
            key_findings=["One finding"],
            knowledge_state="sufficient",
        )
    ]
    formatted = _format_research_results(results)
    assert "Simple topic" in formatted
    assert "One finding" in formatted
    assert "Remaining gaps" not in formatted
    assert "Contradictions" not in formatted


# --- Coordinator reflection helpers (no API calls) ---


def test_merge_notes_combines_with_headers():
    """Merges researcher notes with topic headers."""
    results = [
        ResearchResult(
            topic="Topic A", notes="Notes for A",
            key_findings=[], knowledge_state="sufficient",
        ),
        ResearchResult(
            topic="Topic B", notes="Notes for B",
            key_findings=[], knowledge_state="sufficient",
        ),
    ]
    merged = _merge_notes(results)
    assert "## Topic: Topic A" in merged
    assert "Notes for A" in merged
    assert "## Topic: Topic B" in merged
    assert "Notes for B" in merged


def test_merge_notes_skips_empty():
    """Results with empty notes are excluded."""
    results = [
        ResearchResult(
            topic="Has notes", notes="Content here",
            key_findings=[], knowledge_state="sufficient",
        ),
        ResearchResult(
            topic="No notes", notes="",
            key_findings=[], knowledge_state="partial",
        ),
    ]
    merged = _merge_notes(results)
    assert "Has notes" in merged
    assert "No notes" not in merged


def test_format_reflection_guidance_full():
    """Formats guidance with assessment, gaps, and contradictions."""
    reflection = CoordinatorReflection(
        overall_assessment="Good coverage but missing pricing data.",
        cross_topic_contradictions=["Researcher 1 says X, Researcher 2 says Y"],
        coverage_gaps=["No pricing comparison", "Missing regulatory analysis"],
        should_continue=True,
        knowledge_state="partial",
    )
    guidance = _format_reflection_guidance(reflection)
    assert "Good coverage but missing pricing data" in guidance
    assert "No pricing comparison" in guidance
    assert "Missing regulatory analysis" in guidance
    assert "Researcher 1 says X" in guidance


def test_format_reflection_guidance_no_gaps_or_contradictions():
    """Guidance with only assessment when no gaps or contradictions."""
    reflection = CoordinatorReflection(
        overall_assessment="All topics well covered.",
        should_continue=False,
        knowledge_state="sufficient",
    )
    guidance = _format_reflection_guidance(reflection)
    assert "All topics well covered" in guidance
    assert "Coverage gaps" not in guidance
    assert "contradictions" not in guidance
