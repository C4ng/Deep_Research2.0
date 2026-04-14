"""End-to-end integration tests — run the full pipeline.

Hits real APIs (Gemini + Tavily). Requires valid API keys in .env.
These are slow (~2-5 min each). Run with: pytest -m integration
"""

import re
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_research.graph.graph import build_graph
from deep_research.helpers.source_store import build_source_map, reset_sources_dir, set_sources_dir

pytestmark = pytest.mark.integration


# Disable human-in-the-loop for automated tests
AUTOMATED_CONFIG = {
    "configurable": {
        "allow_clarification": False,
        "allow_human_review": False,
    }
}

# Enable clarification but disable review
CLARIFY_ONLY_CONFIG = {
    "configurable": {
        "allow_clarification": True,
        "allow_human_review": False,
    }
}

# Enable review but disable clarification
REVIEW_ONLY_CONFIG = {
    "configurable": {
        "allow_clarification": False,
        "allow_human_review": True,
    }
}


INITIAL_STATE = {
    "messages": [],
    "research_brief": "",
    "is_simple": False,
    "notes": "",
    "report_metadata": "",
    "final_report": "",
}


@pytest.mark.asyncio
async def test_full_pipeline_produces_report():
    """Full pipeline: question in → markdown report out (HITL disabled)."""
    graph = build_graph()

    result = await graph.ainvoke(
        {
            **INITIAL_STATE,
            "messages": [HumanMessage(content="What is the current state of quantum computing in 2025?")],
        },
        config=AUTOMATED_CONFIG,
    )

    # All state fields should be populated
    assert result["research_brief"], "research_brief should not be empty"
    assert result["notes"], "notes should not be empty"
    assert result["final_report"], "final_report should not be empty"

    # Report should be substantial markdown
    report = result["final_report"]
    assert len(report) > 500, f"Report too short ({len(report)} chars)"
    assert "#" in report, "Report should contain markdown headings"

    # Brief should have title, research question, and approach
    brief = result["research_brief"]
    assert "Title:" in brief
    assert "Approach:" in brief
    assert len(brief) > 50, "Brief should be a detailed research plan"


@pytest.mark.asyncio
async def test_clarify_proceeds_on_clear_question():
    """Clarification enabled with a clear question — should proceed without asking."""
    graph = build_graph()

    result = await graph.ainvoke(
        {
            **INITIAL_STATE,
            "messages": [HumanMessage(content="What is the current state of quantum computing in 2025?")],
        },
        config=CLARIFY_ONLY_CONFIG,
    )

    # Should produce a full report (clarify proceeded, review disabled)
    assert result["research_brief"], "research_brief should not be empty"
    assert result["final_report"], "final_report should not be empty"

    # Clarify should have added a verification message
    has_ai_message = any(isinstance(m, AIMessage) for m in result["messages"])
    assert has_ai_message, "Clarify should add a verification AIMessage"


@pytest.mark.asyncio
async def test_brief_review_cycle():
    """Brief review: generate → user feedback → revise → user approves → full pipeline.

    Tests the graph-exit + re-invocation pattern for human review.
    """
    graph = build_graph()

    # 1. First invoke — brief generated, exits to __end__ for review
    result1 = await graph.ainvoke(
        {
            **INITIAL_STATE,
            "messages": [HumanMessage(content="What is the current state of quantum computing in 2025?")],
        },
        config=REVIEW_ONLY_CONFIG,
    )

    # Should have a brief but no report (exited for review)
    assert result1["research_brief"], "Brief should be generated"
    assert not result1["final_report"], "Should not have a report yet (awaiting review)"
    assert "Title:" in result1["research_brief"]
    assert "Approach:" in result1["research_brief"]

    # 2. Re-invoke with user feedback
    result2 = await graph.ainvoke(
        {
            **result1,
            "messages": result1["messages"] + [
                HumanMessage(content="Focus more on quantum hardware advances and error correction."),
            ],
        },
        config=REVIEW_ONLY_CONFIG,
    )

    # Should have a revised brief, still no report (another review round)
    assert result2["research_brief"], "Revised brief should exist"
    assert not result2["final_report"], "Should not have a report yet (awaiting approval)"

    # 3. Re-invoke with approval
    result3 = await graph.ainvoke(
        {
            **result2,
            "messages": result2["messages"] + [
                HumanMessage(content="Looks good, go ahead."),
            ],
        },
        config=REVIEW_ONLY_CONFIG,
    )

    # Should have completed the full pipeline
    assert result3["research_brief"], "Final brief should exist"
    assert result3["notes"], "Research notes should not be empty"
    assert result3["final_report"], "Final report should not be empty"

    report = result3["final_report"]
    assert len(report) > 500, f"Report too short ({len(report)} chars)"
    assert "#" in report, "Report should contain markdown headings"


@pytest.mark.asyncio
async def test_citation_tracking_end_to_end():
    """Citation system: source files created, [source_id] tags in notes, IDs match store."""
    with tempfile.TemporaryDirectory() as sources_dir:
        set_sources_dir(sources_dir)
        try:
            graph = build_graph()

            result = await graph.ainvoke(
                {
                    **INITIAL_STATE,
                    "messages": [HumanMessage(content="What is the latest stable version of Python?")],
                },
                config=AUTOMATED_CONFIG,
            )

            # 1. Source files were written to the store
            source_map = build_source_map(Path(sources_dir))
            assert len(source_map) > 0, "No source files written to store"

            # 2. Every source file has url and title
            for sid, meta in source_map.items():
                assert meta["url"], f"Source {sid} missing url"
                assert meta["title"], f"Source {sid} missing title"

            # 3. Notes contain [source_id] tags
            notes = result["notes"]
            assert notes, "notes should not be empty"
            found_ids = set(re.findall(r"\[([0-9a-f]{8})\]", notes))
            # At least some source IDs should appear in notes
            # (compression may not preserve all, but should preserve some)
            assert len(found_ids) > 0, (
                f"No [source_id] tags found in notes. "
                f"Store has {len(source_map)} sources. Notes preview: {notes[:300]}"
            )

            # 4. Source IDs in notes should match files in the store
            store_ids = set(source_map.keys())
            unknown_ids = found_ids - store_ids
            assert not unknown_ids, (
                f"Notes reference source IDs not in store: {unknown_ids}"
            )
        finally:
            reset_sources_dir()
