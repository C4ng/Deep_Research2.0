"""Unit tests for the file-based source store.

Pure logic tests — no API calls, no LLM. Tests source ID generation,
file write/read/dedup, source map building, and format_results output.
"""

import tempfile
from pathlib import Path

import pytest

from deep_research.helpers.source_store import (
    build_source_map,
    format_source_map_for_prompt,
    generate_source_id,
    read_source,
    resolve_citations,
    write_source,
)
from deep_research.models import SearchResult
from deep_research.tools.search.base import BaseSearchTool


@pytest.fixture
def sources_dir():
    """Temporary directory for source files, cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# --- generate_source_id ---


def test_generate_source_id_deterministic():
    """Same URL always produces the same source ID."""
    url = "https://example.com/article"
    assert generate_source_id(url) == generate_source_id(url)


def test_generate_source_id_different_urls():
    """Different URLs produce different source IDs."""
    id1 = generate_source_id("https://example.com/a")
    id2 = generate_source_id("https://example.com/b")
    assert id1 != id2


def test_generate_source_id_length():
    """Source ID is 8 hex characters."""
    sid = generate_source_id("https://example.com")
    assert len(sid) == 8
    assert all(c in "0123456789abcdef" for c in sid)


# --- write_source / read_source ---


def test_write_source_creates_file(sources_dir):
    """Writing a source creates a file with correct metadata and content."""
    sid = generate_source_id("https://example.com/test")
    written = write_source(
        sources_dir, sid,
        "https://example.com/test", "Test Page",
        "Summary text", "Key excerpt",
    )
    assert written is True
    assert (sources_dir / f"{sid}.md").exists()


def test_write_source_dedup(sources_dir):
    """Second write for same source ID returns False, file unchanged."""
    sid = generate_source_id("https://example.com/test")
    write_source(
        sources_dir, sid,
        "https://example.com/test", "Test Page",
        "Original summary", "Original excerpt",
    )
    original_content = (sources_dir / f"{sid}.md").read_text()

    written = write_source(
        sources_dir, sid,
        "https://example.com/test", "Test Page",
        "Different summary", "Different excerpt",
    )
    assert written is False
    assert (sources_dir / f"{sid}.md").read_text() == original_content


def test_read_source_exists(sources_dir):
    """Reading an existing source returns correct metadata and content."""
    sid = generate_source_id("https://example.com/test")
    write_source(
        sources_dir, sid,
        "https://example.com/test", "Test Page",
        "Summary text", "Key excerpt",
    )

    source = read_source(sources_dir, sid)
    assert source is not None
    assert source["source_id"] == sid
    assert source["url"] == "https://example.com/test"
    assert source["title"] == "Test Page"
    assert "Summary text" in source["content"]
    assert "Key excerpt" in source["content"]


def test_read_source_not_found(sources_dir):
    """Reading a non-existent source returns None."""
    assert read_source(sources_dir, "nonexistent") is None


# --- build_source_map ---


def test_build_source_map(sources_dir):
    """Builds a mapping of all sources in the directory."""
    sid1 = generate_source_id("https://example.com/a")
    sid2 = generate_source_id("https://example.com/b")
    write_source(sources_dir, sid1, "https://example.com/a", "Page A", "s1", "e1")
    write_source(sources_dir, sid2, "https://example.com/b", "Page B", "s2", "e2")

    smap = build_source_map(sources_dir)
    assert len(smap) == 2
    assert smap[sid1]["url"] == "https://example.com/a"
    assert smap[sid1]["title"] == "Page A"
    assert smap[sid2]["url"] == "https://example.com/b"
    assert smap[sid2]["title"] == "Page B"


def test_build_source_map_empty(sources_dir):
    """Empty directory returns empty map."""
    assert build_source_map(sources_dir) == {}


def test_build_source_map_nonexistent():
    """Non-existent directory returns empty map."""
    assert build_source_map(Path("/nonexistent/path")) == {}


# --- _format_results uses [source_id] ---


def test_format_results_uses_source_ids():
    """Formatted output uses [source_id] tags, not SOURCE N."""
    results = [
        SearchResult(url="https://example.com/a", title="Page A", content="snippet A"),
        SearchResult(url="https://example.com/b", title="Page B", content="snippet B"),
    ]
    summaries = ["Summary A", "Summary B"]
    source_ids = ["abc12345", "def67890"]

    output = BaseSearchTool._format_results(results, summaries, source_ids)

    assert "[abc12345]" in output
    assert "[def67890]" in output
    assert "SOURCE 1" not in output
    assert "SOURCE 2" not in output
    assert "Page A" in output
    assert "Page B" in output
    assert "URL: https://example.com/a" in output


# --- format_source_map_for_prompt ---


def test_format_source_map_for_prompt():
    """Formats source map as readable lookup table."""
    source_map = {
        "abc12345": {"url": "https://example.com/a", "title": "Page A"},
        "def67890": {"url": "https://example.com/b", "title": "Page B"},
    }
    output = format_source_map_for_prompt(source_map)
    assert "[abc12345] Page A — https://example.com/a" in output
    assert "[def67890] Page B — https://example.com/b" in output


def test_format_source_map_empty():
    """Empty source map returns placeholder text."""
    assert format_source_map_for_prompt({}) == "No sources available."


# --- resolve_citations ---


@pytest.fixture
def sample_source_map():
    """Source map with three sources for citation resolution tests."""
    return {
        "abc12345": {"url": "https://example.com/a", "title": "Page A"},
        "def67890": {"url": "https://example.com/b", "title": "Page B"},
        "aaa11111": {"url": "https://example.com/c", "title": "Page C"},
    }


def test_resolve_single_citation(sample_source_map):
    """Single [source_id] replaced with [1]."""
    report = "The market grew [abc12345] significantly."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "[1]" in resolved
    assert "[abc12345]" not in resolved
    assert not warnings


def test_resolve_multi_citation(sample_source_map):
    """Comma-separated [id1, id2] replaced with [1, 2]."""
    report = "Multiple sources agree [abc12345, def67890] on this."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "[1, 2]" in resolved
    assert "[abc12345" not in resolved
    assert not warnings


def test_resolve_repeated_citation(sample_source_map):
    """Same ID cited twice gets the same number."""
    report = "First [abc12345] and again [abc12345]."
    resolved, warnings = resolve_citations(report, sample_source_map)
    # Body has two [1] references, Sources section has one more
    body = resolved.split("## Sources")[0]
    assert body.count("[1]") == 2
    assert "[abc12345]" not in resolved


def test_resolve_ordering_by_first_appearance(sample_source_map):
    """Sequential numbers assigned in order of first appearance."""
    report = "First [def67890] then [abc12345]."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "[1]" in resolved  # def67890 appears first → [1]
    assert "[2]" in resolved  # abc12345 appears second → [2]
    # Verify ordering in Sources section
    pos1 = resolved.index("[1] Page B")
    pos2 = resolved.index("[2] Page A")
    assert pos1 < pos2


def test_resolve_unknown_citation(sample_source_map):
    """Unknown source ID removed and warning returned."""
    report = "Unknown source [ffffffff] cited here."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "[ffffffff]" not in resolved
    assert len(warnings) == 1
    assert "ffffffff" in warnings[0]


def test_resolve_mixed_known_unknown(sample_source_map):
    """In a multi-citation, unknown IDs are dropped, known preserved."""
    report = "Mixed [abc12345, ffffffff] citation."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "[1]" in resolved
    assert "ffffffff" not in resolved
    assert len(warnings) == 1


def test_resolve_appends_sources_section(sample_source_map):
    """Resolved report ends with a Sources section."""
    report = "Finding [abc12345] and [def67890]."
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "## Sources" in resolved
    assert "[1] Page A: https://example.com/a" in resolved
    assert "[2] Page B: https://example.com/b" in resolved


def test_resolve_strips_llm_sources_section(sample_source_map):
    """LLM-generated Sources section is replaced by deterministic one."""
    report = (
        "Finding [abc12345].\n\n"
        "## Sources\n\n"
        "[1] Some LLM-invented source: https://fake.com\n"
        "[2] Another fake: https://also-fake.com\n"
    )
    resolved, warnings = resolve_citations(report, sample_source_map)
    assert "fake.com" not in resolved
    assert "## Sources" in resolved
    assert "[1] Page A: https://example.com/a" in resolved


def test_resolve_no_citations():
    """Report with no source IDs returned unchanged."""
    report = "No citations here."
    source_map = {"abc12345": {"url": "https://example.com", "title": "Page"}}
    resolved, warnings = resolve_citations(report, source_map)
    assert resolved == report
    assert not warnings


def test_resolve_empty_source_map():
    """Empty source map returns report unchanged."""
    report = "Finding [abc12345]."
    resolved, warnings = resolve_citations(report, {})
    assert resolved == report
    assert not warnings


# --- _format_results uses [source_id] ---


def test_format_results_falls_back_to_content():
    """When summary is None, falls back to result.content."""
    results = [
        SearchResult(url="https://example.com/a", title="Page A", content="fallback snippet"),
    ]
    summaries = [None]
    source_ids = ["abc12345"]

    output = BaseSearchTool._format_results(results, summaries, source_ids)

    assert "fallback snippet" in output
    assert "[abc12345]" in output
