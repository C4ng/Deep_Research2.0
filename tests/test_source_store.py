"""Unit tests for the file-based source store.

Pure logic tests — no API calls, no LLM. Tests source ID generation,
file write/read/dedup, source map building, and format_results output.
"""

import tempfile
from pathlib import Path

import pytest

from deep_research.helpers.source_store import (
    build_source_map,
    generate_source_id,
    read_source,
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
