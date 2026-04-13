"""Tests for search tool implementations.

These tests hit real APIs (Tavily, Gemini) — requires valid API keys in .env.
"""

import pytest

from deep_research.configuration import Configuration
from deep_research.models import SearchResult, WebpageSummary
from deep_research.tools.search.tavily import TavilySearchTool


@pytest.fixture
def config():
    return Configuration.from_runnable_config(None)


@pytest.fixture
def tavily_tool(config):
    assert config.tavily_api_key, "TAVILY_API_KEY not set in .env"
    return TavilySearchTool(api_key=config.tavily_api_key)


@pytest.mark.asyncio
async def test_tavily_search_returns_results(tavily_tool):
    """Tavily search returns a non-empty list of SearchResult."""
    results = await tavily_tool.search(
        ["What is coral reef bleaching?"], max_results=3
    )
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.url for r in results)
    assert all(r.title for r in results)


@pytest.mark.asyncio
async def test_tavily_search_deduplicates_by_url(tavily_tool):
    """Overlapping queries should deduplicate results by URL."""
    results = await tavily_tool.search(
        ["coral reef bleaching", "coral reef bleaching causes"],
        max_results=5,
    )
    urls = [r.url for r in results]
    assert len(urls) == len(set(urls)), "Duplicate URLs found in results"


@pytest.mark.asyncio
async def test_tavily_search_includes_raw_content(tavily_tool):
    """At least some results should have raw_content for summarization."""
    results = await tavily_tool.search(
        ["Python programming language"], max_results=3
    )
    has_raw = [r for r in results if r.raw_content]
    assert len(has_raw) > 0, "No results with raw_content — summarization won't work"


@pytest.mark.asyncio
async def test_summarize_content(tavily_tool):
    """Summarization produces a valid WebpageSummary from raw text."""
    sample_content = (
        "Coral bleaching occurs when corals are stressed by changes in conditions "
        "such as temperature, light, or nutrients. When stressed, corals expel the "
        "symbiotic algae living in their tissues, causing them to turn white. "
        "Without these algae, the coral loses its major source of food and is more "
        "susceptible to disease. Mass bleaching events have become more frequent "
        "due to rising ocean temperatures linked to climate change. The Great Barrier "
        "Reef experienced severe bleaching in 2016, 2017, 2020, and 2022."
    )
    summary = await tavily_tool.summarize_content(sample_content)
    assert isinstance(summary, WebpageSummary)
    assert len(summary.summary) > 0
    assert len(summary.key_excerpts) > 0


@pytest.mark.asyncio
async def test_search_and_summarize_formatted_output(tavily_tool):
    """Full search_and_summarize pipeline returns formatted string with sources."""
    output = await tavily_tool.search_and_summarize(
        ["What causes ocean acidification?"], max_results=2
    )
    assert "SOURCE 1" in output
    assert "URL:" in output
    assert len(output) > 100
