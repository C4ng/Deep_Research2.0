"""Tests for the Serper (Google Search) provider.

Unit tests use mocked HTTP responses — no API key needed.
Integration tests hit the real Serper API — requires SERPER_API_KEY in .env.

Run unit tests:       pytest tests/test_serper_search.py -v
Run integration too:  pytest tests/test_serper_search.py -v -m integration
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.configuration import Configuration
from deep_research.models import SearchResult
from deep_research.tools.search.serper import SerperSearchTool

# --- Fixtures for mocked responses ---

SERPER_RESPONSE_BASIC = {
    "organic": [
        {
            "title": "Python Programming",
            "link": "https://python.org",
            "snippet": "Python is a high-level programming language.",
            "position": 1,
        },
        {
            "title": "Python Tutorial",
            "link": "https://docs.python.org/tutorial",
            "snippet": "An introduction to Python programming.",
            "position": 2,
        },
    ]
}

SERPER_RESPONSE_OVERLAP = {
    "organic": [
        {
            "title": "Python Programming (duplicate)",
            "link": "https://python.org",
            "snippet": "Same URL as first query.",
            "position": 1,
        },
        {
            "title": "Unique Result",
            "link": "https://unique.example.com",
            "snippet": "Only in the second query.",
            "position": 2,
        },
    ]
}

SERPER_RESPONSE_EMPTY = {"organic": []}


def _mock_response(data: dict) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


# --- Unit tests (mocked HTTP) ---


class TestSerperSearchTool:
    @pytest.mark.asyncio
    async def test_maps_results_correctly(self):
        """Serper results map to SearchResult with correct fields."""
        mock_resp = _mock_response(SERPER_RESPONSE_BASIC)

        with patch("deep_research.tools.search.serper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = SerperSearchTool(api_key="test-key")
            results = await tool.search(["python"], max_results=5)

        assert len(results) == 2

        r0 = results[0]
        assert r0.url == "https://python.org"
        assert r0.title == "Python Programming"
        assert r0.content == "Python is a high-level programming language."
        assert r0.raw_content is None

        r1 = results[1]
        assert r1.url == "https://docs.python.org/tutorial"
        assert r1.content == "An introduction to Python programming."
        assert r1.raw_content is None

    @pytest.mark.asyncio
    async def test_raw_content_always_none(self):
        """All Serper results have raw_content=None (snippet-only provider)."""
        mock_resp = _mock_response(SERPER_RESPONSE_BASIC)

        with patch("deep_research.tools.search.serper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = SerperSearchTool(api_key="test-key")
            results = await tool.search(["python"], max_results=5)

        assert all(r.raw_content is None for r in results)

    @pytest.mark.asyncio
    async def test_dedup_urls_across_queries(self):
        """Overlapping URLs across queries are deduplicated."""
        mock_resp1 = _mock_response(SERPER_RESPONSE_BASIC)
        mock_resp2 = _mock_response(SERPER_RESPONSE_OVERLAP)

        with patch("deep_research.tools.search.serper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_resp1, mock_resp2])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = SerperSearchTool(api_key="test-key")
            results = await tool.search(
                ["python programming", "python language"], max_results=5
            )

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"
        assert urls.count("https://python.org") == 1
        assert "https://unique.example.com" in urls

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Empty organic results return empty list."""
        mock_resp = _mock_response(SERPER_RESPONSE_EMPTY)

        with patch("deep_research.tools.search.serper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = SerperSearchTool(api_key="test-key")
            results = await tool.search(["nonexistent query"], max_results=5)

        assert results == []


# --- Integration tests (real API) ---


@pytest.fixture
def serper_config():
    return Configuration.from_runnable_config(None)


@pytest.fixture
def serper_tool(serper_config):
    assert serper_config.serper_api_key, "SERPER_API_KEY not set in .env"
    return SerperSearchTool(api_key=serper_config.serper_api_key)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_search_returns_results(serper_tool):
    """Serper search returns a non-empty list of SearchResult."""
    results = await serper_tool.search(
        ["What is coral reef bleaching?"], max_results=3
    )
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.url for r in results)
    assert all(r.title for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_raw_content_none_in_practice(serper_tool):
    """All real Serper results have raw_content=None."""
    results = await serper_tool.search(
        ["Python programming language"], max_results=3
    )
    assert all(r.raw_content is None for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_search_deduplicates_by_url(serper_tool):
    """Overlapping queries should deduplicate results by URL."""
    results = await serper_tool.search(
        ["coral reef bleaching", "coral reef bleaching causes"],
        max_results=5,
    )
    urls = [r.url for r in results]
    assert len(urls) == len(set(urls)), "Duplicate URLs found in results"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_search_and_summarize(serper_tool):
    """Full pipeline returns formatted string with source IDs (no summarization)."""
    output = await serper_tool.search_and_summarize(
        ["What causes ocean acidification?"], max_results=2
    )
    assert re.search(r"\[[0-9a-f]{8}\]", output), "Expected [source_id] tag in output"
    assert "URL:" in output
    assert len(output) > 50
