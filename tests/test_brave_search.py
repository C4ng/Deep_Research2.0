"""Tests for the Brave search provider.

Unit tests use mocked HTTP responses — no API key needed.
Integration tests hit the real Brave API — requires BRAVE_API_KEY in .env.

Run unit tests:       pytest tests/test_brave_search.py -v
Run integration too:  pytest tests/test_brave_search.py -v -m integration
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.configuration import Configuration
from deep_research.models import SearchResult
from deep_research.tools.search.brave import BraveSearchTool, _strip_html

# --- Fixtures for mocked responses ---

BRAVE_RESPONSE_BASIC = {
    "web": {
        "results": [
            {
                "title": "Python Programming",
                "url": "https://python.org",
                "description": "Python is a <strong>programming language</strong>.",
                "extra_snippets": [
                    "Python is widely used in data science.",
                    "Python supports multiple paradigms.",
                    "The latest version is Python 3.12.",
                ],
            },
            {
                "title": "Python Tutorial",
                "url": "https://docs.python.org/tutorial",
                "description": "An introduction to Python.",
                "extra_snippets": [
                    "This tutorial covers the basics of Python.",
                ],
            },
        ]
    }
}

BRAVE_RESPONSE_NO_EXTRAS = {
    "web": {
        "results": [
            {
                "title": "Simple Result",
                "url": "https://example.com",
                "description": "A result with no extra snippets.",
            },
        ]
    }
}

BRAVE_RESPONSE_OVERLAP = {
    "web": {
        "results": [
            {
                "title": "Overlapping Result",
                "url": "https://python.org",
                "description": "Same URL as another query.",
                "extra_snippets": ["Duplicate content."],
            },
            {
                "title": "Unique Result",
                "url": "https://unique.example.com",
                "description": "A unique result.",
                "extra_snippets": ["Only in second query."],
            },
        ]
    }
}


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


# --- Unit tests (mocked HTTP) ---


class TestStripHtml:
    def test_strips_strong_tags(self):
        assert _strip_html("a <strong>bold</strong> word") == "a bold word"

    def test_strips_em_tags(self):
        assert _strip_html("an <em>italic</em> word") == "an italic word"

    def test_strips_nested_tags(self):
        assert _strip_html("<p>text <b>bold</b></p>") == "text bold"

    def test_no_tags_unchanged(self):
        assert _strip_html("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_html("") == ""


class TestBraveSearchTool:
    @pytest.mark.asyncio
    async def test_maps_results_with_extra_snippets(self):
        """Results with extra_snippets get concatenated raw_content."""
        mock_resp = _mock_response(BRAVE_RESPONSE_BASIC)

        with patch("deep_research.tools.search.brave.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = BraveSearchTool(api_key="test-key")
            results = await tool.search(["python"], max_results=5)

        assert len(results) == 2

        # First result: has extra_snippets → raw_content is concatenated
        r0 = results[0]
        assert r0.url == "https://python.org"
        assert r0.title == "Python Programming"
        assert r0.content == "Python is a programming language."  # HTML stripped
        assert r0.raw_content is not None
        assert "Python is widely used in data science." in r0.raw_content
        assert "Python supports multiple paradigms." in r0.raw_content
        assert "The latest version is Python 3.12." in r0.raw_content
        # raw_content starts with the cleaned description
        assert r0.raw_content.startswith("Python is a programming language.")

        # Second result
        r1 = results[1]
        assert r1.url == "https://docs.python.org/tutorial"
        assert r1.raw_content is not None
        assert "This tutorial covers the basics of Python." in r1.raw_content

    @pytest.mark.asyncio
    async def test_no_extra_snippets_sets_raw_content_none(self):
        """Results without extra_snippets have raw_content=None."""
        mock_resp = _mock_response(BRAVE_RESPONSE_NO_EXTRAS)

        with patch("deep_research.tools.search.brave.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = BraveSearchTool(api_key="test-key")
            results = await tool.search(["test"], max_results=5)

        assert len(results) == 1
        assert results[0].raw_content is None
        assert results[0].content == "A result with no extra snippets."

    @pytest.mark.asyncio
    async def test_dedup_urls_across_queries(self):
        """Overlapping URLs across queries are deduplicated."""
        mock_resp1 = _mock_response(BRAVE_RESPONSE_BASIC)
        mock_resp2 = _mock_response(BRAVE_RESPONSE_OVERLAP)

        with patch("deep_research.tools.search.brave.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_resp1, mock_resp2])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = BraveSearchTool(api_key="test-key")
            results = await tool.search(
                ["python programming", "python language"], max_results=5
            )

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"
        # python.org appears in both responses but should only appear once
        assert urls.count("https://python.org") == 1
        # unique result from second query should be present
        assert "https://unique.example.com" in urls

    @pytest.mark.asyncio
    async def test_strips_html_from_description(self):
        """HTML tags in description are stripped in content field."""
        mock_resp = _mock_response(BRAVE_RESPONSE_BASIC)

        with patch("deep_research.tools.search.brave.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            tool = BraveSearchTool(api_key="test-key")
            results = await tool.search(["python"], max_results=5)

        # First result has <strong> in description
        assert "<strong>" not in results[0].content
        assert "programming language" in results[0].content


# --- Integration tests (real API) ---

pytestmark_integration = pytest.mark.integration


@pytest.fixture
def brave_config():
    return Configuration.from_runnable_config(None)


@pytest.fixture
def brave_tool(brave_config):
    assert brave_config.brave_api_key, "BRAVE_API_KEY not set in .env"
    return BraveSearchTool(api_key=brave_config.brave_api_key)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_brave_search_returns_results(brave_tool):
    """Brave search returns a non-empty list of SearchResult."""
    results = await brave_tool.search(
        ["What is coral reef bleaching?"], max_results=3
    )
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.url for r in results)
    assert all(r.title for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_brave_search_has_extra_snippets(brave_tool):
    """At least some results should have raw_content from extra_snippets."""
    results = await brave_tool.search(
        ["Python programming language"], max_results=3
    )
    has_raw = [r for r in results if r.raw_content]
    assert len(has_raw) > 0, "No results with raw_content from extra_snippets"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_brave_search_deduplicates_by_url(brave_tool):
    """Overlapping queries should deduplicate results by URL."""
    results = await brave_tool.search(
        ["coral reef bleaching", "coral reef bleaching causes"],
        max_results=5,
    )
    urls = [r.url for r in results]
    assert len(urls) == len(set(urls)), "Duplicate URLs found in results"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_brave_search_and_summarize(brave_tool):
    """Full search_and_summarize pipeline returns formatted string with sources."""
    output = await brave_tool.search_and_summarize(
        ["What causes ocean acidification?"], max_results=2
    )
    assert re.search(r"\[[0-9a-f]{8}\]", output), "Expected [source_id] tag in output"
    assert "URL:" in output
    assert len(output) > 100
