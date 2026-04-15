"""Brave search provider implementation."""

import asyncio
import re
from typing import Annotated, List

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from deep_research.configuration import Configuration
from deep_research.models import SearchResult
from deep_research.tools.search import SEARCH_DESCRIPTION
from deep_research.tools.search.base import BaseSearchTool

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# Regex to strip HTML tags (Brave returns <strong>, <em>, etc. in descriptions)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return _HTML_TAG_RE.sub("", text)


class BraveSearchTool(BaseSearchTool):
    """Brave search provider.

    Only implements the Brave-specific API call and result parsing.
    Summarization, formatting, and error handling are inherited from BaseSearchTool.
    """

    def __init__(self, api_key: str, config: RunnableConfig | None = None):
        super().__init__(config=config)
        self._api_key = api_key

    async def search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResult]:
        """Execute search queries via Brave Web Search API and return deduplicated results."""
        async with httpx.AsyncClient() as client:
            tasks = [
                client.get(
                    BRAVE_API_URL,
                    params={
                        "q": query,
                        "count": max_results,
                        "extra_snippets": "true",
                    },
                    headers={
                        "X-Subscription-Token": self._api_key,
                        "Accept": "application/json",
                    },
                )
                for query in queries
            ]
            responses = await asyncio.gather(*tasks)

        seen_urls: dict[str, SearchResult] = {}
        for response in responses:
            response.raise_for_status()
            data = response.json()
            for result in data.get("web", {}).get("results", []):
                url = result["url"]
                if url in seen_urls:
                    continue

                description = result.get("description", "")
                extra_snippets = result.get("extra_snippets", [])

                clean_description = _strip_html(description)

                # Concatenate description + extra_snippets for summarization.
                # When extra_snippets are available, the combined text gives
                # the summarization LLM ~6 excerpts to work with.
                if extra_snippets:
                    raw_content = clean_description + "\n\n" + "\n\n".join(extra_snippets)
                else:
                    raw_content = None

                seen_urls[url] = SearchResult(
                    url=url,
                    title=result.get("title", ""),
                    content=clean_description,
                    raw_content=raw_content,
                )
        return list(seen_urls.values())


@tool(description=SEARCH_DESCRIPTION)
async def brave_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Search the web using Brave and return summarized results.

    Args:
        queries: List of search queries to execute.
        max_results: Maximum number of results per query.
        config: Runtime configuration (injected by LangGraph).

    Returns:
        Formatted string of summarized search results.
    """
    configurable = Configuration.from_runnable_config(config)
    search_tool = BraveSearchTool(
        api_key=configurable.brave_api_key, config=config
    )
    return await search_tool.search_and_summarize(
        queries, max_results=max_results
    )
