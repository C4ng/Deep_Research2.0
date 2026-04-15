"""Serper (Google Search) provider implementation."""

import asyncio
from typing import Annotated, List

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from deep_research.configuration import Configuration
from deep_research.models import SearchResult
from deep_research.tools.search import SEARCH_DESCRIPTION
from deep_research.tools.search.base import BaseSearchTool

SERPER_API_URL = "https://google.serper.dev/search"


class SerperSearchTool(BaseSearchTool):
    """Serper (Google Search) provider.

    Snippet-only — no raw_content. The summarization step is skipped;
    Google's snippet is used directly as the result content. This tests
    the fallback path in BaseSearchTool.search_and_summarize().
    """

    def __init__(self, api_key: str, config: RunnableConfig | None = None):
        super().__init__(config=config)
        self._api_key = api_key

    async def search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResult]:
        """Execute search queries via Serper API and return deduplicated results."""
        async with httpx.AsyncClient() as client:
            tasks = [
                client.post(
                    SERPER_API_URL,
                    json={"q": query, "num": max_results},
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                )
                for query in queries
            ]
            responses = await asyncio.gather(*tasks)

        seen_urls: dict[str, SearchResult] = {}
        for response in responses:
            response.raise_for_status()
            data = response.json()
            for result in data.get("organic", []):
                url = result["link"]
                if url in seen_urls:
                    continue
                seen_urls[url] = SearchResult(
                    url=url,
                    title=result.get("title", ""),
                    content=result.get("snippet", ""),
                    raw_content=None,  # Serper is snippet-only
                )
        return list(seen_urls.values())


@tool(description=SEARCH_DESCRIPTION)
async def serper_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Search the web using Serper (Google) and return results.

    Args:
        queries: List of search queries to execute.
        max_results: Maximum number of results per query.
        config: Runtime configuration (injected by LangGraph).

    Returns:
        Formatted string of search results (snippets, no summarization).
    """
    configurable = Configuration.from_runnable_config(config)
    search_tool = SerperSearchTool(
        api_key=configurable.serper_api_key, config=config
    )
    return await search_tool.search_and_summarize(
        queries, max_results=max_results
    )
