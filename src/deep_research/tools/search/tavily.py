"""Tavily search provider implementation."""

import asyncio
from typing import Annotated, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import AsyncTavilyClient

from deep_research.configuration import Configuration
from deep_research.models import SearchResult
from deep_research.tools.search import SEARCH_DESCRIPTION
from deep_research.tools.search.base import BaseSearchTool


class TavilySearchTool(BaseSearchTool):
    """Tavily search provider.

    Only implements the Tavily-specific API call and result parsing.
    Summarization, formatting, and error handling are inherited from BaseSearchTool.
    """

    def __init__(self, api_key: str, config: RunnableConfig | None = None):
        super().__init__(config=config)
        self._client = AsyncTavilyClient(api_key=api_key)

    async def search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResult]:
        """Execute search queries via Tavily API and return deduplicated results."""
        search_tasks = [
            self._client.search(
                query,
                max_results=max_results,
                include_raw_content=True,
                topic="general",
            )
            for query in queries
        ]
        responses = await asyncio.gather(*search_tasks)

        seen_urls: dict[str, SearchResult] = {}
        for response in responses:
            for result in response["results"]:
                url = result["url"]
                if url not in seen_urls:
                    seen_urls[url] = SearchResult(
                        url=url,
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        raw_content=result.get("raw_content"),
                    )
        return list(seen_urls.values())


@tool(description=SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Search the web using Tavily and return summarized results.

    Args:
        queries: List of search queries to execute.
        max_results: Maximum number of results per query.
        config: Runtime configuration (injected by LangGraph).

    Returns:
        Formatted string of summarized search results.
    """
    configurable = Configuration.from_runnable_config(config)
    search_tool = TavilySearchTool(
        api_key=configurable.tavily_api_key, config=config
    )
    return await search_tool.search_and_summarize(
        queries, max_results=max_results
    )
