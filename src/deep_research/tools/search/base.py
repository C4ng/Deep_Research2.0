"""Base search tool with shared summarization and formatting logic.

All search implementations (Tavily, Brave, etc.) inherit from
BaseSearchTool and only need to implement `search()`. Summarization,
result formatting, and error handling are handled here.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_research.configuration import Configuration
from deep_research.graph.model import configurable_model
from deep_research.helpers.source_store import (
    generate_source_id,
    get_sources_dir,
    read_source,
    write_source,
)
from deep_research.models import SearchResult, WebpageSummary

logger = logging.getLogger(__name__)

# Max characters of raw webpage content to send for summarization
MAX_CONTENT_LENGTH = 50000

# Timeout for a single summarization call
SUMMARIZATION_TIMEOUT = 60.0


class BaseSearchTool(ABC):
    """Base search provider with shared summarization logic.

    Subclasses only need to implement `search()` — the provider-specific
    API call. Everything else (summarization, formatting, error handling)
    is inherited.
    """

    def __init__(self, config: RunnableConfig | None = None):
        self._config = config

    @abstractmethod
    async def search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResult]:
        """Execute search queries and return deduplicated results.

        This is the only method subclasses must implement.

        Args:
            queries: List of search queries to execute.
            max_results: Maximum results per query.

        Returns:
            Deduplicated list of SearchResult across all queries.
        """
        ...

    async def summarize_content(self, content: str) -> WebpageSummary:
        """Summarize raw webpage content using the configured summarization model."""
        from deep_research.prompts import summarize_webpage_prompt

        configurable = Configuration.from_runnable_config(self._config)
        model_config = {
            "model": configurable.summarization_model,
            "max_tokens": configurable.summarization_model_max_tokens,
            "temperature": configurable.summarization_model_temperature,
        }
        if configurable.summarization_model_thinking_budget is not None:
            model_config["thinking_budget"] = configurable.summarization_model_thinking_budget

        summary_model = (
            configurable_model
            .with_structured_output(WebpageSummary)
            .with_retry(stop_after_attempt=3)
            .with_config(configurable=model_config)
        )

        prompt = summarize_webpage_prompt.format(
            webpage_content=content[:MAX_CONTENT_LENGTH],
            date=datetime.now().strftime("%B %d, %Y"),
        )
        return await asyncio.wait_for(
            summary_model.ainvoke([HumanMessage(content=prompt)]),
            timeout=SUMMARIZATION_TIMEOUT,
        )

    async def search_and_summarize(
        self, queries: list[str], *, max_results: int = 5
    ) -> str:
        """Search, summarize results, and return formatted output.

        This is the main entry point used by the researcher node.
        Writes each source to the file-based store for citation tracking.
        Cross-round dedup: if a URL was already summarized, reuses the
        stored summary and skips the summarization LLM call.
        """
        results = await self.search(queries, max_results=max_results)

        if not results:
            return "No search results found. Try different search queries."

        sources_dir = get_sources_dir(self._config)

        async def _summarize_one(result: SearchResult) -> tuple[str, str | None]:
            source_id = generate_source_id(result.url)

            # Cross-round dedup: reuse stored summary if available
            existing = read_source(sources_dir, source_id)
            if existing:
                logger.info("Source %s already in store, skipping summarization", source_id)
                return source_id, existing["content"]

            if not result.raw_content:
                return source_id, None

            try:
                summary = await self.summarize_content(result.raw_content)
                content = (
                    f"<summary>\n{summary.summary}\n</summary>\n\n"
                    f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
                )
                write_source(
                    sources_dir, source_id, result.url, result.title,
                    summary.summary, summary.key_excerpts,
                )
                return source_id, content
            except asyncio.TimeoutError:
                logger.warning("Summarization timed out for %s", result.url)
                return source_id, None
            except Exception as e:
                logger.warning("Summarization failed for %s: %s", result.url, e)
                return source_id, None

        pairs = await asyncio.gather(
            *[_summarize_one(r) for r in results]
        )
        source_ids = [p[0] for p in pairs]
        summaries = [p[1] for p in pairs]

        return self._format_results(results, summaries, source_ids)

    @staticmethod
    def _format_results(
        results: list[SearchResult],
        summaries: list[str | None],
        source_ids: list[str],
    ) -> str:
        """Format search results with stable source IDs for citation tracking."""
        output = "Search results:\n\n"
        for result, summary, sid in zip(results, summaries, source_ids):
            content = summary if summary else result.content
            output += f"\n--- [{sid}] {result.title} ---\n"
            output += f"URL: {result.url}\n\n"
            output += f"{content}\n\n"
            output += "-" * 80 + "\n"
        return output
