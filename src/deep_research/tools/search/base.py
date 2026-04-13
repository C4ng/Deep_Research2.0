"""Abstract base interface for search providers.

All search implementations (Tavily, Brave, etc.) inherit from
BaseSearchTool. This ensures consistent behavior across providers
and makes swapping trivial via configuration.
"""

from abc import ABC, abstractmethod

from deep_research.models import SearchResult, WebpageSummary


class BaseSearchTool(ABC):
    """Abstract search provider interface."""

    @abstractmethod
    async def search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResult]:
        """Execute search queries and return deduplicated results.

        Args:
            queries: List of search queries to execute.
            max_results: Maximum results per query.

        Returns:
            Deduplicated list of SearchResult across all queries.
        """
        ...

    @abstractmethod
    async def summarize_content(self, content: str) -> WebpageSummary:
        """Summarize raw webpage content into structured output.

        Args:
            content: Raw webpage text to summarize.

        Returns:
            Structured WebpageSummary with summary and key excerpts.
        """
        ...
