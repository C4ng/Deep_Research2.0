"""Pydantic schemas used across the system.

Data contracts shared between nodes, tools, and state.
Not to be confused with LLM model configuration.
"""

from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    """Structured research brief extracted from user query."""

    title: str = Field(description="Concise title for the research topic")
    research_questions: list[str] = Field(
        description="Specific questions to investigate"
    )
    key_topics: list[str] = Field(
        description="Key topics and subtopics to cover"
    )


class SearchResult(BaseModel):
    """A single search result from any search provider."""

    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    content: str = Field(description="Snippet or summary of the page")
    raw_content: str | None = Field(
        default=None, description="Full raw page content if available"
    )


class WebpageSummary(BaseModel):
    """Structured summary of a webpage, produced by the summarization model."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(
        description="Important quotes or data points extracted verbatim"
    )
