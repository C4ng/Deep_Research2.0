"""Pydantic schemas used across the system.

Data contracts shared between nodes, tools, and state.
Not to be confused with LLM model configuration.
"""

from typing import Literal

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


class Reflection(BaseModel):
    """Structured reflection after a research round."""

    key_findings: list[str] = Field(description="Specific facts learned this round")
    missing_info: list[str] = Field(description="Specific gaps still remaining")
    contradictions: list[str] = Field(
        default_factory=list,
        description="Conflicting information between sources",
    )
    knowledge_state: Literal["insufficient", "partial", "sufficient"] = Field(
        description="Overall research completeness",
    )
    should_continue: bool = Field(
        description="Whether further searching is likely to help",
    )
    next_queries: list[str] = Field(
        default_factory=list,
        description="Targeted queries for the next research round",
    )
