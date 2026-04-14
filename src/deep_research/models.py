"""Pydantic schemas used across the system.

Data contracts shared between nodes, tools, and state.
Not to be confused with LLM model configuration.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClarifyOutput(BaseModel):
    """Structured output from the clarification node."""

    need_clarification: bool = Field(
        description="Whether to ask the user a clarifying question"
    )
    question: str = Field(
        description="The clarifying question to ask (empty if not needed)"
    )
    verification: str = Field(
        description="Acknowledgement message before starting research (empty if clarifying)"
    )


class ResearchBrief(BaseModel):
    """Structured research brief extracted from user query.

    A single well-articulated research question — NOT decomposed into
    subtopics. Topic decomposition is the coordinator's job.
    """

    title: str = Field(description="Concise title for the research topic")
    research_question: str = Field(
        description="Detailed research question with all user constraints preserved"
    )
    approach: str = Field(
        description="Strategic guidance: what kind of question, angles to cover, breadth vs depth"
    )
    is_simple: bool = Field(
        description=(
            "Whether a single researcher can handle this question. "
            "True for narrow, factual queries with one clear answer expected."
        )
    )
    ready_to_proceed: bool = Field(
        default=False,
        description="Whether the user approved the brief or requested changes"
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


class ResearchReflection(BaseModel):
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
    prior_gaps_filled: int = Field(
        default=0,
        description=(
            "How many gaps from the previous round were answered by this "
            "round's findings. 0 if no prior gaps existed or none were filled."
        ),
    )


class ResearchResult(BaseModel):
    """Result from a single researcher, built from accumulated state fields.

    Returned by the dispatch_research tool. Carries structured metadata
    so the coordinator can assess cross-topic completeness without re-reading
    full notes.
    """

    topic: str = Field(description="The research topic that was investigated")
    notes: str = Field(description="Compressed research notes for the final report")
    key_findings: list[str] = Field(
        description="Key findings accumulated across all reflection rounds"
    )
    knowledge_state: Literal["insufficient", "partial", "sufficient"] = Field(
        description="Final knowledge completeness assessment from the researcher"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Gaps remaining after the researcher finished"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictions discovered during research"
    )


class CoordinatorReflection(BaseModel):
    """Structured reflection by the coordinator after collecting research results.

    Assesses cross-topic completeness. The coordinator LLM decides what
    follow-up research to conduct based on gaps and contradictions.
    """

    overall_assessment: str = Field(
        description="Brief assessment of how well the research covers the brief"
    )
    cross_topic_contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictions found between different researchers' findings"
    )
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="Important aspects of the brief not covered by any researcher"
    )
    should_continue: bool = Field(
        description="Whether follow-up research is needed"
    )
    knowledge_state: Literal["insufficient", "partial", "sufficient"] = Field(
        description="Overall completeness across all topics"
    )
