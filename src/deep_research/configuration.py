"""Configuration management for the Deep Research system."""

import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

load_dotenv()


class SearchAPI(Enum):
    """Available search API providers."""

    TAVILY = "tavily"


class Configuration(BaseModel):
    """Main configuration for the Deep Research agent.

    Grows incrementally — this is the minimal version for Increment 1.
    """

    # Model configuration
    research_model: str = Field(
        default="google_genai:gemini-2.5-flash",
        description="Model for research tasks (reasoning, tool calling, brief generation, report writing)",
    )
    research_model_max_tokens: int = Field(
        default=16384,
        description="Maximum output tokens for the research model",
    )
    research_model_temperature: float = Field(
        default=0.5,
        description="Temperature for research model — moderate for query exploration",
    )
    research_model_thinking_budget: Optional[int] = Field(
        default=8192,
        description=(
            "Max thinking/reasoning tokens for the research model. "
            "Set to None to skip (for providers that don't support it). "
            "Reasoning models (Gemini 2.5+) use internal chain-of-thought that "
            "counts against max_tokens — without a budget, thinking can starve output."
        ),
    )
    reflection_thinking_budget: Optional[int] = Field(
        default=2048,
        description=(
            "Max thinking tokens for reflection LLM calls. Lower than research "
            "model — reflection is assessment, not deep reasoning."
        ),
    )
    # TODO: Add a model config resolver that builds provider-aware config dicts.
    # Currently thinking_budget is conditionally passed (only when not None),
    # but a proper resolver would handle all provider-specific params in one place
    # when we add OpenAI/Anthropic support.
    summarization_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Model for summarizing webpage content (cheap, mechanical extraction)",
    )
    summarization_model_max_tokens: int = Field(
        default=4096,
        description="Maximum output tokens for the summarization model",
    )
    summarization_model_temperature: float = Field(
        default=0.0,
        description="Temperature for summarization — low for factual extraction",
    )
    summarization_model_thinking_budget: Optional[int] = Field(
        default=0,
        description=(
            "Max thinking tokens for summarization. Default 0 (disabled) — "
            "summarization is mechanical extraction, reasoning wastes tokens."
        ),
    )

    # Search configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="Search API provider to use",
    )
    max_search_results: int = Field(
        default=5,
        description="Maximum number of results per search query",
    )

    # Researcher limits
    max_searches_per_round: int = Field(
        default=3,
        description="Maximum search tool calls per research round",
    )
    max_research_iterations: int = Field(
        default=3,
        description="Maximum reflection-research cycles before forcing exit",
    )
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum retries for structured output and model calls",
    )

    # Coordinator limits
    max_research_topics: int = Field(
        default=5,
        description="Maximum number of subtopics the coordinator can decompose into",
    )
    max_coordinator_iterations: int = Field(
        default=2,
        description="Maximum coordinator reflection cycles before forcing exit",
    )

    # API keys — loaded from env via dotenv, not hardcoded
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key (loaded from TAVILY_API_KEY env var)",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration from a LangGraph RunnableConfig.

        Resolution order: environment variables > runtime config > defaults.
        """
        configurable = config.get("configurable", {}) if config else {}
        values: dict[str, Any] = {}
        for field_name in cls.model_fields:
            env_val = os.environ.get(field_name.upper())
            config_val = configurable.get(field_name)
            if env_val is not None:
                values[field_name] = env_val
            elif config_val is not None:
                values[field_name] = config_val
        return cls(**values)
