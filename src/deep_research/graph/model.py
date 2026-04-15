"""Shared configurable chat model and provider-aware helpers.

Uses LangChain's deferred model factory pattern — one shared object,
configured per-use via `.with_config()`. Provider swapping is just a
string change in config: "google_genai:gemini-2.5-flash" ->
"anthropic:claude-sonnet-4-20250514" -> "openai:gpt-4.1".

To add a new provider, install its langchain package (e.g.
`pip install langchain-anthropic`) and use its prefix in the model
string. No code changes needed here.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(
    configurable_fields=(
        "model", "max_tokens", "api_key", "temperature",
        "thinking_budget",  # Gemini
        "thinking",         # Anthropic
        "reasoning",        # OpenAI
    ),
)


def get_model_api_key(model: str) -> str | None:
    """Get API key for a model by provider prefix.

    Routes to the conventional env var for each LangChain provider prefix.
    Returns None if the prefix is unrecognized or the env var is unset.
    """
    model_lower = model.lower()
    if model_lower.startswith("openai:"):
        return os.environ.get("OPENAI_API_KEY")
    if model_lower.startswith("anthropic:"):
        return os.environ.get("ANTHROPIC_API_KEY")
    if model_lower.startswith("google_genai:"):
        return os.environ.get("GOOGLE_API_KEY")
    return None


def _is_openai_reasoning_model(model: str) -> bool:
    """Check if an OpenAI model is a reasoning model (o-series).

    Reasoning models (o1, o3, o4-mini, etc.) don't support ``temperature``
    and use ``reasoning`` instead. Standard models (gpt-4.1, gpt-4o, etc.)
    support ``temperature`` but not ``reasoning``.
    """
    # Strip "openai:" prefix, check if model name starts with "o" + digit
    name = model.removeprefix("openai:").lower()
    return len(name) >= 2 and name[0] == "o" and name[1].isdigit()


def build_model_config(
    model: str,
    max_tokens: int,
    temperature: float,
    thinking_budget: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build a provider-aware model config dict.

    Centralizes the model_config construction that was previously copy-pasted
    across 9 node files. Maps our unified ``thinking_budget`` integer to each
    provider's native parameter format:

    - Gemini (``google_genai:``): ``thinking_budget=N``
    - Anthropic (``anthropic:``): ``thinking={"type": "enabled", "budget_tokens": N}``
    - OpenAI reasoning (``openai:o*``): ``reasoning={"effort": ..., "summary": "auto"}``
    - OpenAI standard (``openai:gpt-*``): no thinking/reasoning support

    OpenAI reasoning models (o-series) don't support ``temperature`` — it is
    automatically excluded for those models.

    When *api_key* is not provided, auto-resolves from env by provider prefix.
    """
    resolved_key = api_key or get_model_api_key(model)
    config: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
    }
    if resolved_key:
        config["api_key"] = resolved_key

    # OpenAI reasoning models (o-series) reject temperature
    is_o_series = model.startswith("openai:") and _is_openai_reasoning_model(model)
    if not is_o_series:
        config["temperature"] = temperature

    # Map thinking_budget to provider-specific format
    if thinking_budget is not None and thinking_budget > 0:
        if model.startswith("google_genai:"):
            config["thinking_budget"] = thinking_budget
        elif model.startswith("anthropic:"):
            config["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        elif is_o_series:
            if thinking_budget <= 4096:
                effort = "low"
            elif thinking_budget <= 8192:
                effort = "medium"
            else:
                effort = "high"
            config["reasoning"] = {"effort": effort}
        # openai standard models (gpt-*): no reasoning support, skip
    return config
