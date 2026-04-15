"""Test provider swapping — verify each LLM provider works with structured output.

Tests a single structured output call (ResearchBrief) per provider to verify:
- Model instantiation via init_chat_model + configurable_fields
- Structured output works across providers
- Thinking/reasoning params are accepted (where supported)
- API key routing works

Usage: python scripts/test_provider_swap.py
"""

import asyncio
import time
from datetime import datetime

from langchain_core.messages import HumanMessage

from deep_research.graph.model import build_model_config, configurable_model
from deep_research.models import ResearchBrief
from deep_research.prompts import research_brief_prompt

QUERY = "What are the main causes of coral reef bleaching?"

# thinking_budget=None for models that don't support reasoning
PROVIDERS = [
    ("gemini",           "google_genai:gemini-2.5-flash",      8192),
    ("anthropic",        "anthropic:claude-sonnet-4-20250514",  8192),
    ("openai (gpt-4.1)", "openai:gpt-4.1-mini",               None),   # no reasoning support
    ("openai (o4-mini)", "openai:o4-mini",                     8192),   # reasoning model
]


async def test_structured_output(name: str, model_str: str, thinking_budget: int | None):
    """One structured output call with the given provider."""
    print(f"\n{'='*60}")
    print(f"Provider: {name}")
    print(f"Model:    {model_str}")
    print(f"Thinking: {thinking_budget}")
    print(f"{'='*60}")

    model_config = build_model_config(
        model=model_str,
        max_tokens=8192,
        temperature=0.5,
        thinking_budget=thinking_budget,
    )
    # Print config without api_key
    safe_config = {k: v for k, v in model_config.items() if k != "api_key"}
    print(f"Config:   {safe_config}")

    model = (
        configurable_model
        .with_structured_output(ResearchBrief)
        .with_retry(stop_after_attempt=2)
        .with_config(configurable=model_config)
    )

    prompt = research_brief_prompt.format(
        messages=f"Human: {QUERY}",
        date=datetime.now().strftime("%B %d, %Y"),
        prior_brief="",
        feedback="",
    )

    start = time.time()
    try:
        brief: ResearchBrief = await model.ainvoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start

        print(f"Status:   SUCCESS ({elapsed:.1f}s)")
        print(f"Title:    {brief.title}")
        print(f"Question: {brief.research_question[:120]}...")
        print(f"Approach: {brief.approach[:120]}...")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"Status:   FAILED ({elapsed:.1f}s)")
        print(f"Error:    {type(e).__name__}: {e}")
        return False


async def main():
    print(f"Query: {QUERY}")
    print(f"Testing {len(PROVIDERS)} providers with structured output (ResearchBrief)...")

    results = {}
    for name, model_str, thinking in PROVIDERS:
        results[name] = await test_structured_output(name, model_str, thinking)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name:20s} {status}")


if __name__ == "__main__":
    asyncio.run(main())
