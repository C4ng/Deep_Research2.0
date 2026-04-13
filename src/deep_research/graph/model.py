"""Shared configurable chat model for all graph nodes.

Uses LangChain's deferred model factory pattern — one shared object,
configured per-use via `.with_config()`. Provider swapping is just a
string change in config: "google_genai:gemini-3.1-flash" ->
"anthropic:claude-sonnet-4-20250514" -> "openai:gpt-4.1".

To add a new provider, install its langchain package (e.g.
`pip install langchain-anthropic`) and use its prefix in the model
string. No code changes needed here.
"""

from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "temperature"),
)
