"""Tests for build_model_config(), get_model_api_key(), and _is_openai_reasoning_model().

Verifies provider-aware thinking/reasoning mapping and API key routing.
"""

import os
from unittest.mock import patch

from deep_research.graph.model import (
    _is_openai_reasoning_model,
    build_model_config,
    get_model_api_key,
)


# --- _is_openai_reasoning_model ---


class TestIsOpenAIReasoningModel:
    def test_o1(self):
        assert _is_openai_reasoning_model("openai:o1") is True

    def test_o3(self):
        assert _is_openai_reasoning_model("openai:o3") is True

    def test_o3_mini(self):
        assert _is_openai_reasoning_model("openai:o3-mini") is True

    def test_o4_mini(self):
        assert _is_openai_reasoning_model("openai:o4-mini") is True

    def test_gpt4_is_not_reasoning(self):
        assert _is_openai_reasoning_model("openai:gpt-4.1") is False

    def test_gpt4_mini_is_not_reasoning(self):
        assert _is_openai_reasoning_model("openai:gpt-4.1-mini") is False

    def test_gpt4o_is_not_reasoning(self):
        assert _is_openai_reasoning_model("openai:gpt-4o") is False

    def test_non_openai_model(self):
        assert _is_openai_reasoning_model("anthropic:claude-sonnet-4-20250514") is False


# --- build_model_config: thinking_budget mapping ---


class TestBuildModelConfigThinking:
    def test_gemini_gets_thinking_budget(self):
        """Gemini models get thinking_budget as a flat integer."""
        config = build_model_config(
            model="google_genai:gemini-2.5-flash",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=4096,
        )
        assert config["thinking_budget"] == 4096
        assert "thinking" not in config
        assert "reasoning" not in config

    def test_anthropic_gets_thinking_dict(self):
        """Anthropic models get thinking as a dict with type and budget_tokens."""
        config = build_model_config(
            model="anthropic:claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=8192,
        )
        assert config["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        assert "thinking_budget" not in config
        assert "reasoning" not in config

    def test_openai_o_series_low_effort(self):
        """o-series with thinking_budget <= 4096 maps to effort=low."""
        config = build_model_config(
            model="openai:o4-mini",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=2048,
        )
        assert config["reasoning"] == {"effort": "low"}
        assert "thinking_budget" not in config
        assert "thinking" not in config

    def test_openai_o_series_medium_effort(self):
        """o-series with 4096 < thinking_budget <= 8192 maps to effort=medium."""
        config = build_model_config(
            model="openai:o4-mini",
            max_tokens=16384,
            temperature=0.5,
            thinking_budget=8192,
        )
        assert config["reasoning"]["effort"] == "medium"

    def test_openai_o_series_high_effort(self):
        """o-series with thinking_budget > 8192 maps to effort=high."""
        config = build_model_config(
            model="openai:o3",
            max_tokens=16384,
            temperature=0.5,
            thinking_budget=16384,
        )
        assert config["reasoning"]["effort"] == "high"

    def test_openai_gpt_ignores_thinking_budget(self):
        """Standard GPT models ignore thinking_budget (no reasoning support)."""
        config = build_model_config(
            model="openai:gpt-4.1-mini",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=8192,
        )
        assert "reasoning" not in config
        assert "thinking_budget" not in config
        assert "thinking" not in config

    def test_none_thinking_budget_excluded(self):
        """None thinking_budget produces no thinking/reasoning params."""
        config = build_model_config(
            model="google_genai:gemini-2.5-flash",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=None,
        )
        assert "thinking_budget" not in config
        assert "thinking" not in config
        assert "reasoning" not in config

    def test_zero_thinking_budget_excluded(self):
        """Zero thinking_budget produces no thinking/reasoning params."""
        config = build_model_config(
            model="anthropic:claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.5,
            thinking_budget=0,
        )
        assert "thinking_budget" not in config
        assert "thinking" not in config
        assert "reasoning" not in config

    def test_unknown_provider_no_thinking(self):
        """Unknown provider prefix skips thinking params entirely."""
        config = build_model_config(
            model="together:meta-llama/Llama-3-70b",
            max_tokens=4096,
            temperature=0.7,
            thinking_budget=8192,
        )
        assert "thinking_budget" not in config
        assert "thinking" not in config
        assert "reasoning" not in config


# --- build_model_config: temperature handling ---


class TestBuildModelConfigTemperature:
    def test_gemini_has_temperature(self):
        config = build_model_config(
            model="google_genai:gemini-2.5-flash",
            max_tokens=8192,
            temperature=0.5,
        )
        assert config["temperature"] == 0.5

    def test_anthropic_has_temperature(self):
        config = build_model_config(
            model="anthropic:claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.5,
        )
        assert config["temperature"] == 0.5

    def test_openai_gpt_has_temperature(self):
        config = build_model_config(
            model="openai:gpt-4.1-mini",
            max_tokens=8192,
            temperature=0.5,
        )
        assert config["temperature"] == 0.5

    def test_openai_o_series_no_temperature(self):
        """o-series reasoning models don't support temperature."""
        config = build_model_config(
            model="openai:o4-mini",
            max_tokens=8192,
            temperature=0.5,
        )
        assert "temperature" not in config

    def test_openai_o1_no_temperature(self):
        config = build_model_config(
            model="openai:o1",
            max_tokens=8192,
            temperature=0.7,
        )
        assert "temperature" not in config


# --- build_model_config: base fields ---


class TestBuildModelConfigBase:
    def test_base_fields_present(self):
        """model and max_tokens are always in the config."""
        config = build_model_config(
            model="google_genai:gemini-2.5-flash",
            max_tokens=16384,
            temperature=0.5,
        )
        assert config["model"] == "google_genai:gemini-2.5-flash"
        assert config["max_tokens"] == 16384

    def test_explicit_api_key(self):
        """Explicitly passed api_key is used."""
        config = build_model_config(
            model="anthropic:claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.5,
            api_key="sk-explicit-key",
        )
        assert config["api_key"] == "sk-explicit-key"

    def test_api_key_auto_resolved(self):
        """API key auto-resolves from env when not explicitly passed."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
            config = build_model_config(
                model="google_genai:gemini-2.5-flash",
                max_tokens=8192,
                temperature=0.5,
            )
        assert config["api_key"] == "test-google-key"

    def test_no_api_key_when_unset(self):
        """No api_key in config when env var is unset and none passed."""
        with patch.dict(os.environ, {}, clear=True):
            config = build_model_config(
                model="google_genai:gemini-2.5-flash",
                max_tokens=8192,
                temperature=0.5,
            )
        assert "api_key" not in config


# --- get_model_api_key ---


class TestGetModelApiKey:
    def test_google_prefix(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "gk"}):
            assert get_model_api_key("google_genai:gemini-2.5-flash") == "gk"

    def test_anthropic_prefix(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ak"}):
            assert get_model_api_key("anthropic:claude-sonnet-4-20250514") == "ak"

    def test_openai_prefix(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "ok"}):
            assert get_model_api_key("openai:gpt-4.1") == "ok"

    def test_unknown_prefix_returns_none(self):
        assert get_model_api_key("together:meta-llama/Llama-3-70b") is None

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "ok"}):
            assert get_model_api_key("OpenAI:gpt-4.1") == "ok"

    def test_missing_env_var_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_api_key("google_genai:gemini-2.5-flash") is None
