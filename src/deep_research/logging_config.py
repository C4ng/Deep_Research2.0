"""Logging and tracing configuration for Deep Research.

Sets up:
1. Python logging for local console output (node execution, tool calls)
2. LangSmith tracing when LANGSMITH_TRACING=true in env (automatic via langchain)

LangSmith requires no code changes — LangChain/LangGraph auto-instrument
when LANGSMITH_TRACING=true. This module just handles Python logging.
"""

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def setup_logging() -> None:
    """Configure logging for the deep_research package.

    Respects LOG_LEVEL env var (default: INFO).
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # Configure the deep_research logger hierarchy
    logger = logging.getLogger("deep_research")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.addHandler(handler)
    logger.propagate = False

    # Quiet noisy upstream loggers
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def check_tracing_status() -> dict:
    """Report which observability backends are active. Useful for debugging."""
    return {
        "langsmith_tracing": os.environ.get("LANGSMITH_TRACING", "false").lower() == "true",
        "langsmith_project": os.environ.get("LANGSMITH_PROJECT", ""),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    }
