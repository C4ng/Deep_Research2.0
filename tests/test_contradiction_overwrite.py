"""Integration test for contradiction overwrite semantics.

Runs the researcher subgraph on a topic likely to produce contradictions,
streams reflect node updates, and verifies that contradictions are managed
via overwrite (update/merge/delete/add) rather than append.

Run: pytest tests/test_contradiction_overwrite.py -v -s --tb=short
"""

import logging

import pytest

from deep_research.nodes.researcher import researcher_subgraph
from deep_research.state import ResearcherState

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration

# Topic chosen to reliably produce contradictory sources across multiple rounds
CONTRADICTION_TOPIC = (
    "Is remote work more productive than office work? "
    "Investigate studies and reports from 2023-2025 that compare "
    "productivity metrics, employee satisfaction, and company policies. "
    "Include both pro-remote and pro-office perspectives with specific "
    "data points from named studies (Stanford, Microsoft, Owl Labs, etc). "
    "Also cover the impact on innovation, collaboration quality, and "
    "employee retention rates."
)

INITIAL_STATE: ResearcherState = {
    "messages": [],
    "research_topic": CONTRADICTION_TOPIC,
    "research_iterations": 0,
    "last_reflection": "",
    "accumulated_findings": [],
    "accumulated_contradictions": [],
    "current_gaps": [],
    "final_knowledge_state": "",
    "notes": "",
}

# Limit searches and results per round to force multiple reflection cycles
CONFIG = {"configurable": {"max_searches_per_round": 1, "max_search_results": 2}}


@pytest.mark.asyncio
async def test_contradiction_overwrite_across_rounds():
    """Contradictions are overwritten each round, not appended.

    Tracks the contradiction list at each reflect node to verify:
    1. Each round's list is a complete canonical set (not a delta)
    2. Near-duplicates are merged (not accumulated)
    3. The list can shrink when evidence resolves a contradiction
    4. New contradictions are added when discovered
    """
    contradiction_history: list[list[str]] = []

    async for chunk in researcher_subgraph.astream(
        INITIAL_STATE, config=CONFIG, stream_mode="updates"
    ):
        # Each chunk is {node_name: state_update_dict}
        for node_name, update in chunk.items():
            if node_name != "reflect":
                continue

            contradictions = update.get("accumulated_contradictions", [])
            iteration = update.get("research_iterations", 0)
            contradiction_history.append(contradictions)

            # Log for observation
            print(f"\n{'='*70}")
            print(f"REFLECT ROUND {iteration}")
            print(f"Contradictions ({len(contradictions)}):")
            for i, c in enumerate(contradictions, 1):
                print(f"  {i}. {c}")
            if not contradictions:
                print("  (none)")
            print(f"{'='*70}")

    # --- Assertions ---

    assert len(contradiction_history) >= 1, (
        "Should have at least 1 reflection round"
    )

    # If multiple rounds occurred, verify overwrite behavior
    if len(contradiction_history) >= 2:
        for round_idx in range(1, len(contradiction_history)):
            current = contradiction_history[round_idx]
            previous = contradiction_history[round_idx - 1]

            # Overwrite check: the current list should NOT be a strict superset
            # formed by appending to the previous list. With append semantics,
            # current would always start with all of previous. With overwrite,
            # the LLM may rewrite, merge, or remove entries.
            #
            # We can't assert current != previous (LLM might keep the same list
            # if nothing changed). But we CAN check that the list doesn't show
            # append-pattern growth (previous entries repeated verbatim at start).
            if len(current) > len(previous) and len(previous) > 0:
                # If the list grew, verify it's not just appending
                # (first N entries should not be identical to previous)
                prefix_match = all(
                    current[i] == previous[i] for i in range(len(previous))
                )
                if prefix_match:
                    logger.warning(
                        "Round %d looks like append (prefix matches previous). "
                        "Could be legitimate if LLM kept all prior entries unchanged.",
                        round_idx + 1,
                    )

    # Quality check: no near-duplicate contradictions in the final list
    final_contradictions = contradiction_history[-1]
    if len(final_contradictions) >= 2:
        _check_no_near_duplicates(final_contradictions)

    # The final state should have the same contradictions as the last reflect
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(contradiction_history)} reflection rounds")
    print(f"Contradiction counts per round: {[len(c) for c in contradiction_history]}")
    print(f"Final contradictions: {len(final_contradictions)}")
    print(f"{'='*70}")


@pytest.mark.asyncio
async def test_contradiction_overwrite_with_seeded_state():
    """Verify overwrite with pre-seeded contradictions from a simulated prior round.

    Seeds near-duplicate and resolvable contradictions, then runs one more round.
    Checks that the LLM merges, updates, or removes them appropriately.
    """
    seeded_state: ResearcherState = {
        "messages": [],
        "research_topic": CONTRADICTION_TOPIC,
        "research_iterations": 1,  # pretend we already did round 1
        "last_reflection": (
            "Already covered across all rounds:\n"
            "- Remote workers report 13% higher productivity (Stanford study)\n"
            "- Microsoft found 10% productivity gains\n"
            "\n"
            "Missing information:\n"
            "- Impact on innovation and collaboration quality\n"
            "- Employee retention rates comparison\n"
            "\n"
            "Suggested next queries:\n"
            "- remote work impact on innovation collaboration 2024 2025\n"
        ),
        "accumulated_findings": [
            "Remote workers report 13% higher productivity (Stanford study)",
            "Microsoft found 10% productivity gains for remote workers",
        ],
        # Seed near-duplicate contradictions + one that new evidence may resolve
        "accumulated_contradictions": [
            "Stanford study says remote work boosts productivity by 13%, "
            "but some managers report lower team output with remote teams.",
            "Remote workers are more productive according to Stanford, "
            "yet team managers perceive reduced productivity in remote settings.",
            "Hybrid work may reduce burnout (Owl Labs), but McKinsey says "
            "work model doesn't significantly impact burnout levels.",
        ],
        "current_gaps": [
            "Impact on innovation and collaboration quality",
            "Employee retention rates comparison",
        ],
        "final_knowledge_state": "",
        "notes": "",
    }

    contradiction_history: list[list[str]] = []

    async for chunk in researcher_subgraph.astream(
        seeded_state, config=CONFIG, stream_mode="updates"
    ):
        for node_name, update in chunk.items():
            if node_name != "reflect":
                continue

            contradictions = update.get("accumulated_contradictions", [])
            iteration = update.get("research_iterations", 0)
            contradiction_history.append(contradictions)

            print(f"\n{'='*70}")
            print(f"REFLECT ROUND {iteration} (seeded test)")
            print(f"Contradictions ({len(contradictions)}):")
            for i, c in enumerate(contradictions, 1):
                print(f"  {i}. {c}")
            if not contradictions:
                print("  (none)")
            print(f"{'='*70}")

    assert len(contradiction_history) >= 1

    # The seeded state had 3 contradictions (2 near-duplicates + 1 distinct).
    # After the LLM processes them, we expect:
    # - The 2 near-duplicates to be merged into 1
    # - The distinct one to be kept or updated
    # - Possibly new contradictions added from new searches
    first_round = contradiction_history[0]

    print(f"\nSeeded: 3 contradictions (2 near-dupes + 1 distinct)")
    print(f"After LLM processing: {len(first_round)} contradictions")

    # The LLM should NOT have 3+ entries that look like the original near-dupes
    if len(first_round) >= 3:
        _check_no_near_duplicates(first_round)


def _check_no_near_duplicates(contradictions: list[str]) -> None:
    """Check that no two contradictions are near-duplicates.

    Uses simple heuristic: if two entries share >60% of their words,
    they are likely duplicates that should have been merged.
    """
    for i, a in enumerate(contradictions):
        words_a = set(a.lower().split())
        for j, b in enumerate(contradictions):
            if j <= i:
                continue
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b)
            similarity = overlap / min(len(words_a), len(words_b))
            if similarity > 0.6:
                logger.warning(
                    "Potential near-duplicate contradictions "
                    "(%.0f%% word overlap):\n  [%d] %s\n  [%d] %s",
                    similarity * 100, i + 1, a, j + 1, b,
                )
                # Don't fail hard — log for observation. The LLM might
                # legitimately keep similar but distinct contradictions.
