# Increment 7 — Research Quality + Report Polish

**Goal**: Dead-end detection, contradiction dedup, hallucinated citation marking, and report structure fixes.
**Status**: Planning
**Depends on**: Increment 6 (Final Report Redesign) — report_metadata, citation resolution

## Overview

Three real problems that need code/structural fixes, plus two report prompt
issues observed in Increment 6:

1. **Dead-end detection** — the LLM is optimistic about "one more search" even
   when gaps persist unchanged. Programmatic routing needed.
2. **Contradiction dedup** — append reducer causes verbatim duplicates in
   `accumulated_contradictions`. Bug fix.
3. **`[unverified]` marking** — hallucinated citations are silently removed.
   The reader should see which claims lost their citation.
4. **Report contradictions buried inline** — no visual distinction. Increment 6
   observation.
5. **No summary "Areas for Further Research"** — gaps noted inline but no
   top-level collection. Increment 6 observation.

**What we're NOT adding** (LLM already handles these adequately):
- Contradiction resolution queries — the LLM naturally includes these in
  `next_queries` when it sees contradictions in context
- Multi-source corroboration tracking — the LLM notes agreement/disagreement
  when it sees multiple sources
- Source authority awareness — the LLM sees full URLs and knows .gov/.edu
- Coordinator low-confidence handling — coordinator already reasons about
  knowledge_state
- Report self-check — we already instruct "do NOT invent source IDs"
- Researcher contradiction guidance — researcher already acts on reflection

These can be added as targeted prompt fixes if integration testing reveals
the LLM isn't handling them well enough.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dead-end detection | **LLM-assessed `prior_gaps_filled` field** | Programmatic string comparison is fragile (LLM rephrases gaps). The LLM already sees prior gaps in context — one integer field, clean routing. |
| Dead-end routing | **Reformulate once, then force exit** | Round 2+: if `prior_gaps_filled == 0` and gaps exist, inject reformulation guidance. Round 3+: if still dead end, force exit. Two strikes = give up. |
| Contradiction dedup | **Exact string dedup at accumulation** | Filter in reflect node before appending. Case-insensitive exact match — won't catch semantic duplicates but catches the common verbatim copy case. |
| Hallucinated citations | **`[unverified]` replacement** | Currently `resolve_citations` silently removes unknown IDs. Replace with `[unverified]` so the reader sees which claims lost their citation. |
| Report structure | **Prompt refinement only** | Two targeted prompt changes for observed problems: visual hierarchy for contradictions + summary "Areas for Further Research" section. |

---

## Design

### Dead-End Detection Flow

```
reflect node:
  1. LLM produces ResearchReflection (now includes prior_gaps_filled)
  2. Check dead-end condition:
     - prior_gap_count = len(state["current_gaps"])  # last round's gaps
     - dead_end = (prior_gap_count > 0
                   and reflection.prior_gaps_filled == 0
                   and iteration >= 2)
  3. If dead_end and iteration < 3:
     - Inject reformulation guidance into last_reflection
     - Route to researcher (one more chance with different angle)
  4. If dead_end and iteration >= 3:
     - Force exit to summarize (override should_continue)
  5. Normal routing otherwise (unchanged)
```

### [unverified] Flow

```
resolve_citations():
  Current: unknown source IDs → bracket removed entirely
  New:     unknown source IDs → replaced with [unverified]

Example:
  Input:  "The market grew 40% [a1b2c3d4]."  (a1b2c3d4 not in store)
  Before: "The market grew 40% ."
  After:  "The market grew 40% [unverified]."
```

---

## Implementation Steps

### Step 0 — Dead-end detection: model + routing

**Files**: `src/deep_research/models.py`, `src/deep_research/prompts.py`, `src/deep_research/nodes/researcher/reflect.py`

**models.py** — Add `prior_gaps_filled` to `ResearchReflection`:

```python
class ResearchReflection(BaseModel):
    # ... existing fields ...
    prior_gaps_filled: int = Field(
        default=0,
        description=(
            "How many gaps from the previous round were answered by this "
            "round's findings. 0 if no prior gaps existed or none were filled."
        ),
    )
```

**prompts.py** — Add field criteria for `prior_gaps_filled` to
`researcher_reflection_prompt`:

```
prior_gaps_filled:
- Count how many gaps listed in prior_context's "Gaps identified last round"
  were substantively answered by this round's findings.
- 0 if there were no prior gaps, or if none were filled.
- A gap is "filled" if the findings now contain a meaningful answer, even if
  partial. A gap is "unfilled" if the search returned nothing relevant.
```

**reflect.py** — Add dead-end check in routing logic:

```python
# After reflection, before routing decision
prior_gap_count = len(state.get("current_gaps", []))
dead_end = (
    prior_gap_count > 0
    and reflection.prior_gaps_filled == 0
    and iteration >= 2
)

if dead_end and not should_stop:
    if iteration >= 3:
        # Reformulation already attempted — force exit
        logger.warning(
            "Dead end persists after reformulation — forcing exit "
            "(round %d, %d unfilled gaps)", iteration, prior_gap_count
        )
        should_stop = True
    else:
        # First dead end — inject reformulation guidance
        logger.info(
            "Dead end detected (round %d, %d gaps unfilled) — "
            "injecting reformulation guidance", iteration, prior_gap_count
        )
        reformulation_note = (
            "\n\nDEAD END: The gaps above persisted despite targeted searches. "
            "Do NOT repeat similar queries. Instead:\n"
            "- Try synonyms or alternative terminology\n"
            "- Approach from a different angle or adjacent topic\n"
            "- Search for the information in different source types "
            "(academic papers, government reports, industry analyses)\n"
            "- Broaden or narrow the scope to find related information"
        )
        # Appended to the formatted reflection passed to the researcher
```

**Tests** (unit):
- `test_dead_end_forces_exit_after_two_rounds` — mock reflection with
  `prior_gaps_filled=0`, current_gaps non-empty, iteration=3 → routes to summarize
- `test_dead_end_reformulates_on_first_detection` — same but iteration=2 →
  routes to researcher with reformulation guidance in last_reflection
- `test_no_dead_end_when_gaps_filled` — `prior_gaps_filled > 0` → normal routing
- `test_no_dead_end_on_first_round` — iteration=1, no prior gaps → normal routing

---

### Step 1 — Contradiction dedup in reflect node

**File**: `src/deep_research/nodes/researcher/reflect.py`

Before appending to `accumulated_contradictions`, filter out duplicates:

```python
# Deduplicate contradictions before accumulating
existing = {c.lower().strip() for c in state.get("accumulated_contradictions", [])}
new_contradictions = [
    c for c in reflection.contradictions
    if c.lower().strip() not in existing
]

accumulation_update = {
    "accumulated_findings": reflection.key_findings,
    "accumulated_contradictions": new_contradictions,  # deduped
    "current_gaps": reflection.missing_info,
    "research_iterations": iteration,
}
```

**Tests** (unit):
- `test_contradiction_dedup_exact` — same string in state and reflection →
  not re-appended
- `test_contradiction_dedup_case_insensitive` — "Market size conflicts" vs
  "market size conflicts" → deduped
- `test_contradiction_dedup_preserves_new` — new contradictions still appended

---

### Step 2 — `[unverified]` for hallucinated citations

**File**: `src/deep_research/helpers/source_store.py`

Change `resolve_citations` behavior for unknown IDs:

```python
def _replace_match(match: re.Match) -> str:
    ids_str = match.group(1)
    sids = re.split(r"\s*,\s*", ids_str)
    nums = [str(id_to_num[sid]) for sid in sids if sid in id_to_num]
    if not nums:
        return "[unverified]"  # was: "" (silent removal)
    return "[" + ", ".join(nums) + "]"
```

For mixed brackets (some known, some unknown): keep the known IDs, still
resolve them. Only replace the entire bracket with `[unverified]` when
ALL IDs in the bracket are unknown.

**Tests** (update existing + add new):
- Update `test_resolve_unknown_citation` — expect `[unverified]` instead of `""`
- Update `test_resolve_mixed_known_unknown` — mixed bracket still resolves known IDs
  (behavior unchanged for mixed case)
- Add `test_resolve_all_unknown_bracket` — `[unknown1, unknown2]` → `[unverified]`

---

### Step 3 — Report prompt: visual hierarchy + summary section

**File**: `src/deep_research/prompts.py`

Two targeted changes to `final_report_prompt`, both addressing Increment 6
observations:

**3a. Visual hierarchy for contradictions** — replace instruction 4:

```
4. When research_metadata lists contradictions, present them with clear
   visual distinction — use a subheading (### Conflicting Evidence) and
   present each side with its source. Do not bury contradictions in
   running text.
```

**3b. Summary "Areas for Further Research"** — add instruction:

```
8. At the end of the main body, include a brief "Areas for Further
   Research" section summarizing all gaps and open questions in one
   place, even if they were mentioned inline. This gives the reader a
   quick reference for follow-up.
```

**Tests**: Integration only — observe report output for visual distinction
and summary section.

---

### Step 4 — Integration testing + observation

Run the full integration suite and a complex manual query to observe:

1. **Dead-end detection**: Does reformulation trigger? Does forced exit work?
   - Look for "Dead end detected" and "Dead end persists" in logs
   - Check if the researcher tries different angles after reformulation
2. **Contradiction dedup**: Are duplicate contradictions gone from metadata?
3. **`[unverified]`**: Do hallucinated citations get flagged visibly?
4. **Report structure**: Are contradictions presented with clear hierarchy?
   Does the report have a summary "Areas for Further Research"?

Document observations. If the LLM isn't handling contradiction resolution,
source authority, or corroboration well enough, add targeted prompt fixes
as a follow-up step.

---

## Files Changed

| File | Change |
|------|--------|
| `src/deep_research/models.py` | Add `prior_gaps_filled: int` to `ResearchReflection` |
| `src/deep_research/nodes/researcher/reflect.py` | Dead-end detection routing + contradiction dedup |
| `src/deep_research/helpers/source_store.py` | `[unverified]` for unknown citations |
| `src/deep_research/prompts.py` | `prior_gaps_filled` field criteria + report visual hierarchy + summary section |
| `tests/test_reflect.py` (new) | Dead-end detection + contradiction dedup unit tests |
| `tests/test_source_store.py` | Update/add `[unverified]` tests |

## Files NOT Changed

| File | Why not |
|------|---------|
| `state.py` | No new state fields — `prior_gaps_filled` is on ResearchReflection (model output), not state |
| `configuration.py` | No new config — dead-end thresholds use iteration count already tracked |
| `coordinator/reflect.py` | No changes — LLM already handles cross-topic assessment adequately |
| `tools/search/base.py` | No authority tag — LLM already sees URLs |
| `nodes/report.py` | No code changes — prompt changes handle report improvements |
| `nodes/researcher/adapter.py` | Unchanged — contradiction dedup happens upstream in reflect |
| `nodes/researcher/researcher.py` | No changes — LLM already acts on reflection guidance |

## Deferred (add only if integration testing shows need)

- **Prompt: contradiction resolution in next_queries** — if LLM doesn't
  naturally include resolution queries when contradictions exist
- **Prompt: multi-source corroboration** — if LLM doesn't note when multiple
  sources agree
- **Prompt: source authority** — if LLM ignores .gov/.edu credibility in
  contradiction evaluation
- **Prompt: coordinator low-confidence** — if coordinator doesn't handle
  insufficient researchers well
- **URL authority tag in `_format_results`** — if prompt-only isn't enough
- **Semantic contradiction dedup** — if exact match doesn't catch enough dupes
- **Coordinator topic reassignment** — dispatch same topic with different angle
