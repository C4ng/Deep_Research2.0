# Increment 7 — Research Quality + Report Polish

**Goal**: Dead-end detection, contradiction dedup, hallucinated citation marking, and report structure fixes.
**Status**: Steps 0–3 implemented, Step 4 observations complete, contradiction redesign in progress
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
| Contradiction dedup | **LLM-managed overwrite** | Integration testing showed exact string dedup insufficient — LLM rephrases across rounds. Changed to overwrite reducer: LLM outputs full canonical list each round, handling semantic dedup/merge/resolution naturally. |
| Hallucinated citations | **`[unverified]` replacement** | Currently `resolve_citations` silently removes unknown IDs. Replace with `[unverified]` so the reader sees which claims lost their citation. |
| Report structure | **Prompt refinement only** | Two targeted prompt changes: `### Conflicting Evidence` with analysis guidance + `### Open Questions` within topic sections (not standalone at end). |

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

### Step 0 — Dead-end detection: model + routing ✅

**Files**: `src/deep_research/models.py`, `src/deep_research/prompts.py`, `src/deep_research/nodes/researcher/reflect.py`
**Commit**: `74adb9e`

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

### Step 1 — Contradiction management: LLM-managed overwrite 🔄

**Files**: `src/deep_research/state.py`, `src/deep_research/nodes/researcher/reflect.py`, `src/deep_research/prompts.py`

**Evolution**: Originally implemented as programmatic exact-string dedup with
append reducer (`e8e2203`). Integration testing revealed near-duplicate
contradictions survive string matching because the LLM rephrases slightly
across rounds. Redesigned to use LLM semantic reasoning with overwrite
semantics — the LLM outputs the full canonical contradiction list each round,
merging/deduplicating/resolving as it sees fit.

**state.py** — Changed `accumulated_contradictions` from append reducer to overwrite:
```python
# Before: Annotated[list[str], operator.add]  (append)
# After:  list[str]                            (overwrite)
accumulated_contradictions: list[str]
```

**reflect.py** — Removed programmatic dedup, pass LLM output directly:
```python
accumulation_update = {
    "accumulated_findings": reflection.key_findings,
    "accumulated_contradictions": reflection.contradictions,  # LLM manages the full list
    "current_gaps": reflection.missing_info,
    "research_iterations": iteration,
}
```

**prompts.py** — Updated `contradictions` field criteria to instruct full-list output:
```
contradictions:
- Output the COMPLETE current list of all known contradictions — not just
  new ones from this round. This field overwrites the previous round's list.
- Merge, deduplicate, or remove contradictions that have been resolved by
  new evidence.
```

The LLM already sees prior contradictions via `_format_accumulated_context()`
("Previously identified contradictions:"), so it has full context to produce
an updated canonical list.

**Tests** (unit):
- `test_contradictions_overwrite_replaces_previous` — LLM consolidates prior entries
- `test_contradictions_overwrite_can_clear` — LLM resolves all → empty list
- `test_contradictions_overwrite_passes_through` — new contradictions pass through unfiltered

---

### Step 2 — `[unverified]` for hallucinated citations ✅

**File**: `src/deep_research/helpers/source_store.py`
**Commit**: `12d63ec`

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

### Step 3 — Report prompt: visual hierarchy + contextual open questions ✅

**File**: `src/deep_research/prompts.py`
**Commit**: `c71a5b4`

Two targeted changes to `final_report_prompt`, both addressing Increment 6
observations:

**3a. Visual hierarchy for contradictions** — replaced instruction 4 with
structured analysis guidance: `### Conflicting Evidence` subsections within
topic sections, with analysis of why sources differ (methodology, timeframe,
scope) and guidance on which claim has stronger support.

**3b. Contextual open questions** — replaced standalone "Areas for Further
Research" at the end (user rejected) with `### Open Questions` subsections
within each relevant topic section. The metadata labels (not investigated,
searched but not found, partial coverage) are used to explain what remains
unknown and why. This keeps future directions near the corresponding context
rather than buried at the bottom.

---

### Step 4 — Integration testing + observation ✅

Ran full end-to-end test with complex query (intermittent fasting health effects).
170 sources cited.

**Dead-end detection — effective**:
- `prior_gaps_filled` values ranged 0–3 across rounds, showing the LLM assesses
  gap filling correctly.
- Every time reformulation guidance was injected ("DEAD END: ..."), the researcher
  filled at least some previously-stuck gaps in the next round (prior_gaps_filled=1–3).
- No "dead end persists after reformulation" events — reformulation worked every
  time. The "force exit after 2 strikes" path was never triggered; max_iterations (3)
  catches those cases first. This is fine — the reformulation guidance is pulling its weight.

**[unverified] markers — working as designed**:
- 3 `[unverified]` markers in the final report, 17 hallucinated citations caught total.
- Most hallucinated IDs appeared alongside known IDs in mixed brackets — the known
  ones were kept, only unknowns dropped. `[unverified]` only appears when ALL citations
  in a bracket were hallucinated, flagging the riskiest claims for the reader.

**Contradiction quality in report — substantially improved**:
- Dedicated `### Conflicting Evidence` subsections with structured analysis (why sources
  differ — methodology, study design, timeframe) and guidance on which claim has stronger
  support. Example: calorie reduction vs. eating window debate includes the ChronoFast
  trial (2025) as key evidence and explains isocaloric vs ad libitum study differences.

**Open Questions sections — well-placed**:
- 4 `### Open Questions` subsections within relevant topic sections. The LLM correctly
  translated metadata labels (e.g., "searched but not found") into reader-facing
  explanations of what remains unknown and why.

**Contradiction dedup in metadata — limitation identified**:
- Exact string dedup catches verbatim copies but near-duplicates survive (LLM
  rephrases the same contradiction slightly across rounds). Led to the contradiction
  management redesign (Step 1 → overwrite semantics).

---

## Files Changed

| File | Change |
|------|--------|
| `src/deep_research/models.py` | Add `prior_gaps_filled: int` to `ResearchReflection` |
| `src/deep_research/state.py` | `accumulated_contradictions` from append reducer to overwrite |
| `src/deep_research/nodes/researcher/reflect.py` | Dead-end detection routing + contradiction overwrite (removed programmatic dedup) |
| `src/deep_research/helpers/source_store.py` | `[unverified]` for unknown citations |
| `src/deep_research/prompts.py` | `prior_gaps_filled` field criteria, contradiction overwrite instruction, report visual hierarchy + open questions |
| `tests/test_nodes.py` | Dead-end detection + contradiction overwrite unit tests |
| `tests/test_source_store.py` | Update/add `[unverified]` tests |

## Deferred (add only if future testing shows need)

Integration testing showed the LLM handles these adequately without explicit prompting:

- ~~**Semantic contradiction dedup**~~ — resolved by switching to LLM-managed overwrite
- **Prompt: contradiction resolution in next_queries** — LLM naturally includes
  resolution queries when it sees contradictions in context
- **Prompt: multi-source corroboration** — LLM notes agreement/disagreement
  when it sees multiple sources on the same topic
- **Prompt: source authority** — LLM sees full URLs and correctly weights
  .gov/.edu sources in contradiction analysis
- **Prompt: coordinator low-confidence** — coordinator reasons about
  knowledge_state adequately
- **URL authority tag in `_format_results`** — not needed, LLM already sees URLs
- **Coordinator topic reassignment** — dispatch same topic with different angle
