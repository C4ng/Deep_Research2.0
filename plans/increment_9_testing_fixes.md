# Increment 9 — Testing & Fixes

**Goal**: End-to-end testing of the pipeline, capturing issues found and fixes applied.
**Status**: Complete
**Depends on**: Increment 8 (Provider Flexibility)

---

## Issue 1 — Fail-fast on tool errors ✓

**Found**: All search API calls fail (quota exceeded), but system wastes ~40 LLM
calls iterating through 3 reflection rounds × 5 researchers × 2 coordinator rounds.

**Root cause**: Errors caught in `_execute_tool_safely()` become plain strings.
No node distinguishes "tool error" from "tool returned data."

**Fix**: Shared `helpers/errors.py` with `TOOL_ERROR_PREFIX` and
`all_tools_failed(messages)`. Researcher `reflect()` checks before calling
reflection LLM — if all tools failed, skips LLM, routes to summarize with
`final_knowledge_state="error"`. Coordinator `coordinator_reflect()` checks
`latest_round_result_count` state field — if zero, exits without LLM call.

**Files**: `helpers/errors.py` (new), `nodes/researcher/reflect.py`,
`nodes/coordinator/reflect.py`, `nodes/coordinator/coordinator.py`, `state.py`

---

## Issue 2 — HITL re-enters clarify on resume

**Found**: After user approves the research brief ("good to go"), the graph
re-enters `clarify` → `write_brief` instead of proceeding directly to
coordinator. Wastes 2 LLM calls (re-clarification + re-generating brief).

**Root cause**: Graph uses `Command(goto="__end__")` for HITL pause. This
terminates the graph run. Next `ainvoke()` starts fresh from `START → clarify`.
The checkpointer saves state, but doesn't save "where we were in the graph."

**Log evidence**:

```
brief: Showing brief to user for review
You: good to go
clarify: Assessing whether clarification is needed     ← wasted
clarify: No clarification needed, proceeding to write_brief
brief: Brief generated: ... (ready=True)               ← re-generated
brief: Proceeding to coordinator
```

**Proposed fix**: Conditional routing at START based on state. If `research_brief`
exists in state, skip clarify and go directly to `write_brief`. Clarify never sets
`research_brief` — only write_brief does — so this cleanly distinguishes
"resuming for clarification" vs "resuming for brief review."

`interrupt()` is the LangGraph-native HITL pattern but requires node restructuring
and has a gotcha: nodes replay from scratch on resume, so the brief LLM call runs
again (potentially producing a different result). Conditional routing is simpler,
no node changes, same outcome.

### Steps

**Step 1** — Add `_route_start()` in `graph/graph.py`

```python
def _route_start(state: AgentState) -> str:
    """Skip clarify when resuming for brief review."""
    if state.get("research_brief"):
        return "write_brief"
    return "clarify"
```

Replace `graph.add_edge(START, "clarify")` with:

```python
graph.add_conditional_edges(START, _route_start, ["clarify", "write_brief"])
```

**Step 2** — Skip LLM call in `write_brief` on approval

If `prior_brief` exists and the user's latest message is clearly an approval,
skip the LLM call entirely and proceed with the existing brief. Only re-invoke
the LLM when the user gives revision feedback.

Detection: check last HumanMessage content — short messages without specific
change requests indicate approval. The LLM already does this (sets
`ready_to_proceed=True`) but costs a full LLM call just to parse intent.

Simpler: if `prior_brief` exists and last message has no substantive feedback
(under ~20 words, no question marks, no "change/modify/add/remove" keywords),
treat as approval.

Actually simpler: always re-invoke the LLM but with a tiny model/low tokens —
or just accept the 1 LLM call as the cost of reliable intent detection. The
real savings is skipping clarify (step 1).

Decision: keep the LLM call for intent detection (it's one call, and keyword
heuristics are fragile). The main fix is step 1 — skipping clarify saves
1 wasted LLM call per resume.

**Step 3** — Update `scripts/run.py` to only print new messages

Current runner prints ALL AI messages on every resume, including old ones.
Track the message count and only print messages added since last invocation.

**Step 4** — Tests

- Verify: resume after brief review goes directly to write_brief (check logs,
no "Assessing whether clarification is needed")
- Verify: resume after clarify question still routes to clarify correctly
- Verify: first run with no prior state routes to clarify as before

**Files**: `graph/graph.py`, `nodes/brief.py`, `scripts/run.py`

**Status**: Fixed — `_route_start()` added, `is_simple` removed (see Issue 5)

---

## Issue 3 — No graceful degradation when search fails

**Found**: When Tavily quota is exceeded, the system produces an empty report
saying "no information was retrieved." It doesn't try alternative search
providers or fall back to the LLM's own knowledge.

**Solution**: LLM knowledge fallback. When search fails, the researcher's
summarizer generates notes from LLM training knowledge with provenance
markers so the final report knows the source.

**DuckDuckGo attempted and removed**: DDG was initially added as a free
search fallback between the paid provider and LLM knowledge. Removed after
live testing showed it provided no value — most calls returned empty (rate-
limited under concurrent load), and the few results that came back were
garbage (dictionary definitions of "solid" instead of solid-state battery
research). The LLM knowledge fallback produces better content. Simplifies
the fallback chain to: paid provider fails → LLM knowledge directly.

### Step 1 — LLM knowledge fallback in summarizer ✓

**File**: `src/deep_research/nodes/researcher/summarizer.py`

Summarizer filters junk ToolMessages (`_is_junk()` checks error/no-result
prefixes). When no usable tool results remain, falls back to
`_generate_from_llm_knowledge()` which generates notes from LLM training
data marked with `[source: LLM training knowledge]`.

**File**: `src/deep_research/helpers/errors.py` — Added `NO_RESULTS_PREFIX`
constant and `no_search_results()` detection function.

**File**: `src/deep_research/nodes/researcher/reflect.py` — Early exit on
`no_search_results()` alongside existing `all_tools_failed()`, routing to
summarize with `final_knowledge_state="unavailable"`.

**File**: `src/deep_research/prompts.py` — Added `llm_knowledge_fallback_prompt`
(writes substantively, marks output with `[source: LLM training knowledge]`).
Added instruction 8 to `final_report_prompt` for handling LLM-sourced notes
with provenance.

**Status**: Done

### Step 2 — Fix `knowledge_state` validation error (from live test) ✓

**Found**: Researcher reflect sets `final_knowledge_state="error"` when all
tools fail. Coordinator `dispatch_research` passes this to `ResearchResult`
which only accepts `Literal["insufficient", "partial", "sufficient"]` →
Pydantic validation error crashes the coordinator.

**Fix**: Added `"unavailable"` to knowledge_state Literal in
`ResearchReflection`, `ResearchResult`, and `CoordinatorReflection`.
Changed early-exit path to set `"unavailable"` instead of `"error"`.
Updated prompts to document the new value. Coordinator reflection now
exits immediately when `knowledge_state == "unavailable"` (retrying
search when search is unavailable is pointless).

**Status**: Done

### Step 3 — Tests ✓

**File**: `tests/test_nodes.py`

- 3 `no_search_results` detection tests (all errors, mixed, no messages)
- 1 reflect early exit on no results test
- 3 summarizer LLM fallback tests (empty messages, error state, junk
  filtering)
- 1 `_execute_tool_safely` error handling test
- 2 `knowledge_state` "unavailable" acceptance tests

**Status**: Done — 52 tests passing

### Files changed

| File | Change |
|------|--------|
| `src/deep_research/nodes/researcher/summarizer.py` | LLM knowledge fallback, `_is_junk()` filter |
| `src/deep_research/nodes/researcher/reflect.py` | `no_search_results()` early exit, `"unavailable"` state |
| `src/deep_research/nodes/coordinator/reflect.py` | Exit on `knowledge_state == "unavailable"` |
| `src/deep_research/helpers/errors.py` | `NO_RESULTS_PREFIX`, `no_search_results()` |
| `src/deep_research/prompts.py` | `llm_knowledge_fallback_prompt`, final report instruction 8, knowledge_state docs |
| `src/deep_research/models.py` | Add `"unavailable"` to Literal types |
| `tests/test_nodes.py` | 11 unit tests |

---

## Issue 4 — Brief message formatting

**Found**: Brief shown to user for review had no framing — just raw content
dumped as an AI message with no indication it was a draft for approval.

**Fix**: Added review framing to the AIMessage in `write_brief`:
"Here's the research plan I've drafted. Let me know if you'd like to adjust..."

**Files**: `nodes/brief.py`

**Status**: Fixed

---

## Issue 5 — `is_simple` binary classification unstable

**Found**: The `is_simple` field in `ResearchBrief` flipped between LLM calls —
`is_simple=True` on first generation, `is_simple=False` on re-generation after
user approval. This caused the brief node to route to `researcher` (single) on
first call but `coordinator` (multi-topic) on re-invocation, producing
inconsistent behavior.

**Root cause**: Non-deterministic LLM output. A binary "is this question simple?"
classification is inherently unstable — the same question can be judged either
way depending on the LLM run.

**Fix**: Remove `is_simple` entirely. Always route to coordinator. The coordinator
handles both simple and complex queries — a simple query just gets fewer subtopics.

**Changes**:

- `models.py` — removed `is_simple` field from `ResearchBrief`
- `prompts.py` — replaced simplicity assessment with "Do NOT include subtopics"
- `state.py` — removed `is_simple` from `AgentState`
- `brief.py` — always routes to `coordinator`, removed `researcher` routing
- `graph/graph.py` — removed `researcher` node and edge
- `tests/test_nodes.py` — removed/rewritten 4 brief routing tests
- `tests/test_integration.py` — removed `is_simple` from `INITIAL_STATE`
- `scripts/compare_serper.py`, `scripts/compare_brave.py` — removed from state
- `scripts/test_provider_swap.py` — removed `brief.is_simple` print

**Status**: Fixed


