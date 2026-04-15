# Increment 9 — Testing & Fixes
**Goal**: End-to-end testing of the pipeline, capturing issues found and fixes applied.
**Status**: In progress
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

**Two levels of fallback needed**:

1. **Search provider failover**: If configured provider fails, try other
   providers whose API keys are available in env (e.g., `BRAVE_API_KEY`,
   `SERPER_API_KEY` are set but Brave/Serper aren't the configured provider).
   The tool registry already knows all providers — it just needs to try
   alternatives on failure.

2. **LLM knowledge fallback**: If ALL search providers fail, the researcher
   should still produce a report from the LLM's training knowledge rather
   than returning empty. The model knows about LangGraph and CrewAI — it
   just wasn't asked because the system assumes search is the only source.

**Proposed fix**:

- **Provider failover** (in `tools/registry.py` or `tools/search/base.py`):
  On tool error, check env for other provider API keys. If found, retry
  with an alternative provider. This is transparent to the researcher node.

- **LLM knowledge fallback** (in `reflect.py` or `summarizer.py`):
  When `final_knowledge_state == "error"`, prompt the summarizer to generate
  notes from the LLM's own knowledge with a clear disclaimer ("based on
  training data, not live search results").

**Status**: Not started

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

---

## Files changed (all issues)

| File | Issue | Change |
|------|-------|--------|
| `src/deep_research/helpers/errors.py` | #1 | **NEW** — `TOOL_ERROR_PREFIX`, `all_tools_failed()` |
| `src/deep_research/nodes/researcher/researcher.py` | #1 | Import `TOOL_ERROR_PREFIX` from helpers |
| `src/deep_research/nodes/researcher/reflect.py` | #1 | Early exit via `all_tools_failed()` |
| `src/deep_research/nodes/coordinator/coordinator.py` | #1 | Return `latest_round_result_count` |
| `src/deep_research/nodes/coordinator/reflect.py` | #1 | Early exit on zero results |
| `src/deep_research/state.py` | #1, #5 | Add `latest_round_result_count`; remove `is_simple` |
| `src/deep_research/graph/graph.py` | #2, #5 | `_route_start()` conditional routing; remove `researcher` node |
| `src/deep_research/nodes/brief.py` | #4, #5 | Review framing; remove `is_simple` routing, always coordinator |
| `src/deep_research/models.py` | #5 | Remove `is_simple` from `ResearchBrief` |
| `src/deep_research/prompts.py` | #5 | Remove simplicity assessment section |
| `scripts/run.py` | #2 | Only print new messages on resume |
| `scripts/compare_serper.py` | #5 | Remove `is_simple` from state |
| `scripts/compare_brave.py` | #5 | Remove `is_simple` from state |
| `scripts/test_provider_swap.py` | #5 | Remove `brief.is_simple` print |
| `tests/test_nodes.py` | #1, #5 | 7 fail-fast tests; remove/rewrite brief routing tests |
| `tests/test_integration.py` | #5 | Remove `is_simple` from `INITIAL_STATE` |
