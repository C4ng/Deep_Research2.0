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

**Proposed fix**: Use LangGraph's `interrupt()` instead of `Command(goto="__end__")`.
`interrupt()` pauses execution mid-node — resume picks up exactly where it left off.
No re-entry through clarify.

Affects: `nodes/clarify.py`, `nodes/brief.py`, `graph/graph.py`

**Status**: Not started

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

**Status**: Fixed (this session)

---

## Files changed (all issues)

| File | Issue | Change |
|------|-------|--------|
| `src/deep_research/helpers/errors.py` | #1 | **NEW** — `TOOL_ERROR_PREFIX`, `all_tools_failed()` |
| `src/deep_research/nodes/researcher/researcher.py` | #1 | Import `TOOL_ERROR_PREFIX` from helpers |
| `src/deep_research/nodes/researcher/reflect.py` | #1 | Early exit via `all_tools_failed()` |
| `src/deep_research/nodes/coordinator/coordinator.py` | #1 | Return `latest_round_result_count` |
| `src/deep_research/nodes/coordinator/reflect.py` | #1 | Early exit on zero results |
| `src/deep_research/state.py` | #1 | Add `latest_round_result_count` to CoordinatorState |
| `src/deep_research/nodes/brief.py` | #4 | Add review framing to brief message |
| `tests/test_nodes.py` | #1 | 7 fail-fast tests |
