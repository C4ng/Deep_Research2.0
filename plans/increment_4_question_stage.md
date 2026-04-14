# Increment 4 — Question Stage (Clarification + Scoping)
**Goal**: Well-formed research questions through user interaction, with adaptive routing based on question needs.
**Status**: Complete ✅

## Overview

The question stage sits at the front of the pipeline, before research begins. It has four responsibilities:

1. **Clarification**: Resolve ambiguity by asking the user questions (optional, config-gated)
2. **Brief generation**: Transform the (now-clear) user query into a research question with strategic approach guidance — NOT decomposed into exact topics (that's the coordinator's job)
3. **Human review**: Let the user review and modify the research plan before resources are spent
4. **Routing**: Simple questions go directly to a single researcher, bypassing the coordinator entirely

**Main graph**:
```
START → clarify → write_brief → researcher  → final_report → END
                              → coordinator → final_report → END
```

Both `clarify` and `write_brief` are user interaction nodes using the same graph-exit pattern. Each reads messages, calls the LLM, and routes via `Command` — either exiting to `__end__` (returning output to the user for feedback) or proceeding to the next node.

---

## Motivation

### Problem 1: Brief does topic decomposition (coordinator's job)

Original `ResearchBrief` had `research_questions: list[str]` and `key_topics: list[str]`. The brief decomposed the query into subtopics, then the coordinator decomposed *again*. This was redundant — the brief was prematurely structuring the problem.

The brief is now a single, well-articulated question that preserves the user's intent, constraints, and explicitly marks what's unspecified. The coordinator decides how to decompose.

### Problem 2: Ambiguous queries go straight to research

"quantum computing" could mean anything. A clarification step catches ambiguity before wasting research resources.

### Problem 3: All queries get the same treatment

A simple factual question ("What is the latest React version?") got 5 researchers and multi-round reflection. Now `is_simple` routes these directly to a single researcher.

### Problem 4: No user involvement before research

The user had no chance to shape the research plan. Now the brief is a draft plan the user reviews and can modify — adding angles, adjusting priorities, or approving as-is.

---

## Key Design Decisions

**Clarification is optional and conservative**: Gated by `allow_clarification` config (default: true). When enabled, the model is instructed to ask only when genuinely needed — acronyms, unclear scope, critical ambiguity. One clarifying question per invocation, almost never two. The user should feel helped, not interrogated.

**Both nodes use the same graph-exit pattern**: Clarify and write_brief are user ↔ AI communication nodes using `Command(goto="__end__")`. Clarify exits with a question; write_brief exits with the draft plan. On re-invocation, each reads the full message history and decides whether to proceed or iterate. No `interrupt()` — the graph-exit + re-invocation pattern is simpler and works without a checkpointer.

**One prompt handles both fresh generation and revision**: The `research_brief_prompt` has optional `{prior_brief}` and `{feedback}` sections. When empty, it generates fresh. When populated, it revises. The LLM also sets `ready_to_proceed` to signal whether the user approved or requested changes — this drives routing without any heuristic code.

**write_brief owns routing**: The brief node determines `is_simple` and `ready_to_proceed`, and routes directly via `Command`. No separate conditional edge, routing function, or review node. The node makes all decisions and routes itself.

**Brief includes strategic guidance**: The `approach` field is not just what to research but how — angles to cover, breadth vs depth, priorities. The coordinator reads this as a starting point for decomposition. Strategy flows: brief approach → coordinator decomposition → researcher topic/context → search queries. The user can shape strategy by modifying the approach during review.

**Coordinator executes strategy, doesn't create it**: The coordinator reads the brief's strategic guidance and decomposes into topics accordingly. It no longer reasons about question type or approach from scratch — that's already in the brief.

**graph.py is pure orchestration**: Node registration, edges, compilation. No LLM calls, no prompts, no state mapping. All node logic lives in node modules.

---

## Schemas

### ClarifyOutput

```python
class ClarifyOutput(BaseModel):
    need_clarification: bool      # whether to ask the user a question
    question: str                 # the clarifying question (empty if not needed)
    verification: str             # acknowledgement before research (empty if clarifying)
```

### ResearchBrief

```python
class ResearchBrief(BaseModel):
    title: str                    # concise title
    research_question: str        # detailed research question with constraints
    approach: str                 # strategic guidance: angles, breadth/depth, priorities
    is_simple: bool               # whether a single researcher can handle this
    ready_to_proceed: bool        # whether the user approved the brief or requested changes
```

`ready_to_proceed` (default: False) is the LLM's routing signal. On a fresh brief it's False (needs review). When the user re-invokes, the LLM reads the prior brief + user message and sets it True (approval) or False (changes requested). This replaces any heuristic approval detection.

---

## Component Input/Output Map

### clarify
```
IN:  messages: [HumanMessage(user query)]
OUT: Command(goto="__end__")      → with question as AIMessage (needs clarification)
     Command(goto="write_brief")  → with verification as AIMessage (clear question)
```

### write_brief
```
IN:  messages: [Human(query), AI(verification), ...]
     research_brief: "" or prior brief string
     
LLM: research_brief_prompt(messages, prior_brief, feedback)
   → ResearchBrief(title, question, approach, is_simple, ready_to_proceed)

OUT: Command(goto="__end__")       → first draft or revision with changes (review cycle)
     Command(goto="researcher")    → approved + is_simple=True
     Command(goto="coordinator")   → approved + is_simple=False
```

### coordinator
```
IN:  research_brief: "Title: ...\n\nQuestion...\n\nApproach: - bullet points..."
     research_results: [] (first round) or [ResearchResult, ...] (follow-up)

LLM sees: coordinator_system_prompt with {research_brief} injected
        → reads Title + question + Approach
        → decomposes into subtopics

OUT: dispatch_research(topic, context) tool calls
     → each spawns a researcher subgraph
```

### researcher (inside subgraph, per subtopic)
```
IN:  research_topic: "Subtopic\n\nContext: why this matters..."

LLM: research_system_prompt with {topic}
   → calls tavily_search(query) up to max_searches_per_round per round
   → up to max_research_iterations rounds (search → reflect loop)

OUT: notes: compressed findings with citations
     accumulated_findings, current_gaps, contradictions
     → wrapped as ResearchResult for coordinator
```

### final_report
```
IN:  research_brief + merged notes from all researchers
OUT: final_report: markdown report with citations
```

---

## Information Flow

```
User sends question
  → clarify node (reads messages)
  ├─ [need_clarification=true] → Command(goto="__end__") with question
  │     → user answers → graph re-invoked → clarify sees full history → proceeds
  └─ [need_clarification=false] → Command(goto="write_brief") with verification
       → write_brief node (reads messages + state)
            → generates ResearchBrief (structured output)
            ├─ [allow_human_review and not ready_to_proceed]
            │     → Command(goto="__end__") with brief as AIMessage
            │     → user reviews, gives feedback or approves
            │     → graph re-invoked → clarify skips → write_brief reads history
            │     → LLM revises brief, sets ready_to_proceed
            │     → feedback → exit again / approval → proceed
            └─ [ready_to_proceed or review disabled]
                 ├─ [is_simple=true]  → Command(goto="researcher")
                 └─ [is_simple=false] → Command(goto="coordinator")
                      → ... research pipeline ... → final_report → END
```

### Strategy flow through the system

The approach field in the brief flows indirectly to search:

```
User question → brief prompt → approach: "focus on hardware, breadth over depth"
  → coordinator reads approach → decomposes: "quantum hardware", "error correction", ...
    → each researcher gets topic + context from coordinator
      → researcher LLM decides search queries based on topic
        → tavily_search("quantum hardware superconducting 2025")
```

The researcher never sees the approach directly — it's translated by the coordinator into specific topic assignments with context. This is intentional: researchers are focused and scoped, they don't need strategic context.

---

## Steps

### Step 0 — Simplify ResearchBrief schema + update brief prompt ✅

Reform the brief to produce a single research question instead of decomposed subtopics.

- `ResearchBrief`: drop `research_questions[]` and `key_topics[]`, add `research_question: str`
- Rewrite `research_brief_prompt` with new guidelines (specificity, open-ended dimensions, no assumptions)
- Update `write_research_brief` to format the new schema
- Update tests

### Step 1 — Add is_simple routing + researcher adapter ✅

Simple questions bypass the coordinator and go directly to a single researcher.

- Add `is_simple: bool` to `ResearchBrief` and `AgentState`
- Add `route_by_complexity` conditional edge after `write_brief`
- Add `run_single_researcher` adapter in `graph.py`
- Tests for routing function

### Step 2 — Update coordinator prompt with strategy examples ✅

Add few-shot examples showing how different question types map to different decomposition strategies.

- Coordinator prompt instruction 1: examples for comparison, survey, analysis, pros/cons
- Updated trigger message for round 1
- Later removed: strategy examples moved to brief's approach field instead (step 5-6)

### Step 3 — Add ClarifyOutput schema + clarify prompt ✅

- `ClarifyOutput` schema, `clarify_prompt`, `allow_clarification` config

### Step 4 — Implement clarify node + wire into graph ✅

- `clarify_with_user` node with graph exit pattern
- Wired: `START → clarify → write_brief → ...`

### Step 5 — Add approach field to ResearchBrief ✅

Enrich the brief with strategic guidance so the user can review and modify the research plan.

- `ResearchBrief`: add `approach: str`
- `research_brief_prompt`: add approach generation guidelines (bullet points)
- `nodes/brief.py`: include approach in formatted brief_str

### Step 6 — Remove strategy reasoning from coordinator prompt ✅

- Remove few-shot strategy examples from coordinator prompt
- Coordinator reads brief's approach as starting point, applies own judgment

### Step 7 — Add human review config ✅

- Added `allow_human_review: bool` to Configuration (default: true)

### Step 8 — Extend research_brief_prompt for revision ✅

The prompt currently only generates a fresh brief. Extend it to also handle
revision when prior brief + user feedback are present in context.

**prompts.py**:
- Add optional `{prior_brief}` and `{feedback}` placeholders to `research_brief_prompt`
- When both are empty: prompt works as before (fresh generation)
- When populated: prompt instructs the LLM to revise the prior brief
  incorporating user feedback, keeping the same ResearchBrief structured output
- Remove `revise_brief_prompt` (added prematurely in an earlier edit)

No behavior change yet — brief node still calls the prompt the same way,
just with empty prior_brief/feedback.

### Step 9 — Rewrite write_research_brief with Command routing ✅

Redesign the brief node to match the clarify node pattern: read messages,
call LLM, route via Command.

**nodes/brief.py** — `write_research_brief` returns `Command`:
- Read messages and check state for existing `research_brief`
- Build prompt: if `research_brief` already exists in state AND last message
  is user feedback → populate `{prior_brief}` and `{feedback}` for revision.
  Otherwise → fresh generation (empty prior_brief/feedback)
- Call LLM with structured output (ResearchBrief) — same call for both cases
- Format brief string from structured output
- **Routing decision**:
  - If `allow_human_review` and not `ready_to_proceed`
    → `Command(goto="__end__", update={research_brief, is_simple, messages: [AIMessage with brief]})`
    The user sees the brief and can respond with feedback or approval
  - If review disabled, or `ready_to_proceed=True`
    → `Command(goto="researcher" or "coordinator", update={research_brief, is_simple})`
    based on `is_simple`

**`ready_to_proceed` field on ResearchBrief** (default: False):
- On a fresh brief, LLM sets it False (needs user review)
- On re-invocation with user feedback, LLM reads prior brief + feedback and:
  - Sets True if user approved → proceeds to research
  - Sets False if user requested changes → revises and exits for another round
- LLM detects approval vs feedback — no heuristic code needed
- Multiple review rounds supported naturally

**Return type change**: `write_research_brief` now returns
`Command[Literal["__end__", "researcher", "coordinator"]]` instead of `dict`.

### Step 10 — Move run_single_researcher to adapter.py ✅

Extract the researcher adapter from graph.py into its own module.

**nodes/researcher/adapter.py** (new file):
- Move `run_single_researcher` function from `graph/graph.py`
- Maps AgentState → ResearcherState, invokes researcher_subgraph, maps back
- Import `researcher_subgraph` from `nodes/researcher/__init__`

### Step 11 — Clean graph.py to pure orchestration ✅

Remove all node logic from graph.py. It should only register nodes and wire edges.

**graph/graph.py**:
- Remove `human_review` function and node
- Remove `route_by_complexity` function and conditional edge
- Remove `run_single_researcher` function (now in adapter.py)
- Remove imports no longer needed: `interrupt`, `configurable_model`,
  `Configuration`, `HumanMessage`
- Import `run_single_researcher` from `nodes.researcher.adapter`
- Node registration: clarify, write_brief, researcher, coordinator, final_report
- Edges: `START → clarify`, `researcher → final_report`, `coordinator → final_report`,
  `final_report → END`
- No edge from write_brief — it routes itself via Command
- No edge from clarify — it routes itself via Command
- Update module docstring to reflect new architecture
- Result: 52 lines of pure orchestration

### Step 12 — Tests ✅

**Unit tests** (35 total, mocked LLM, no API calls):
- Brief node: review disabled → routes to researcher or coordinator based on is_simple
- Brief node: review enabled, first draft → routes to __end__ with brief
- Brief node: revision approved (ready_to_proceed=True) → routes to coordinator
- Brief node: revision with feedback (ready_to_proceed=False) → exits to __end__ again
- Clarify node tests: disabled/needs clarification/no clarification
- Schema validation, coordinator formatting helpers (existing)

**Integration tests** (3 total, real APIs):
- Full pipeline with `AUTOMATED_CONFIG` (all HITL disabled) — question → report
- Clarification enabled with clear question — proceeds without asking
- Full HITL review cycle: generate brief → user feedback → revise → user approves → report
- All passing (unit: ~30s, integration: ~7.5min)

---

## Observations

### What worked well

- **Same pattern for both interaction nodes**: Clarify and write_brief use identical graph-exit + re-invocation. The code is consistent and easy to reason about. No `interrupt()` complexity, no checkpointer requirement for basic HITL.

- **LLM-driven routing via `ready_to_proceed`**: Letting the LLM decide "is this approval or feedback?" is cleaner than heuristic detection. The LLM reads the user's message in context and makes a judgment call — exactly what it's good at.

- **One prompt for fresh + revision**: No prompt duplication. The optional `{prior_brief}` and `{feedback}` sections are empty on first pass, populated on revision. Same structured output either way.

- **graph.py is tiny**: 52 lines, no logic. All behavior lives in node modules where it belongs.

### What to watch

- **Clarification on re-invocation**: When the graph is re-invoked for brief review, clarify runs again. It sees the full history and should skip. The prompt says "If the message history shows you have already asked a clarifying question, almost never ask another." This works in testing, but edge cases (long histories, ambiguous follow-up messages) haven't been stress-tested.

- **Approach field quality**: The approach is free-text from the LLM. Its usefulness depends on prompt quality. In testing, the LLM produces reasonable bullet-point strategies, but we haven't measured whether the coordinator actually decomposes differently with vs without approach guidance.

- **Review cycle termination**: The current design supports unlimited review rounds (feedback → revise → feedback → revise → ...). In practice the user controls this by approving. But a misbehaving LLM that never sets `ready_to_proceed=True` would loop forever. A max review iterations guard could be added if this becomes an issue.

- **Re-invocation requires full state**: The graph-exit pattern means the caller must pass the full state back (including `research_brief`) on re-invocation. This is fine for programmatic use but needs careful UX design for a chat interface — the client must track and replay state.

---

## Resolved Questions

1. **How does write_brief detect approval vs feedback?** Via `ready_to_proceed: bool` on ResearchBrief. The LLM reads the prior brief + user's latest message and sets it True (approval) or False (changes requested). No heuristic code needed — the LLM makes a contextual judgment.

2. **Should there be a separate revision prompt?** No. One prompt with optional `{prior_brief}` and `{feedback}` sections handles both fresh generation and revision. When empty, it generates fresh; when populated, it revises.

3. **Should `is_simple` be more nuanced?** Kept binary for now. Start simple, add nuance if the binary gate proves too coarse.

4. **Does the clarify node interfere with brief review?** On re-invocation for brief review, clarify sees the full message history and skips. Tested and working, though edge cases with long histories remain untested.

---

## File Changes Summary

### New files
- `src/deep_research/nodes/clarify.py` — clarification node
- `src/deep_research/nodes/researcher/adapter.py` — researcher state adapter

### Modified files
- `src/deep_research/models.py` — added ClarifyOutput, ResearchBrief (reformed with approach, is_simple, ready_to_proceed)
- `src/deep_research/prompts.py` — added clarify_prompt, rewrote research_brief_prompt (with revision support), simplified coordinator_system_prompt
- `src/deep_research/nodes/brief.py` — rewritten with Command routing + review cycle
- `src/deep_research/graph/graph.py` — simplified to pure orchestration (52 lines)
- `src/deep_research/configuration.py` — added allow_clarification, allow_human_review
- `src/deep_research/state.py` — added is_simple to AgentState
- `src/deep_research/__init__.py` — added ClarifyOutput, ResearchBrief to public exports
- `tests/test_nodes.py` — 35 tests covering all new behavior
- `tests/test_integration.py` — 3 integration tests including full HITL review cycle
