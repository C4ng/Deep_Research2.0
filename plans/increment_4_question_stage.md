# Increment 4 — Question Stage (Clarification + Scoping)
**Goal**: Well-formed research questions through user interaction, with adaptive routing based on question needs.
**Status**: In Progress (steps 0-4 done)

## Overview

The question stage sits at the front of the pipeline, before research begins. It has four responsibilities:

1. **Clarification**: Resolve ambiguity by asking the user questions (optional, config-gated)
2. **Brief generation**: Transform the (now-clear) user query into a research question with strategic approach guidance — NOT decomposed into exact topics (that's the coordinator's job)
3. **Human review**: Let the user review and modify the research plan before resources are spent
4. **Routing**: Simple questions go directly to a single researcher, bypassing the coordinator entirely

**Main graph (after)**:
```
START → clarify → write_brief → researcher  → final_report → END
                              → coordinator → final_report → END
```

Both `clarify` and `write_brief` are user interaction nodes using the same graph-exit pattern. Each reads messages, calls the LLM, and routes via `Command` — either exiting to `__end__` (returning output to the user for feedback) or proceeding to the next node.

---

## Motivation

### Problem 1: Brief does topic decomposition (coordinator's job)

Current `ResearchBrief` has `research_questions: list[str]` and `key_topics: list[str]`. The brief decomposes the query into subtopics, then the coordinator decomposes *again* from those subtopics into researcher assignments. This is redundant — the brief is prematurely structuring the problem.

The brief should be a single, well-articulated question that preserves the user's intent, constraints, and explicitly marks what's unspecified. The coordinator is the one who decides how to decompose.

### Problem 2: Ambiguous queries go straight to research

"quantum computing" could mean anything: recent news? Investment opportunities? Technical deep-dive? For a PhD thesis or a blog post? The system currently guesses. A clarification step catches this before wasting research resources.

### Problem 3: All queries get the same treatment

A simple factual question ("What is the latest React version?") gets 5 researchers and multi-round reflection — massive overkill. The system should route simple questions directly to a single researcher, skipping coordinator overhead entirely.

### Problem 4: Coordinator doesn't reason about strategy

The coordinator currently always decomposes into ~5 topics regardless of the question. It should think deliberately about what kind of question it's facing and what research approach fits — how many angles, breadth vs depth, what each researcher should focus on. This is the coordinator's job, not a separate strategy node's.

---

## Key Design Decisions

**Clarification is optional and conservative**: Gated by `allow_clarification` config (default: true). When enabled, the model is instructed to ask only when genuinely needed — acronyms, unclear scope, critical ambiguity. One clarifying question per invocation, almost never two. The user should feel helped, not interrogated.

**Clarification uses graph exit pattern**: When clarification is needed, the node returns `Command(goto=END)` with the question as an AIMessage. The caller re-invokes the graph with the user's answer appended to messages. This works naturally with chat UIs and LangGraph's checkpointing.

**Brief becomes a single research question**: Replace `ResearchBrief(title, research_questions, key_topics)` with a simpler schema. The brief is a detailed, well-articulated paragraph in first person, not a decomposed structure. Key guidelines: maximize specificity, fill unstated dimensions as open-ended, avoid unwarranted assumptions.

**Brief includes strategic guidance**: The research brief is not just "what to research" but also "how to approach it" — angles to cover, breadth vs depth preference, priorities. This is a strategy document the user can review and modify. The coordinator reads this guidance and decides exact topic decomposition. No separate strategy node needed.

**Brief and clarify use the same interaction pattern**: Both are user ↔ AI communication nodes using the graph-exit pattern (`Command(goto="__end__")`). Clarify exits with a question; write_brief exits with the draft plan. On re-invocation, each reads the full message history (including prior output + user response) and decides whether to proceed or iterate. One unified prompt handles both fresh generation and revision — if prior brief + feedback exist in messages, the LLM revises; otherwise it generates from scratch.

**write_brief owns routing**: The brief node determines `is_simple` and routes directly via `Command(goto="researcher")` or `Command(goto="coordinator")`. No separate conditional edge or routing function needed — the node makes the decision and routes, same as clarify decides and routes.

**Coordinator executes strategy, doesn't create it**: The coordinator reads the brief's strategic guidance and decomposes into exact topics accordingly. It no longer reasons about question type or approach from scratch — that's already in the brief. For follow-up rounds (gap-filling, contradiction resolution), the coordinator operates autonomously.

**graph.py is pure orchestration**: Node registration, edges, compilation. No LLM calls, no prompts, no state mapping. All node logic lives in node modules.

---

## Schemas

### ClarifyOutput (new)

```python
class ClarifyOutput(BaseModel):
    need_clarification: bool      # whether to ask the user a question
    question: str                 # the clarifying question (empty if not needed)
    verification: str             # acknowledgement before research (empty if clarifying)
```

### ResearchBrief (reformed)

```python
class ResearchBrief(BaseModel):
    title: str                    # concise title
    research_question: str        # detailed research question with constraints
    approach: str                 # strategic guidance: angles, breadth/depth, priorities
    is_simple: bool               # whether a single researcher can handle this
```

Drops `research_questions: list[str]` and `key_topics: list[str]`. The coordinator handles exact decomposition.

`research_question` captures what to research with all user constraints preserved.

`approach` captures how to approach it — what kind of question this is, what angles matter, breadth vs depth, priorities. This is the strategy the coordinator follows. Example: "This is a broad survey. Cover distinct facets: technology advancements, key players, real-world applications, challenges. Breadth over depth. Market/investment data would be valuable context."

`is_simple` drives routing — true for narrow, factual queries a single researcher can handle.

---

## State Changes

### AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_brief: str
    is_simple: bool              # NEW — routing flag from brief generation
    notes: str
    final_report: str
```

`is_simple` drives the conditional edge after `write_brief`.

---

## Information Flow

```
User sends question
  → clarify node (reads messages)
  ├─ [need_clarification=true] → Command(goto="__end__") with question
  │     → user answers → graph re-invoked → clarify sees full history → proceeds
  └─ [need_clarification=false] → Command(goto="write_brief") with verification
       → write_brief node (reads messages)
            → generates ResearchBrief (structured output)
            ├─ [allow_human_review=true] → Command(goto="__end__") with brief as AIMessage
            │     → user reviews, gives feedback or approves
            │     → graph re-invoked → clarify skips → write_brief reads history
            │     → sees prior brief + feedback → revises or proceeds
            └─ [approved or review disabled]
                 ├─ [is_simple=true]  → Command(goto="researcher")
                 └─ [is_simple=false] → Command(goto="coordinator")
                      → final_report → END
```

### User interaction cycle (same pattern for both nodes)

Both clarify and write_brief follow the same cycle:
1. Read full message history
2. Call LLM (structured output)
3. Decide: need user input? → exit to `__end__` with AIMessage
4. User responds → graph re-invoked → node reads updated history
5. Ready to proceed? → route to next node via `Command`

The prompt handles both fresh and revision cases — if prior output + user
feedback exist in messages, the LLM revises; otherwise it generates fresh.

### Brief as strategy document

The `research_brief` string includes both the question and approach:
```
Title: Quantum Computing in 2025

What is the current state of quantum computing in 2025...

Approach:
- Broad survey: cover distinct facets rather than deep-diving one
- Important angles: technology, key players, applications, challenges
- Market/investment data would be valuable context
```

The user sees this as an AIMessage and can respond with feedback ("focus
more on hardware") or approval. The coordinator reads the final brief
and decides exact topic decomposition.

### Simple question path

For `is_simple=true`, the researcher adapter maps `AgentState` → `ResearcherState`:
- `research_brief` → `research_topic`
- Initialize empty accumulator fields
- On return: `ResearcherState.notes` → `AgentState.notes`

The adapter lives in `nodes/researcher/adapter.py`.

---

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

### Step 8 — Extend research_brief_prompt to handle revision

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

### Step 9 — Rewrite write_research_brief with Command routing + review cycle

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
  - If `allow_human_review` and this is the first draft (no prior brief in state)
    → `Command(goto="__end__", update={research_brief, is_simple, messages: [AIMessage with brief]})`
    The user sees the brief and can respond with feedback or approval
  - If review disabled, or user approved, or this is a revision pass
    → `Command(goto="researcher" or "coordinator", update={research_brief, is_simple})`
    based on `is_simple`

**How approval is detected**: If `research_brief` already exists in state
(meaning we showed it before), the node is being re-invoked. The LLM reads
the prior brief + user's latest message and produces a (possibly unchanged)
ResearchBrief. The node always proceeds after a revision pass — one round
of feedback, then move on. The user can re-invoke again if they want more
changes (the graph-exit pattern supports this naturally).

**Return type change**: `write_research_brief` now returns
`Command[Literal["__end__", "researcher", "coordinator"]]` instead of `dict`.

### Step 10 — Move run_single_researcher to nodes/researcher/adapter.py

Extract the researcher adapter from graph.py into its own module.

**nodes/researcher/adapter.py** (new file):
- Move `run_single_researcher` function from `graph/graph.py`
- Maps AgentState → ResearcherState, invokes researcher_subgraph, maps back
- Import `researcher_subgraph` from `nodes/researcher/__init__`

**nodes/researcher/__init__.py**:
- Add `run_single_researcher` to `__all__` exports

### Step 11 — Clean graph.py to pure orchestration

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

### Step 12 — Tests

Update tests to match the restructured code.

**Unit tests** (mocked LLM, no API calls):
- Brief node: review disabled → routes to researcher or coordinator based on is_simple
- Brief node: review enabled, first draft → routes to __end__ with brief
- Brief node: revision pass (prior brief in state + feedback) → routes to next node
- Clarify node tests: keep existing (disabled/needs clarification/no clarification)
- Schema validation: keep existing
- Coordinator formatting helpers: keep existing

**Integration test** (real APIs):
- Full pipeline with `AUTOMATED_CONFIG` (HITL disabled)
- Verify brief includes approach, report is substantial

---

## Open Questions

1. **Clarification in programmatic use**: When the system is invoked via API (not chat UI), clarification requires the caller to handle re-invocation. The `allow_clarification=false` config covers this, but we should document the pattern for callers who want clarification.

2. **Should `is_simple` be more nuanced?** Currently binary. We could add a middle ground — "focused" questions that need the coordinator but with constrained decomposition. Start simple, add nuance if the binary gate proves too coarse.

3. **How does write_brief detect approval vs feedback?** When the user re-invokes after seeing the brief, the node reads messages. It needs to distinguish "user approved" from "user gave feedback." Options: check if the last user message is a short approval phrase, or always treat any re-invocation after brief as a revision pass (the LLM decides if the feedback changes anything).
