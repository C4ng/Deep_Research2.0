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
clarify → write_brief → [human review] → [simple?] → researcher_subgraph → final_report
                                          [else]   → coordinator_subgraph → final_report
```

Where `clarify` may exit to `__end__` (returning a question to the user) or proceed to `write_brief`. Human review is an interrupt point where the user can modify the brief.

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

**Simple questions bypass the coordinator**: The `write_brief` node assesses whether the question is simple enough for a single researcher. If yes, it sets `is_simple=True` in state, and a conditional edge routes directly to the researcher subgraph. The researcher subgraph already exists — the coordinator invokes it via `dispatch_research`. We just invoke it directly from the main graph with a thin state adapter.

**Brief includes strategic guidance**: The research brief is not just "what to research" but also "how to approach it" — angles to cover, breadth vs depth preference, priorities. This is a strategy document the user can review and modify. The coordinator reads this guidance and decides exact topic decomposition. No separate strategy node needed.

**Human review after brief generation**: The brief is the highest-leverage point for user intervention — it shapes all downstream research before resources are spent. An interrupt after `write_brief` lets the user see the research plan (question + strategy), modify angles, add perspectives, or adjust priorities. This happens in the main graph (simple interrupt), not inside the coordinator subgraph.

**Coordinator executes strategy, doesn't create it**: The coordinator reads the brief's strategic guidance and decomposes into exact topics accordingly. It no longer reasons about question type or approach from scratch — that's already in the brief. For follow-up rounds (gap-filling, contradiction resolution), the coordinator operates autonomously.

**Simple vs non-simple is the only routing decision**: The only code-level orchestration is the binary gate: simple → single researcher, else → coordinator. Everything beyond that is the coordinator's judgment.

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
User message(s)
  → clarify node
  ├─ [need_clarification=true] → END (return question to user)
  │     → user provides answer → graph re-invoked → clarify sees full history
  └─ [need_clarification=false] → verification message + proceed
       → write_brief node
            → LLM call: ResearchBrief (title + research_question + approach + is_simple)
            → state: {research_brief: str, is_simple: bool}
            → [human review interrupt — user can modify the brief]
       → conditional edge
            ├─ [is_simple=true] → researcher adapter → researcher_subgraph → final_report
            └─ [is_simple=false] → coordinator_subgraph → final_report
```

### Brief as strategy document

The `research_brief` string stored in state includes both the question and the approach:
```
Title: Quantum Computing in 2025

I want a comprehensive overview of the current state of quantum computing
in 2025, covering technology, key players, applications, and challenges.
No constraints on geography or specific companies. Prioritize recent
developments and concrete data over speculation.

Approach: This is a broad survey. Cover distinct facets rather than going
deep on any single one. Important angles: technology advancements, who's
driving it, where it's being used, what's holding it back.
Market/investment data would be valuable context.
```

The user sees this at the interrupt point and can modify it — add angles,
remove irrelevant ones, adjust priorities. The coordinator reads the
(possibly modified) brief and decides exact topic decomposition.

### Simple question path

For `is_simple=true`, a thin adapter node maps `AgentState` → `ResearcherState`:
- `research_brief` → `research_topic` (the full brief becomes the topic)
- Initialize empty accumulator fields
- On return: `ResearcherState.notes` → `AgentState.notes`

The adapter is a simple function, not a subgraph. The researcher subgraph
is invoked programmatically (like `dispatch_research` does) and results are
written back to `AgentState`.

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

### Step 5 — Add approach field to ResearchBrief

Enrich the brief with strategic guidance so the user can review and modify the research plan.

**models.py**:
- Add `approach: str` to `ResearchBrief` — strategic guidance describing what kind of question this is, what angles matter, breadth vs depth, priorities

**prompts.py**:
- Update `research_brief_prompt`: add instruction to generate an approach section. Guidelines:
  - Think about what kind of research this question needs
  - Identify the important angles/facets to cover
  - State whether breadth or depth is more important
  - Note any priorities or special considerations
  - This is guidance for the coordinator — not an exact topic list

**nodes/brief.py**:
- Include `approach` in the formatted `brief_str`:
  `f"Title: {brief.title}\n\n{brief.research_question}\n\nApproach: {brief.approach}"`

### Step 6 — Remove strategy reasoning from coordinator prompt

Now that strategic guidance lives in the brief, the coordinator doesn't need to reason about strategy from scratch. It reads the brief (which includes the approach) and decomposes accordingly.

**prompts.py**:
- Remove the few-shot strategy examples from `coordinator_system_prompt` instruction 1
- Replace with: "Read the research brief, including its approach guidance, and decompose into focused subtopics accordingly."
- Keep instructions 2-5 (dispatch mechanics, complementary topics, prior research, no summary)

### Step 7 — Add human review interrupt after write_brief

Add an interrupt point so the user can review and modify the research brief before research begins.

**graph/graph.py**:
- Add interrupt after `write_brief` node using LangGraph's `interrupt()` mechanism
- The interrupt surfaces the `research_brief` string to the user
- The user can: approve as-is, or modify the brief (edit text directly)
- On resume, the (possibly modified) brief flows to routing → coordinator/researcher
- Make this config-gated (`allow_human_review: bool`, default: true for interactive, false for programmatic)

**configuration.py**:
- Add `allow_human_review: bool = Field(default=True)`

**Notes on interrupt pattern**: LangGraph's `interrupt()` pauses the graph and returns the interrupt value to the caller. The caller presents it to the user, collects their input, and resumes with `Command(resume=...)`. This is different from the clarify pattern (graph exit + re-invocation) — interrupt preserves graph position.

### Step 8 — Tests

**Unit tests** (no API calls):
- `ClarifyOutput` schema validation
- `ResearchBrief` new schema (research_question + approach + is_simple)
- Clarify node routing: clarification needed → END, not needed → write_brief
- Clarify node skip: `allow_clarification=false` → straight to write_brief
- Route function: `is_simple=true` → researcher, `is_simple=false` → coordinator
- Brief formatting includes approach section

**Integration test**:
- Clear, complex question: verify flows clarify → write_brief → coordinator → report
- Verify brief includes approach section with strategic guidance
- Verify coordinator decomposes based on the brief's approach (check traces)
- Simple question: verify bypasses coordinator

---

## Open Questions

1. **Clarification in programmatic use**: When the system is invoked via API (not chat UI), clarification requires the caller to handle re-invocation. The `allow_clarification=false` config covers this, but we should document the pattern for callers who want clarification.

2. **Should `is_simple` be more nuanced?** Currently binary. We could add a middle ground — "focused" questions that need the coordinator but with constrained decomposition. Start simple, add nuance if the binary gate proves too coarse.

3. **Human review UX**: The interrupt surfaces the brief as text. How should the user modify it? Direct text editing is simplest. A structured form (edit approach separately from question) is more guided but more complex. Start with text.

4. **Interrupt vs graph exit for human review**: Using `interrupt()` (preserves graph position) is cleaner than graph exit (requires re-invocation). But it requires the caller to support LangGraph's interrupt/resume protocol. Clarification uses graph exit because the user's response is a new message; human review uses interrupt because the user is editing existing state, not adding a message.
