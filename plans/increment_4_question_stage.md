# Increment 4 — Question Stage (Clarification + Scoping)
**Goal**: Well-formed research questions through user interaction, with adaptive routing based on question needs.
**Status**: Planning

## Overview

The question stage sits at the front of the pipeline, before research begins. It has three responsibilities:

1. **Clarification**: Resolve ambiguity by asking the user questions (optional, config-gated)
2. **Brief generation**: Transform the (now-clear) user query into a single well-articulated research question — NOT a decomposed list of subtopics (that's the coordinator's job)
3. **Routing**: Simple questions go directly to a single researcher, bypassing the coordinator entirely

**Main graph (after)**:
```
clarify → write_brief → [simple?] → researcher_subgraph → final_report
                         [else]   → coordinator_subgraph → final_report
```

Where `clarify` may exit to `__end__` (returning a question to the user) or proceed to `write_brief`.

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

**No pre-defined question type enum or research parameters**: Instead of classifying questions into fixed types (factual, survey, comparison, etc.) and mapping to strategy tables, the coordinator reasons freely about what kind of research the question needs. Its prompt is updated to think deliberately about approach before decomposing — how many angles, breadth vs depth, what each researcher should cover. No external "strategy advisor" — the coordinator does its job.

**Simple vs non-simple is the only routing decision**: The only code-level orchestration is the binary gate: simple → single researcher, else → coordinator. Everything beyond that is the coordinator's judgment. This keeps the architecture clean — one routing decision, not a strategy pipeline.

---

## Schemas

### ClarifyOutput (new)

```python
class ClarifyOutput(BaseModel):
    need_clarification: bool      # whether to ask the user a question
    question: str                 # the clarifying question (empty if not needed)
    verification: str             # acknowledgement before research (empty if clarifying)
```

### ResearchBrief (simplified)

```python
class ResearchBrief(BaseModel):
    title: str                    # concise title
    research_question: str        # single well-articulated research question (paragraph)
    is_simple: bool               # whether a single researcher can handle this
```

Drops `research_questions: list[str]` and `key_topics: list[str]`. The coordinator handles decomposition.

`is_simple` drives routing — the LLM decides as part of brief generation whether this question needs multi-topic decomposition or can be handled by one researcher. Criteria: single factual lookup, narrow scope, one clear answer expected.

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
            → LLM call: generate ResearchBrief (title + research_question + is_simple)
            → state: {research_brief: str, is_simple: bool}
       → conditional edge
            ├─ [is_simple=true] → researcher adapter → researcher_subgraph → final_report
            └─ [is_simple=false] → coordinator_subgraph → final_report
```

### Simple question path

For `is_simple=true`, a thin adapter node maps `AgentState` → `ResearcherState`:
- `research_brief` → `research_topic` (the full brief becomes the topic)
- Initialize empty accumulator fields
- On return: `ResearcherState.notes` → `AgentState.notes`

The adapter is a simple function, not a subgraph. The researcher subgraph is invoked programmatically (like `dispatch_research` does) and results are written back to `AgentState`.

---

## Coordinator Prompt Update

The coordinator prompt gets a strategic reasoning instruction. Instead of pre-defined types or parameters, we ask the coordinator to think deliberately about approach:

```
<task>
Read the research brief below and determine the right research approach.

Before dispatching researchers, think about what kind of question this is
and what research strategy fits. Consider how many distinct angles need
independent investigation, whether the question needs breadth (many topics,
surface-level) or depth (fewer topics, thorough), and what each researcher
should focus on.

Then decompose into focused subtopics and dispatch researchers accordingly.
</task>
```

This replaces the current "decompose it into focused subtopics" instruction with a deliberate reasoning step. The coordinator naturally adapts — a comparison question gets 2-3 researchers per subject, a survey gets 4-5 covering different facets, an analysis gets fewer but deeper researchers.

---

## Steps

### Step 0 — Simplify ResearchBrief schema + update brief prompt

Reform the brief to produce a single research question instead of decomposed subtopics.

**models.py**:
- Change `ResearchBrief`: drop `research_questions: list[str]` and `key_topics: list[str]`, add `research_question: str`
- Keep `title: str` (used in report generation and logging)

**prompts.py**:
- Rewrite `research_brief_prompt`: instruct to produce a detailed, specific research question in first person. Key guidelines:
  - Maximize specificity — include all known user preferences and constraints
  - Fill unstated but necessary dimensions as open-ended (don't invent, state as flexible)
  - Avoid unwarranted assumptions
  - If specific sources should be prioritized, include them
  - If query is in a specific language, note to prioritize sources in that language
  - This is the sole input the coordinator sees — be thorough

**nodes/brief.py**:
- Update `write_research_brief` to format the new schema
- `brief_str` becomes: `f"Title: {brief.title}\n\n{brief.research_question}"`

**Tests**:
- Update any tests that depend on `research_questions` or `key_topics` fields

### Step 1 — Add is_simple routing + researcher adapter

Add the simple question bypass path.

**models.py**:
- Add `is_simple: bool` to `ResearchBrief`

**prompts.py**:
- Add `is_simple` criteria to the brief prompt: "Determine if this is a simple question that a single researcher can handle — a narrow, factual query with one clear answer expected. If so, set is_simple to true."

**state.py**:
- Add `is_simple: bool` to `AgentState` (default False)

**nodes/brief.py**:
- Return `is_simple` from the brief alongside `research_brief`

**graph/graph.py**:
- Add routing function after `write_brief`:
  ```python
  def route_by_complexity(state: AgentState) -> str:
      return "researcher" if state.get("is_simple", False) else "coordinator"
  ```
- Add `researcher` node — thin adapter that invokes `researcher_subgraph` directly:
  ```python
  async def run_single_researcher(state: AgentState, config: RunnableConfig) -> dict:
      initial_state = {
          "messages": [],
          "research_topic": state["research_brief"],
          "research_iterations": 0,
          "last_reflection": "",
          "accumulated_findings": [],
          "accumulated_contradictions": [],
          "current_gaps": [],
          "final_knowledge_state": "",
          "notes": "",
      }
      result = await researcher_subgraph.ainvoke(initial_state, config)
      return {"notes": result["notes"]}
  ```
- Conditional edge: `write_brief` → `route_by_complexity` → (`researcher` | `coordinator`)
- Both paths converge on `final_report`

### Step 2 — Update coordinator prompt for strategic reasoning

Update the coordinator to think deliberately about research approach.

**prompts.py**:
- Revise `coordinator_system_prompt` task section: before decomposing, reason about what kind of question this is and what approach fits. Think about how many angles, breadth vs depth, what each researcher should cover.
- Remove the static `{max_research_topics}` from the instruction text (keep the config as a hard cap but don't advertise a number the coordinator should target). Instead: "Dispatch as many researchers as the question genuinely needs — a focused comparison may need 2-3, a broad survey may need more."

**nodes/coordinator/coordinator.py**:
- Update prompt formatting if template variables changed

### Step 3 — Add ClarifyOutput schema + clarify prompt

**models.py**:
- Add `ClarifyOutput(need_clarification, question, verification)`

**prompts.py**:
- Add `clarify_prompt`: assess whether clarification is needed based on message history. Guidelines:
  - If acronyms, abbreviations, or unknown terms exist, ask
  - If scope is genuinely ambiguous (could mean very different research directions), ask
  - If message history shows a prior clarifying exchange, almost never ask again
  - Don't ask for unnecessary information or information already provided
  - When not clarifying, provide a verification message summarizing understanding
  - Be concise — one well-structured question, not an interrogation

**configuration.py**:
- Add `allow_clarification: bool = Field(default=True)`

### Step 4 — Implement clarify node + wire into graph

**nodes/clarify.py**:
- `clarify_with_user(state, config)` node
- If `allow_clarification` is false: `Command(goto="write_brief")`
- Otherwise: call LLM with structured output `ClarifyOutput`
- If `need_clarification`: `Command(goto="__end__", update={"messages": [AIMessage(question)]})`
- If not: `Command(goto="write_brief", update={"messages": [AIMessage(verification)]})`

**graph/graph.py**:
- Add `clarify` node before `write_brief`
- `START → clarify` (clarify uses Command for routing to `write_brief` or `__end__`)
- Rest of graph unchanged: `write_brief → route → (researcher | coordinator) → final_report`

**Notes on re-invocation**: When the graph exits for clarification, the caller appends the user's answer to messages and re-invokes. The graph starts from `clarify` again, which now sees the full message history including the answer. The clarify node re-evaluates and either asks another question (rare) or proceeds.

### Step 5 — Tests

**Unit tests** (no API calls):
- `ClarifyOutput` schema validation
- `ResearchBrief` new schema (single `research_question` + `is_simple` fields)
- Clarify node routing: clarification needed → END, not needed → write_brief
- Clarify node skip: `allow_clarification=false` → straight to write_brief
- Route function: `is_simple=true` → researcher, `is_simple=false` → coordinator
- Researcher adapter: maps AgentState → ResearcherState correctly

**Integration test**:
- Clear, simple question: verify flows clarify → write_brief → researcher → report (no coordinator)
- Clear, complex question: verify flows clarify → write_brief → coordinator → report
- Verify brief is a single research question (not decomposed list)
- Verify coordinator decomposes effectively from the simplified brief
- Verify coordinator reasons about strategy (check traces for deliberate approach reasoning)

---

## Open Questions

1. **Clarification in programmatic use**: When the system is invoked via API (not chat UI), clarification requires the caller to handle re-invocation. The `allow_clarification=false` config covers this, but we should document the pattern for callers who want clarification.

2. **Should `is_simple` be more nuanced?** Currently binary. We could add a middle ground — "focused" questions that need the coordinator but with constrained decomposition. Start simple, add nuance if the binary gate proves too coarse.

3. **Coordinator strategy quality**: After updating the coordinator prompt to reason about strategy, we need to verify via traces that it actually adapts (e.g., dispatches 2 researchers for a comparison, 5 for a broad survey). If it still defaults to ~5 every time, the prompt needs further calibration.
