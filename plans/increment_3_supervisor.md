# Increment 3 — Supervisor + Multi-Topic Decomposition
**Goal**: Complex questions decomposed into subtopics, each researched independently, with cross-topic synthesis and follow-up.
**Status**: Planning

## Overview

The supervisor sits between `write_brief` and `final_report` in the main graph. It receives the research brief, decomposes it into subtopics, dispatches researchers (one per subtopic), collects results, reflects on cross-topic completeness/contradictions, and either assigns follow-up research or exits.

**Main graph (after)**:
`write_brief` → `supervisor_subgraph` → `final_report`

**Supervisor subgraph**:
`supervisor` → `supervisor_tools` → `supervisor_reflect` → (`supervisor` | END)

**Researcher subgraph** (state isolated, accumulates knowledge across rounds):
`researcher` → `researcher_tools` → `reflect` → (`researcher` | `summarizer` → END)

Each researcher is invoked as a subgraph call via a supervisor tool. The researcher gets only its assigned topic — not the full brief or other researchers' findings (context isolation).

---

## Key Design Decisions

**Structured reflection at both levels**: `ResearchReflection` (researcher) and `SupervisorReflection` (supervisor) — same pattern, different schemas. Both enable programmatic tracking. Required for dead-end detection in Increment 5.

**Knowledge accumulation, not message accumulation**: Each researcher round produces structured knowledge (findings, contradictions, gaps). This knowledge accumulates in state fields with append reducers — not thrown away after routing. Messages are transient (trimmed for context engineering); structured knowledge persists across rounds and flows to downstream stages.

**Information flow principle**: When entering the next stage, keep structured knowledge from previous stages that impacts current decisions. Specifically:
- Reflection captures key findings + strategic observations from AI reasoning → accumulates in state
- Summarizer sees accumulated findings to prioritize compression
- Supervisor receives `ResearchResult` with full accumulated metadata, not just compressed notes
- Report node sees combined notes with topic structure

**Researchers return structured metadata**: `ResearchResult` with accumulated `key_findings`, `knowledge_state`, `missing_info`, `contradictions` alongside compressed notes. Built from accumulated state, not a separate LLM call.

**Supervisor uses tools for dispatching**: `conduct_research(topic, context)` tool invokes a researcher subgraph. Makes dispatching explicit and traceable in LangSmith.

**Sequential execution first**: Researchers run one at a time. Parallel execution deferred to Increment 5.

**File restructure**: Node files grouped by subgraph as packages (`nodes/researcher/`, `nodes/supervisor/`). Standalone nodes (`brief.py`, `report.py`) stay at `nodes/` level.

---

## Information Flow

### What exists per researcher round

| Source | Contains | Kept as | Who uses it downstream |
|--------|----------|---------|----------------------|
| AIMessage | Model's reasoning — why it chose queries, connections it noticed | Reflection captures strategic observations into `key_findings` | Subsequent rounds (via accumulated_findings), Supervisor (via ResearchResult) |
| ToolMessages | Raw search results (Tavily-summarized) | Raw in `messages` (trimmed on subsequent rounds), compressed by summarizer | Reflect (current round), Summarizer (all rounds) |
| ResearchReflection | Structured assessment — findings, gaps, contradictions | Accumulated in state fields (append reducers) | Next researcher round, Summarizer (prioritization), Supervisor (ResearchResult) |
| Summarizer output | Compressed notes | `notes` string | Supervisor, Report node |

### Accumulation vs overwrite

| Field | Behavior | Why |
|-------|----------|-----|
| `accumulated_findings` | **Append** each round | Findings grow — round 2 adds to round 1's discoveries |
| `accumulated_contradictions` | **Append** each round | New contradictions discovered across rounds |
| `current_gaps` | **Overwrite** each round | Gaps are the *current* state — old gaps either filled or reformulated |
| `last_reflection` | **Overwrite** each round | Only the latest reflection guidance matters for the next round |
| `messages` | **Append** (via add_messages) but **trimmed** for LLM calls | Context engineering — full history in state, trimmed view for LLM |

### Cross-stage flow

```
Researcher rounds (accumulate structured knowledge)
  → accumulated_findings, accumulated_contradictions, current_gaps
  → Summarizer (reads accumulated_findings for prioritization + ToolMessages for content)
  → ResearchResult (built from accumulated state + compressed notes)
  → Supervisor (reads ResearchResult metadata for cross-topic decisions)
  → SupervisorReflection (cross-topic gaps, contradictions, follow-ups)
  → Report node (combined notes with topic headers)
```

---

## Schemas

### ResearchReflection (rename from Reflection)

Same schema, renamed for clarity:

```python
class ResearchReflection(BaseModel):
    key_findings: list[str]        # facts + strategic observations from this round
    missing_info: list[str]
    contradictions: list[str]
    knowledge_state: Literal["insufficient", "partial", "sufficient"]
    should_continue: bool
    next_queries: list[str]
```

Reflection prompt updated: "Capture notable strategic observations (connections between sources, unexpected scope, quality signals) as key_findings, not just factual discoveries."

### SupervisorReflection + FollowUp (new)

```python
class FollowUp(BaseModel):
    topic: str             # what to research
    reason: str            # why — gap, deepen, contradiction
    context: str           # relevant findings so far, what to resolve

class SupervisorReflection(BaseModel):
    key_findings: list[str]             # synthesis across all researchers
    missing_info: list[str]             # gaps across the whole question
    contradictions: list[str]           # conflicts between researchers
    knowledge_state: Literal["insufficient", "partial", "sufficient"]
    should_continue: bool
    follow_ups: list[FollowUp]          # topics to assign next round
```

Field criteria live in the supervisor reflection prompt, not schema descriptions.

### ResearchResult (new)

What a researcher returns to the supervisor — built from accumulated state, not a separate LLM call:

```python
class ResearchResult(BaseModel):
    topic: str                          # what was researched
    notes: str                          # compressed findings (from summarizer)
    key_findings: list[str]             # accumulated across all reflection rounds
    knowledge_state: Literal["insufficient", "partial", "sufficient"]
    missing_info: list[str]             # final gaps (from last reflection)
    contradictions: list[str]           # accumulated across all reflection rounds
```

---

## State Design

### ResearcherState (new)

```python
class ResearcherState(TypedDict):
    messages: Annotated[list, add_messages]
    research_topic: str                  # assigned subtopic (not full brief)
    research_iterations: int
    last_reflection: str                 # formatted guidance for next round (overwrite)

    # Accumulated across reflection rounds (append reducers)
    accumulated_findings: Annotated[list[str], operator.add]
    accumulated_contradictions: Annotated[list[str], operator.add]
    current_gaps: list[str]              # overwrite — current state of gaps

    notes: str                           # summarizer output
```

Key differences from current `AgentState` usage:
- `research_brief` → `research_topic` (focused subtopic, not full brief)
- New accumulator fields with append reducers
- `current_gaps` is last-write-wins (always the latest assessment)
- No `final_report` — researcher doesn't produce reports

### SupervisorState (new)

```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]        # for tool calling mechanism (cleared each round)
    research_brief: str                             # from parent AgentState
    notes: str                                      # combined output (for parent AgentState mapping)
    research_results: Annotated[list[ResearchResult], operator.add]
    supervisor_iterations: int
    last_supervisor_reflection: str
```

`messages` is needed for the LangGraph tool calling pattern (AIMessage with tool_calls → ToolMessages). Cleared at the start of each supervisor round to prevent accumulation.

`notes` field overlaps with `AgentState.notes` by name — LangGraph auto-maps it when the subgraph is added as a node. Supervisor reflection writes combined notes here on exit.

### AgentState changes

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_brief: str
    notes: str                           # combined notes from supervisor
    final_report: str
    # removed: research_iterations, last_reflection (moved to ResearcherState)
```

---

## Supervisor Flow (per round)

**Round 1:**
1. Supervisor sees `research_brief` + empty `research_results`
2. LLM decomposes into subtopics, calls `conduct_research(topic=..., context=...)` for each
3. `supervisor_tools` executes each tool call → runs researcher subgraph → collects `ResearchResult`
4. `supervisor_reflect` assesses all results: cross-topic completeness, gaps, contradictions
5. If `should_continue` + `follow_ups` not empty → route back to `supervisor` with reflection
6. If done → merge all `research_results[].notes` with topic headers into `notes`, route to END

**Round 2+ (follow-up):**
1. Supervisor sees `research_brief` + all prior `research_results` + `last_supervisor_reflection`
2. LLM reads reflection guidance (which follow-ups, why), calls `conduct_research` for each
3. Same cycle: tools → reflect → continue or exit

---

## Steps

### Step 0 — Restructure node files into subgraph packages

Move existing node files into packages:

```
nodes/
    researcher/
        __init__.py         # exports: researcher_subgraph (wiring moved here from researcher.py)
        researcher.py       # researcher + researcher_tools node functions only
        reflect.py          # research reflection node (from nodes/reflect.py)
        summarizer.py       # from nodes/compress.py, rename function compress_research → summarize_research
    brief.py                # unchanged
    report.py               # unchanged
```

Changes:
- Move `nodes/researcher.py` → `nodes/researcher/researcher.py` (node functions only, subgraph wiring to `__init__.py`)
- Move `nodes/reflect.py` → `nodes/researcher/reflect.py`
- Move `nodes/compress.py` → `nodes/researcher/summarizer.py`, rename `compress_research` → `summarize_research`
- `nodes/researcher/__init__.py` builds and exports `researcher_subgraph`
- Update all imports (`graph/graph.py`, `__init__.py`, tests)
- Rename `Reflection` → `ResearchReflection` in `models.py`, update all references

No behavior changes. All existing tests must pass.

### Step 1 — Researcher state isolation + knowledge accumulation

This step introduces `ResearcherState` and changes the researcher's reflection to accumulate structured knowledge instead of discarding it.

**state.py**:
- Add `ResearcherState` with accumulator fields (`accumulated_findings`, `accumulated_contradictions`, `current_gaps`)
- Remove `research_iterations` and `last_reflection` from `AgentState` (moved to `ResearcherState`)

**nodes/researcher/reflect.py**:
- Reflect node now *appends* to `accumulated_findings` and `accumulated_contradictions` instead of only formatting for `last_reflection`
- When routing to summarizer: keep the final `ResearchReflection` data in accumulators (don't discard)
- `_format_reflection()` now includes accumulated context: "Already covered across all rounds: {accumulated_findings}"
- Reflection prompt update: instruct to capture strategic observations (connections between sources, scope signals) as `key_findings`, not just bare facts

**nodes/researcher/researcher.py**:
- Change state type from `AgentState` to `ResearcherState`
- `research_brief` → `research_topic` in prompt formatting
- Topic goes in SystemMessage only — no fake HumanMessage. Researcher is invoked programmatically (no real user input), so messages for LLM call = `[SystemMessage]` only. SystemMessage carries: role, instructions, assigned topic, tool limits, and prior reflection (if round 2+)
- On subsequent rounds, researcher sees `accumulated_findings` via `last_reflection` (already covered items)

**nodes/researcher/summarizer.py**:
- Optionally pass `accumulated_findings` to the compression prompt so the summarizer knows what the reflection identified as important — helps prioritize during compression

**nodes/researcher/__init__.py**:
- Build subgraph with `ResearcherState` instead of `AgentState`

**Tests**:
- Update existing unit tests for new state shape
- Test accumulator behavior: findings from round 1 persist into round 2
- Test that `current_gaps` overwrites (not appends)

### Step 2 — Supervisor schemas + state

**models.py**:
- Add `FollowUp`, `SupervisorReflection`, `ResearchResult`
- `ResearchResult` constructed from researcher's accumulated state fields

**state.py**:
- Add `SupervisorState` with `research_results: Annotated[list[ResearchResult], operator.add]`
- Update `AgentState`: remove `research_iterations`, `last_reflection` (done in Step 1), keep `notes`

**__init__.py**:
- Add new schemas to exports

### Step 3 — Supervisor prompts

Add to `prompts.py`:

**`supervisor_system_prompt`**: 
- Role: research coordinator that decomposes complex questions into focused subtopics
- Task: read the research brief, identify subtopics, dispatch researchers via `conduct_research` tool
- Instructions: how to read prior `research_results` and `last_supervisor_reflection`, how to formulate focused topics with context
- Limits: `{max_research_topics}` cap on initial decomposition

**`supervisor_reflection_prompt`**:
- Input: `{research_brief}`, `{research_results}` (formatted summary of all results with metadata)
- Instructions: assess cross-topic completeness, identify gaps no researcher covered, find contradictions *between* researchers
- `<field_criteria>` for `knowledge_state`, `should_continue`, `follow_ups`
- Follow-up criteria: when to deepen (partial knowledge_state), when to add new topic (uncovered angle), when to resolve contradiction (conflicting findings across researchers)

Use `<section>` tags and numbered instructions consistent with existing prompts.

### Step 4 — Conduct research tool

Create `nodes/supervisor/tools.py`:

```python
@tool
async def conduct_research(topic: str, context: str) -> str:
    """Research a specific topic. Each call spawns a focused researcher."""
    # Build ResearcherState
    initial_state = {
        "messages": [],
        "research_topic": f"{topic}\n\nContext: {context}",
        "research_iterations": 0,
        "last_reflection": "",
        "accumulated_findings": [],
        "accumulated_contradictions": [],
        "current_gaps": [],
        "notes": "",
    }
    # Run researcher subgraph
    result = await researcher_subgraph.ainvoke(initial_state, config)
    # Build ResearchResult from accumulated state
    research_result = ResearchResult(
        topic=topic,
        notes=result["notes"],
        key_findings=result["accumulated_findings"],
        knowledge_state=...,  # from last reflection's assessment
        missing_info=result["current_gaps"],
        contradictions=result["accumulated_contradictions"],
    )
    return research_result.model_dump_json()
```

The tool returns JSON string (LangChain tools return strings). `supervisor_tools` parses it back into `ResearchResult`.

Note: `knowledge_state` needs to come from the last reflection round. Options:
- Add `final_knowledge_state` field to `ResearcherState` (set by reflect on exit)
- Infer from `current_gaps` (empty = sufficient, non-empty = partial)

Decision: add `final_knowledge_state: str` to `ResearcherState`, set by reflect when routing to summarizer.

### Step 5 — Supervisor node + supervisor_tools node

Create `nodes/supervisor/supervisor.py`:

**`supervisor()` node**:
- Build system prompt from `research_brief` + formatted `research_results` + `last_supervisor_reflection`
- Context engineering: for `research_results`, include metadata (topic, knowledge_state, key_findings, missing_info, contradictions) but NOT full notes — notes are already compressed and the supervisor doesn't need to re-read them for dispatching. It needs to know *what was covered and what's missing*.
- Call LLM with `conduct_research` tool bound
- Clear prior messages on subsequent rounds (don't accumulate across supervisor rounds)
- Return `{"messages": [response]}`

**`supervisor_tools()` node**:
- Execute tool calls (each runs a researcher subgraph via `conduct_research`)
- Parse `ResearchResult` from each tool response
- Return `{"messages": tool_messages, "research_results": new_results}`

### Step 6 — Supervisor reflection node

Create `nodes/supervisor/reflect.py`:

**`supervisor_reflect()` node**:
- Format `research_results` as structured input: for each result, show topic, knowledge_state, key_findings, missing_info, contradictions
- Call model with `.with_structured_output(SupervisorReflection)`
- Routing:
  - Stop if: `not should_continue` or `knowledge_state == "sufficient"` or `iterations >= max`
  - Continue if: `should_continue` and `follow_ups` not empty

**On exit** (routing to END):
- Merge all `research_results[].notes` with topic headers:
  ```
  ## Topic: {result.topic}
  {result.notes}
  ```
- Write merged string to `notes` (maps to `AgentState.notes` via field name overlap)
- Return `Command(goto="__end__", update={"notes": combined, ...})`

**On continue** (routing back to supervisor):
- Format `follow_ups` as readable guidance in `last_supervisor_reflection`
- Clear `messages` (supervisor builds fresh context each round)
- Return `Command(goto="supervisor", update={"last_supervisor_reflection": formatted, "messages": [], ...})`

### Step 7 — Wire supervisor subgraph + update main graph

Create `nodes/supervisor/__init__.py`:

```python
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("supervisor_reflect", supervisor_reflect)
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("supervisor_tools", "supervisor_reflect")
# supervisor_reflect uses Command for conditional routing
supervisor_subgraph = supervisor_builder.compile()
```

Update `graph/graph.py`:

```python
# Before: write_brief → researcher → final_report
# After:  write_brief → supervisor → final_report
graph.add_node("supervisor", supervisor_subgraph)
graph.add_edge("write_brief", "supervisor")
graph.add_edge("supervisor", "final_report")
```

State mapping (automatic via field name overlap):
- In: `AgentState.research_brief` → `SupervisorState.research_brief`
- Out: `SupervisorState.notes` → `AgentState.notes`

### Step 8 — Configuration

Add to `configuration.py`:

- `max_supervisor_iterations: int = 2` — bounds supervisor reflection cycles (default low for testing)
- `max_research_topics: int = 5` — cap on initial decomposition (prompt-based, not enforced by routing)

### Step 9 — Tests + integration

**Unit tests** (no API calls):
- `ResearchReflection` accumulator behavior: findings append across rounds, gaps overwrite
- `SupervisorReflection` schema validation
- `FollowUp` schema validation  
- `ResearchResult` construction from accumulated researcher state
- State reducer for `research_results` (append behavior)
- Supervisor reflection formatting (research_results → prompt input)
- Combined notes formatting with topic headers

**Integration test**:
- Update initial state for new `AgentState` shape
- Verify supervisor decomposes into multiple subtopics
- Verify each researcher runs independently with isolated state
- Verify `research_results` contains metadata (key_findings, contradictions, etc.)
- Verify `notes` contains combined output with topic headers
- Verify follow-up round triggers when first round has gaps

---

## Observations from Step 0-1 testing

**Researcher receives full brief as topic (pre-supervisor)**: The `_run_researcher` adapter passes the entire research brief (title + 6 questions + 8 topics) as `research_topic`. The researcher treated this as one broad survey — batched 5 queries mapping ~1:1 to the research questions, got 68k chars of results, and reflection said "sufficient" after 1 round. This is expected: the supervisor will decompose into focused subtopics, each researcher gets one narrow topic and goes deeper.

**Reflection says "sufficient" after 1 round with broad topic**: With 5 broad queries returning comprehensive surface-level coverage, the model legitimately saw all questions addressed. Not a prompt issue — the topic is too broad for one researcher. Focused subtopics from the supervisor will naturally require multiple rounds to reach depth.

**Accumulated findings are substantive**: 16 key findings with source references (e.g., "companies raised $3.77 billion in the first nine months, nearly tripling 2024's total"). These are concrete facts, not vague summaries. The `key_findings` criteria update ("strategic observations") is working.

**Contradictions carry source references but not URLs**: The contradiction field cited "Source 2, 12, 13, 22" — numbered references within the Tavily search results, not standalone URLs. URLs live in the raw tool output and flow through the summarizer. Acceptable for now — contradictions are used for routing decisions at the supervisor level, not for direct citation.

**Compression ratio still aggressive**: 68k → 6.5k chars (9%). Expected with broad overlapping results. Should improve with focused subtopics producing less redundancy.

**Gemini requires HumanMessage**: `SystemMessage`-only calls fail with `ValueError: contents are required.` — Gemini maps SystemMessage to `system_instruction` (separate from `contents`). Added minimal HumanMessage ("Begin researching the topic described above.") as provider requirement.

---

## Resolved Questions

1. **State mapping between subgraphs**: LangGraph auto-maps fields by name when schemas overlap. `SupervisorState` has `research_brief` and `notes` (same names as `AgentState`), so the compiled subgraph can be added directly as a node — no wrapper needed. Extra fields (`research_results`, `supervisor_iterations`) are internal. For the researcher subgraph invoked from a tool, it's programmatic `.ainvoke()` with explicit state transform.

2. **Combined notes format**: Topic headers — `## Topic: {topic}\n{notes}` per researcher result.

3. **Supervisor messages**: `SupervisorState` includes `messages` for the LangGraph tool calling mechanism. Cleared each supervisor round to prevent blind accumulation.

4. **Knowledge preservation across stages**: Structured accumulator fields (`accumulated_findings`, `accumulated_contradictions`, `current_gaps`) replace the discard-after-use pattern. Reflection appends to accumulators; summarizer uses them for prioritization; `ResearchResult` carries them to the supervisor.

## Open Questions

1. **Supervisor context engineering**: With many researchers returning findings, the supervisor's input context can grow. Current plan: supervisor sees metadata (findings, gaps, contradictions) but not full notes for dispatching decisions. Supervisor reflection sees the same. Monitor in traces — if context grows too large, add a compression step for supervisor input.

2. **`knowledge_state` for ResearchResult**: Need `final_knowledge_state` in `ResearcherState` (set by reflect on exit). Could alternatively infer from `current_gaps` (empty → sufficient). Current plan: explicit field.

---

## File tree after Increment 3

```
src/deep_research/
    __init__.py               # + ResearchReflection, SupervisorReflection, ResearchResult, FollowUp exports
    configuration.py          # + max_supervisor_iterations, max_research_topics
    models.py                 # + SupervisorReflection, FollowUp, ResearchResult; Reflection → ResearchReflection
    prompts.py                # + supervisor_system_prompt, supervisor_reflection_prompt; updated reflection_prompt
    state.py                  # + SupervisorState, ResearcherState; AgentState simplified
    graph/
        graph.py              # write_brief → supervisor_subgraph → final_report
    nodes/
        brief.py              # unchanged
        report.py             # unchanged
        researcher/
            __init__.py       # researcher_subgraph wiring (ResearcherState)
            researcher.py     # researcher + researcher_tools nodes
            reflect.py        # research reflection node (with accumulation)
            summarizer.py     # renamed from compress.py
        supervisor/
            __init__.py       # supervisor_subgraph wiring
            supervisor.py     # supervisor + supervisor_tools nodes
            reflect.py        # supervisor reflection node
            tools.py          # conduct_research tool
```
