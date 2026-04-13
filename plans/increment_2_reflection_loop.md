# Increment 2 — Researcher Reflection Loop
**Goal**: Researcher iterates based on what it learns — system-controlled, not model-controlled.
**Subgraph**: `researcher` → `researcher_tools` → `reflect` → (`researcher` | `compress` → END)

## Overview

Currently the researcher loops until the model stops calling tools or hits max rounds — the *model* decides when to stop. Increment 2 adds **structured reflection** so the *system* controls routing based on typed fields (knowledge_state, missing_info, should_continue). It also adds **research compression** before returning findings to the main graph.

**New components**:
- `Reflection` Pydantic schema — structured output with key_findings, missing_info, knowledge_state, etc.
- `reflect` node — calls model with `.with_structured_output(Reflection)`, routes on fields
- `compress` node — extracts all ToolMessage content from messages, compresses into concise notes
- Two new prompts: reflection + compression
- `max_research_iterations` + `research_iterations` state field

**What changes**:
- Researcher subgraph gains two new nodes (`reflect`, `compress`)
- Routing moves from model-controlled (stop calling tools) to system-controlled (reflection fields)
- `AgentState` gains `research_iterations: int` for proper iteration tracking
- Notes are built from raw tool results and compressed before reaching the report node

**What stays the same**:
- Main graph wiring (`write_brief` → `researcher` → `final_report`)
- `researcher` node — reads reflection from state to guide next round
- `researcher_tools` node — simplified (always routes to `reflect`)

---

## Steps

### Step 1 — Reflection schema + state update

Add the `Reflection` Pydantic model to `models.py` and `research_iterations` to state.

**File modified**: `src/deep_research/models.py`

```python
class Reflection(BaseModel):
    """Structured reflection after a research round."""
    key_findings: list[str]      # what we learned this round
    missing_info: list[str]      # gaps still remaining
    contradictions: list[str]    # conflicting information found (within this topic)
    knowledge_state: Literal["insufficient", "partial", "sufficient"]
    should_continue: bool        # model's recommendation
    next_queries: list[str]      # what to search next if continuing
```

`knowledge_state` replaces `confidence: float`:
- `"insufficient"` — major gaps, core questions unanswered
- `"partial"` — some gaps but making progress
- `"sufficient"` — can answer the research question

Contradictions here are **researcher-level** (within one topic). Global contradictions across topics come in Increment 3 when the supervisor compares findings from multiple researchers.

Contradictions are captured in the schema but **no special processing logic yet** — the model naturally considers them when generating `next_queries`. Explicit contradiction resolution routing is deferred to Increment 5.

**File modified**: `src/deep_research/state.py`

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_brief: str
    notes: str
    final_report: str
    research_iterations: int     # NEW — reflection cycle count
    last_reflection: str         # NEW — serialized Reflection for researcher to read
```

`research_iterations`: Proper counter, incremented by reflect node each cycle.

`last_reflection`: Formatted string of the full Reflection output (missing_info, contradictions, next_queries). The researcher node reads this to guide its next round of searching — it sees *why* it's continuing, not just what to search.

---

### Step 2 — Prompts

Add reflection and compression prompts to `prompts.py`.

**File modified**: `src/deep_research/prompts.py`

**reflection_prompt**: Instruct the model to assess research progress so far.
- Placeholders: `{research_brief}` (original goal), `{findings}` (all tool results so far)
- Must produce structured `Reflection` output
- **Criteria for `knowledge_state`** (the model needs concrete guidelines, not just the field name):
  - `"insufficient"`: core research questions unanswered, or fewer than 2 supporting sources
  - `"partial"`: some questions answered but notable gaps remain
  - `"sufficient"`: all research questions addressed with supporting sources
- **Criteria for `should_continue`** (escape hatch — when to stop even if knowledge is incomplete):
  - `False` when: last searches returned mostly overlapping info, topic is too niche for web search, remaining gaps require expertise/data that search can't provide
  - `True` when: concrete gaps exist that targeted queries could fill
- **`missing_info`**: must be specific and actionable (e.g. "no data on 2024 revenue figures"), not vague ("more info needed")
- **`contradictions`**: cite which sources disagree and on what
- These criteria are draft — calibrate after reviewing real traces in LangSmith

**compress_research_prompt**: Instruct the model to compress raw tool results into concise notes.
- Placeholders: `{research_brief}` (for relevance filtering), `{tool_results}` (raw ToolMessage content extracted from messages)
- Key instructions: preserve citations/URLs, deduplicate overlapping info, keep specific facts/data/quotes, remove boilerplate, target ~30% of input length

---

### Step 3 — Reflect node

New node that performs structured reflection after tool execution.

**File created**: `src/deep_research/nodes/reflect.py`

```python
async def reflect(state: AgentState, config: RunnableConfig) -> Command[Literal["researcher", "compress"]]:
```

**What it does**:
1. Extracts all ToolMessage content from messages as findings
2. Calls model with `reflection_prompt` + `.with_structured_output(Reflection)` + `.with_retry()`
3. Increments `research_iterations`
4. Routes based on:
   - `should_continue=False` → `compress`
   - `knowledge_state == "sufficient"` → `compress`
   - `research_iterations >= max_research_iterations` → `compress` (forced)
   - Otherwise → `researcher`
5. When routing to `researcher`: writes full reflection to `last_reflection` state field
6. When routing to `compress`: clears `last_reflection`
7. Logs: knowledge_state, gap count, contradiction count, iteration number, routing decision

**No fake messages**: Reflection feedback passes through `last_reflection` state field, not injected as HumanMessage/SystemMessage. The researcher reads it from state.

---

### Step 4 — Compress node

New node that compresses raw tool results before returning.

**File created**: `src/deep_research/nodes/compress.py`

```python
async def compress_research(state: AgentState, config: RunnableConfig) -> dict:
```

**What it does**:
1. Extracts all ToolMessage content from `messages` — these are the raw search results
2. If total content is short enough (below threshold, e.g. <2000 chars), skip LLM compression
3. Calls model with `compress_research_prompt` (uses **summarization model** — cheaper, no reasoning needed)
4. Returns `{"notes": compressed_notes}`
5. Logs compression ratio (input chars → output chars)

**Input is raw tool results**, not any model summary. This ensures compression sees the original exact information from search, no information lost through an intermediate model pass.

---

### Step 5 — Refactor researcher subgraph

Update the researcher subgraph to include the new nodes.

**File modified**: `src/deep_research/nodes/researcher.py`

**New subgraph wiring**:
```
START → researcher → researcher_tools → reflect → (researcher | compress → END)
```

**Changes to `researcher_tools`**:
- Remove routing logic — no longer decides whether to end
- Remove the "no tool calls → extract notes" exit path
- Always routes to `reflect` via edge (not Command-based routing)
- Still executes tool calls in parallel, still respects `max_tool_call_rounds`
- When model produces no tool calls: pass through to `reflect` (reflect decides if truly done)

**Changes to `researcher`**:
- Reads `last_reflection` from state — if present, incorporates into system prompt as research guidance (what's missing, what contradictions to resolve, suggested queries)
- First invocation: no reflection available, proceeds as before

**`researcher_tools` no longer writes `notes`**. Notes flow: tool results in messages → compress extracts → compress writes `notes`.

---

### Step 6 — Configuration

Add reflection-related config field.

**File modified**: `src/deep_research/configuration.py`

```python
max_research_iterations: int = Field(
    default=3,
    description="Maximum reflection-research cycles before forcing exit",
)
```

Two distinct limits serve different failure modes:
- `max_tool_call_rounds` — bounds tool calls *within* a single research round (model keeps calling tools without stopping)
- `max_research_iterations` — bounds the outer reflect loop (reflection keeps saying "continue")

---

### Step 7 — Tests + integration

**File modified**: `tests/test_nodes.py` — Add unit tests:
- Reflection routing: `should_continue=False` → compress
- Reflection routing: `knowledge_state="sufficient"` → compress
- Reflection routing: max iterations reached → compress (forced)
- Reflection routing: `knowledge_state="insufficient"` + `should_continue=True` → researcher
- Compress: short content skips LLM compression
- Compress: long content is compressed with ratio logged

**File modified**: `tests/test_integration.py` — Update e2e test:
- Verify the pipeline still produces a report
- Verify `research_iterations >= 1` (at least one reflection cycle happened)

---

## File tree changes (Increment 2)

```
src/deep_research/
    models.py              # + Reflection schema
    prompts.py             # + reflection_prompt, compress_research_prompt
    state.py               # + research_iterations, last_reflection
    nodes/
        reflect.py         # NEW — structured reflection node
        compress.py        # NEW — research compression node
        researcher.py      # MODIFIED — subgraph wiring, researcher_tools simplified
    configuration.py       # + max_research_iterations
```

## Design decisions

**Why `knowledge_state` enum instead of `confidence: float`**:
- `"insufficient" / "partial" / "sufficient"` is immediately readable in traces and logs
- No ambiguity about thresholds (what does 0.6 mean?)
- Routing logic is cleaner: `if knowledge_state == "sufficient"` vs `if confidence >= 0.8`

**Why reflection passes through state (not messages)**:
- `HumanMessage` should only come from the user — injecting fake ones is confusing
- `SystemMessage` mid-conversation is non-standard and clutters message history
- State field `last_reflection` is explicit, readable, and doesn't pollute the conversation

**Why researcher sees full reflection (not just queries)**:
- `missing_info` tells the researcher *what gaps* to fill, not just *what to search*
- `contradictions` tells the researcher to look for resolution, not just more data
- `next_queries` are suggestions, but the researcher may choose better queries given full context

**Why compress sees raw tool results (not model summary)**:
- No information loss through intermediate summarization
- Compression model works from original source data
- Model summaries can hallucinate or omit details that compression would preserve

**Contradictions: captured now, processed later**:
- Reflection schema captures `contradictions` field — model fills it naturally
- No special routing logic yet (no "if contradictions → target resolution searches")
- The model already considers contradictions when generating `next_queries`
- Explicit contradiction resolution routing deferred to Increment 5

**Two distinct iteration limits**:
- `max_tool_call_rounds` (existing, default 10) — within one research round, bounds tool calls
- `max_research_iterations` (new, default 3) — bounds reflect→research cycles
- Both needed: different failure modes (tool-calling runaway vs. reflection-loop runaway)

## What this increment does NOT include (deferred)
- Dead-end detection: comparing `missing_info` across rounds (Increment 5)
- Contradiction resolution: explicit routing when contradictions found (Increment 5)
- Supervisor-level contradictions across topics (Increment 3)
- Supervisor low-confidence handling (Increment 3)
- Two-tier notes: raw + compressed (Increment 6)
