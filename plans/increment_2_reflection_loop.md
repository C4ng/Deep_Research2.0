# Increment 2 — Researcher Reflection Loop
**Goal**: Researcher iterates based on what it learns — system-controlled, not model-controlled.
**Subgraph**: `researcher` → `researcher_tools` → `reflect` → (`researcher` | `compress` → END)
**Status**: Complete

## Overview

The researcher now operates in a reflection loop: search → reflect → (continue or compress). Routing is system-controlled via structured `Reflection` fields (knowledge_state, missing_info, should_continue), not model-controlled (model stopping tool calls).

**Components added**:
- `Reflection` Pydantic schema — structured output with key_findings, missing_info, knowledge_state, contradictions, next_queries
- `reflect` node — calls model with `.with_structured_output(Reflection)`, routes on fields
- `compress` node — extracts all ToolMessage content, compresses into concise notes via summarization model
- Two new prompts: `reflection_prompt`, `compress_research_prompt`
- State fields: `research_iterations`, `last_reflection`
- Config: `max_research_iterations`, `max_searches_per_round`

**What changed from plan**:
- Removed inner tool-calling loop (researcher ↔ researcher_tools). One LLM call per round, then reflect. No `max_tool_call_rounds`.
- `max_tool_call_rounds` replaced by `max_searches_per_round` (prompt-based, not routing-based)
- Researcher trims prior AI/Tool messages on subsequent rounds to prevent thinking budget exhaustion from context bloat
- Research model budgets expanded (max_tokens: 8192→16384, thinking_budget: 4096→8192)
- `key_findings` included in reflection feedback so researcher avoids redundant searches
- Reflection prompt uses `<section>` tags and `<field_criteria>` for consistency with other prompts

---

## Steps

### Step 1 — Reflection schema + state update ✓

**File modified**: `src/deep_research/models.py`

```python
class Reflection(BaseModel):
    key_findings: list[str]
    missing_info: list[str]
    contradictions: list[str]       # default_factory=list
    knowledge_state: Literal["insufficient", "partial", "sufficient"]
    should_continue: bool
    next_queries: list[str]         # default_factory=list
```

**Changed**: `confidence: float` replaced with `knowledge_state` enum — numerical scores aren't intuitive for review. Categories are immediately readable in traces.

Field descriptions kept minimal and consistent with other models (plain "what this field is"). Judgment criteria (what "sufficient" means, when to set `should_continue=False`) live in the reflection prompt, not field descriptions.

**File modified**: `src/deep_research/state.py` — Added `research_iterations: int`, `last_reflection: str`.

---

### Step 2 — Prompts ✓

**File modified**: `src/deep_research/prompts.py`

**reflection_prompt**: Uses `<instructions>` and `<field_criteria>` tags for consistency.
- Concrete criteria for `knowledge_state` and `should_continue` in `<field_criteria>`
- Instructions: compare findings against brief, identify gaps, note contradictions
- These criteria are draft — calibrate after reviewing real traces

**compress_research_prompt**: Compress raw tool results.
- Preserve citations/URLs, deduplicate, keep facts/data, target ~30% of input length
- Uses `{research_brief}` for relevance filtering, `{tool_results}` for raw content

**Changed**: `research_system_prompt` updated to tell researcher about reflection loop, remove "stop when confident" (reflection controls stopping), add `{max_searches_per_round}` placeholder, instruct to act on reflection guidance.

---

### Step 3 — Reflect node ✓

**File created**: `src/deep_research/nodes/reflect.py`

- `_extract_tool_results()` — extracts all ToolMessage content
- `_format_reflection()` — formats full Reflection (key_findings, missing_info, contradictions, next_queries) as readable text for `last_reflection`
- `reflect()` — structured reflection, routes to `researcher` or `compress`

**Changed**: `key_findings` included in formatted reflection output so researcher knows what's already covered and avoids redundant searches.

Reflection feedback passes through `last_reflection` state field, not messages. `HumanMessage` is only for user input.

---

### Step 4 — Compress node ✓

**File created**: `src/deep_research/nodes/compress.py`

- Extracts raw ToolMessage content from messages
- Skips LLM compression below 2000 chars threshold
- Uses summarization model (cheaper, no reasoning)
- Falls back to raw tool results if compression returns empty

---

### Step 5 — Refactor researcher subgraph ✓

**File modified**: `src/deep_research/nodes/researcher.py`

**Changed from plan**: Removed inner tool-calling loop entirely.

Original plan had `researcher ↔ researcher_tools` inner loop bounded by `max_tool_call_rounds`. Removed because:
- The `tavily_search` tool accepts `queries: List[str]` — model sends multiple queries in one call
- Reflection handles iteration control — inner loop was redundant
- Inner loop required per-iteration counting which was complex and error-prone

New flow is strictly linear per round: `researcher → researcher_tools → reflect`.

**Context management**: On subsequent rounds (`research_iterations > 0`), researcher trims prior AIMessages and ToolMessages from the LLM call. Reflection already captured key findings; full history stays in state for compress. This prevents thinking budget exhaustion from growing context.

Both `researcher` and `researcher_tools` return plain dicts now (no Command routing). Edges handle flow. Only `reflect` uses Command for conditional routing.

---

### Step 6 — Configuration ✓

**File modified**: `src/deep_research/configuration.py`

**Changed**:
- `max_tool_call_rounds` removed — inner loop eliminated
- `max_searches_per_round` added (default 3) — injected into prompt, not enforced by routing
- `max_research_iterations` added (default 3) — bounds reflect→research cycles
- `research_model_max_tokens` increased to 16384
- `research_model_thinking_budget` increased to 8192

---

### Step 7 — Tests + integration ✓

**File modified**: `tests/test_nodes.py` — Unit tests (no API calls):
- `_extract_tool_results`: extracts ToolMessage content, skips empty
- `_format_reflection`: formats all fields, handles minimal case
- `compress_research`: short content skips compression, empty returns ""

**File modified**: `tests/test_integration.py`:
- Verifies `research_iterations >= 1`
- New state fields in initial state

**File modified**: `src/deep_research/__init__.py`:
- `Reflection` added to public exports

---

## File tree after Increment 2

```
src/deep_research/
    __init__.py            # + Reflection export
    configuration.py       # + max_research_iterations, max_searches_per_round
    models.py              # + Reflection schema
    prompts.py             # + reflection_prompt, compress_research_prompt; updated research_system_prompt
    state.py               # + research_iterations, last_reflection
    nodes/
        reflect.py         # NEW — structured reflection node
        compress.py        # NEW — research compression node
        researcher.py      # MODIFIED — no inner loop, context trimming
```

## Design decisions

**Why `knowledge_state` enum instead of `confidence: float`**:
- `"insufficient" / "partial" / "sufficient"` is immediately readable in traces and logs
- No ambiguity about thresholds (what does 0.6 mean?)
- Routing logic is cleaner: `if knowledge_state == "sufficient"` vs `if confidence >= 0.8`

**Why reflection passes through state (not messages)**:
- `HumanMessage` should only come from the user — injecting fake ones is confusing
- State field `last_reflection` is explicit, readable, and doesn't pollute the conversation

**Why researcher sees full reflection (not just queries)**:
- `key_findings` tells the researcher what's already covered — avoids redundant searches
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

**Why no inner tool-calling loop**:
- `tavily_search` accepts `queries: List[str]` — model batches multiple queries per call
- Reflection handles iteration control — inner loop was redundant
- Per-iteration tool call counting was complex and error-prone
- One LLM call → tools → reflect is simpler and gives reflection more control

**Why trim messages on subsequent rounds**:
- Gemini's thinking tokens consumed entire output budget with large context (~60k chars)
- Prior tool results and AI summaries are redundant after reflection captures key findings
- Full history stays in state for compress — trimming is only for the LLM call
- `research_iterations > 0` detects subsequent rounds (SystemMessage not in state)

**Field descriptions vs prompt criteria**:
- Pydantic Field descriptions say *what the field is* (consistent with other models)
- The reflection prompt provides *judgment criteria* (what "sufficient" means, when to stop)
- Schema is stable; prompt criteria are draft and calibrated from traces

---

## Observations from integration testing

**Researcher searches one topic per round**: With the full research brief (7+ questions), the model picks one angle per round. Expected — Increment 3's supervisor decomposes into subtopics, each researcher gets a focused topic.

**Compression ratio is aggressive (7% vs target 30%)**: 132k chars → 8.8k chars. The broad topic produces overlapping search results with high redundancy. With focused subtopics in Increment 3, searches will overlap less and compression ratio should approach the 30% target.

**Thinking budget exhaustion on large context**: Gemini's reasoning tokens consumed the entire output budget when prior search results accumulated in messages (~60k chars). Fixed by trimming prior AI/Tool messages on subsequent rounds. Expanding model budgets (max_tokens=16384, thinking_budget=8192) provided additional headroom.

**Model batches queries in tool calls**: The `tavily_search` tool accepts `queries: List[str]`, so the model sends 3-4 queries per tool call. One tool call = multiple searches. The prompt limit "up to N search calls" maps to tool invocations, not individual queries.

## TODOs

- Calibrate reflection prompt criteria from LangSmith traces (knowledge_state thresholds, should_continue)
- Tune compression prompt — current 7% ratio may lose detail for focused subtopics
- Consider token-based context trimming (via `trim_messages`) as a complement to type-based filtering
- Dead-end detection: compare `missing_info` across rounds (Increment 5)

## What this increment does NOT include (deferred)
- Dead-end detection: comparing `missing_info` across rounds (Increment 5)
- Contradiction resolution: explicit routing when contradictions found (Increment 5)
- Supervisor-level contradictions across topics (Increment 3)
- Supervisor low-confidence handling (Increment 3)
- Two-tier notes: raw + compressed (Increment 6)
