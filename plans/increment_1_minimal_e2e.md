# Increment 1 — Minimal End-to-End
**Goal**: Question in → researched markdown report out.
**Graph**: `write_brief` → `researcher` (subgraph) → `final_report`
**Status**: Complete

## Overview

The pipeline takes a user question, generates a structured research brief, runs a researcher agent that searches and synthesizes findings, then produces a final markdown report with citations.

**Components**:
- `Configuration` — Pydantic config with per-role model settings, env var resolution
- `AgentState` — TypedDict with messages, research_brief, notes, final_report
- `configurable_model` — deferred model factory via `init_chat_model`, no hardcoded defaults
- `BaseSearchTool` ABC — shared summarization/formatting, subclasses only implement `search()`
- `TavilySearchTool` + tool registry — dynamic tool assembly from config
- Three graph nodes: `write_brief`, `researcher` (two-node subgraph), `final_report`
- Observability: LangSmith tracing + Python logging + fallback flagging via trace metadata

## Steps

Each step produces a reviewable, testable, committable change.

---

### Step 1 — Bootstrap the package ✓

Create the minimal Python package skeleton with dependencies.

**Files created**:
```
pyproject.toml             # package metadata, dependencies
src/deep_research/
    __init__.py            # package init, version
```

**Dependencies** (initial):
- `langgraph>=0.4`, `langchain-core>=0.3`, `langchain>=0.3`
- `langchain-google-genai>=2.1` (Gemini provider)
- `tavily-python>=0.5`
- `pydantic>=2.0`, `python-dotenv>=1.0`
- Dev: `pytest>=8.0`, `pytest-asyncio>=0.24`

**Added**: `.env.example`, `.gitignore`, `SETUP.md` (uv setup guide).

---

### Step 2 — Configuration ✓

Minimal config: which model to use, API keys from env.

**Files created**:
```
src/deep_research/configuration.py
```

**What it contains**:
- `Configuration` Pydantic class with:
  - `research_model: str` (default: `"google_genai:gemini-2.5-flash"`)
  - `research_model_max_tokens`, `research_model_temperature`
  - `research_model_thinking_budget: Optional[int]` (default: 4096) — added in Step 13
  - `summarization_model: str` (default: `"google_genai:gemini-2.5-flash-lite"`)
  - `summarization_model_max_tokens`, `summarization_model_temperature`
  - `summarization_model_thinking_budget: Optional[int]` (default: 0) — added in Step 13
  - `search_api: SearchAPI` enum (default: `TAVILY`)
  - `max_search_results: int` (default: `5`)
  - `max_tool_call_rounds`, `max_structured_output_retries`
- **Changed**: `from_runnable_config()` instead of `from_env()` — resolution order: env vars > runtime config > defaults. Aligns with LangGraph's config passing pattern.

---

### Step 3 — State contract ✓

Define the minimal state that flows through the graph.

**Files created**:
```
src/deep_research/state.py
```

**What it contains**:
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    research_brief: str                       # structured brief from user query
    notes: str                                # accumulated research findings
    final_report: str                         # the output
```

All fields except `messages` use last-write-wins (no reducer). Custom reducers deferred to Increment 3 when multiple researchers write to shared state.

---

### Step 4 — Pydantic models ✓

Data schemas used across the system.

**Files created**:
```
src/deep_research/models.py
```

**What it contains**:
- `ResearchBrief(BaseModel)`: `title`, `research_questions: list[str]`, `key_topics: list[str]`
- `SearchResult(BaseModel)`: `url`, `title`, `content`, `raw_content: str | None`
- `WebpageSummary(BaseModel)`: `summary`, `key_excerpts`

These are shared data contracts — not tools, not state. Clean separation.

---

### Step 5 — Configurable model ✓

Shared deferred model factory using LangChain's `init_chat_model` pattern.

**Files created**:
```
src/deep_research/graph/__init__.py
src/deep_research/graph/model.py
```

**What it contains**:
```python
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "temperature", "thinking_budget"),
)
```

**Changed**: No hardcoded default model — fully deferred. `thinking_budget` added to `configurable_fields` to pass through to Gemini; conditionally applied in nodes (only when not None) so non-Gemini providers don't break.

**Why this pattern** (instead of a `get_model()` helper):
- One shared object, configured per-use via `.with_config()`
- Chains naturally with `.bind_tools()`, `.with_structured_output()`, `.with_retry()`
- Provider swapping is just a string change: `"google_genai:gemini-2.5-flash"` → `"anthropic:claude-sonnet-4-20250514"`
- `temperature` added beyond reference — gives per-role control (low for summarization, moderate for research)

**How nodes use it** — build config dict, conditionally add provider-specific params:
```python
model_config = {
    "model": configurable.research_model,
    "max_tokens": configurable.research_model_max_tokens,
    "temperature": configurable.research_model_temperature,
}
if configurable.research_model_thinking_budget is not None:
    model_config["thinking_budget"] = configurable.research_model_thinking_budget

model = (
    configurable_model
    .bind_tools(tools)
    .with_retry(stop_after_attempt=3)
    .with_config(configurable=model_config)
)
```

**Other LangChain Runnable patterns to use later**:
- `.with_retry()` — already using for structured output (Increment 1)
- `.with_fallbacks([fallback_model])` — automatic failover on rate limits (Increment 5+)
- `.configurable_alternatives()` — swap entire implementations at runtime; considered but our ABC inheritance is cleaner for search tool swapping

---

### Step 6 — Search tool: base interface ✓

Define the abstract interface that all search providers implement.

**Files created**:
```
src/deep_research/tools/__init__.py
src/deep_research/tools/search/__init__.py
src/deep_research/tools/search/base.py
```

**What `base.py` contains**:
```python
class BaseSearchTool(ABC):
    @abstractmethod
    async def search(self, queries: list[str], *, max_results: int = 5) -> list[SearchResult]:
        """Execute search queries and return results."""
        ...
```

**Changed**: Only `search()` is abstract. `summarize_content()`, `search_and_summarize()`, and `_format_results()` are concrete shared methods in the base class — subclasses only implement the provider-specific API call.

Constants: `MAX_CONTENT_LENGTH=50000`, `SUMMARIZATION_TIMEOUT=60.0`.

---

### Step 7 — Search tool: Tavily implementation ✓

Concrete search provider inheriting from the base.

**Files created**:
```
src/deep_research/tools/search/tavily.py
src/deep_research/tools/registry.py
tests/test_search.py
```

**What it contains**:
- `TavilySearchTool(BaseSearchTool)` — only implements `search()` (Tavily API call + URL dedup)
- `@tool`-decorated `tavily_search` function for LangGraph tool binding
- **Added**: `tools/registry.py` with `get_all_tools(config)` — dynamic tool assembly from config. Nodes never reference specific tools directly.
- 5 tests: search returns results, dedup, raw content, summarize_content, search_and_summarize pipeline

---

### Step 8 — Prompts ✓

All prompt templates for this stage.

**Files created**:
```
src/deep_research/prompts.py
```

**What it contains** (only the prompts needed for Increment 1):
- `research_brief_prompt`: Instruct the model to extract a structured research brief from user messages. Includes `{date}` and `{messages}` placeholders.
- `research_system_prompt`: Researcher instructions — search for information on the given topic, use search tools, stay focused. Includes `{topic}`, `{date}`.
- `summarize_webpage_prompt`: Instruct model to extract key info and excerpts from raw webpage content. Includes `{webpage_content}`, `{date}`.
- `final_report_prompt`: Generate a comprehensive markdown report from research findings. Includes `{brief}`, `{notes}`, `{date}`.

We write our own prompts, informed by but not copied from the reference. Draft versions — calibrate later with LangSmith observability.

**Note**: Prompts were written alongside their respective nodes rather than as a separate step.

---

### Step 9 — Node: write_brief ✓

First graph node. Transforms user question into structured research brief.

**Files created**:
```
src/deep_research/nodes/__init__.py
src/deep_research/nodes/brief.py
tests/test_nodes.py
```

**What it contains**:
```python
async def write_research_brief(state: AgentState, config: RunnableConfig) -> dict:
```
- Reads `messages` from state
- Calls model with `research_brief_prompt` + `.with_structured_output(ResearchBrief)`
- Returns `{"research_brief": brief_as_string}` — formatted string for downstream consumption
- 2 tests: returns non-empty brief, has expected structure (Title/Questions/Topics)

---

### Step 10 — Node: researcher ✓

The researcher node — searches and accumulates notes.

**Files created**:
```
src/deep_research/nodes/researcher.py
```

**Changed**: Instead of single-pass, implemented as a two-node subgraph pattern:

```python
async def researcher(state, config) -> Command[Literal["researcher_tools"]]:
    # Single LLM call with bound tools

async def researcher_tools(state, config) -> Command[Literal["researcher", "__end__"]]:
    # Execute tool calls in parallel, route back or end
```

- `researcher` — one LLM invocation, returns `Command(goto="researcher_tools")`
- `researcher_tools` — parallel tool execution via `asyncio.gather`, routes to `"researcher"` (more searching) or `"__end__"` (done)
- Loop continues until model stops calling tools or `max_tool_call_rounds` reached
- Uses `.text` (not `.content`) for provider-generic text extraction
- Fallback to raw tool results on empty summary, flagged in LangSmith trace

---

### Step 11 — Node: final_report ✓

Generate the final markdown report from accumulated notes.

**Files created**:
```
src/deep_research/nodes/report.py
```

**What it contains**:
```python
async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
```
- Reads `research_brief` and `notes` from state
- Calls model with `final_report_prompt`
- Returns `{"final_report": report_markdown}`
- Fallback to raw notes on empty output, flagged in LangSmith trace

No token-limit retry yet — that's Increment 5.

---

### Step 12 — Graph wiring ✓

Wire all three nodes into a LangGraph and expose it.

**Files created**:
```
src/deep_research/graph/graph.py
```

**What it contains**:
```python
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("write_brief", write_research_brief)
    graph.add_node("researcher", researcher_subgraph)
    graph.add_node("final_report", final_report_generation)
    graph.add_edge(START, "write_brief")
    graph.add_edge("write_brief", "researcher")
    graph.add_edge("researcher", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile()

deep_researcher = build_graph()
```

**Changed**: Researcher is a compiled subgraph (two internal nodes), not a single node. Main graph still treats it as one node.

---

### Step 13 — End-to-end integration test ✓

Run the full pipeline with a real question.

**Files created**:
```
tests/test_integration.py
src/deep_research/logging_config.py
```

**What it does**:
- Creates graph via `build_graph()`
- Invokes with a question, asserts all state fields populated, report >500 chars with markdown headings

**Added (not in original plan)**:
- LangSmith tracing (env vars, auto-instrumented via LangChain)
- Python logging in all nodes (`logging_config.py`)
- `thinking_budget` control — Gemini reasoning tokens can consume entire output budget without a cap
- Fallback flagging via `get_current_run_tree().metadata` for human review in LangSmith

---

### Step 14 — Public API re-exports ✓

Public API surface for external consumers.

**Changed**: Re-exported from `__init__.py` directly instead of a separate `utils.py`.

Exports: `deep_researcher`, `build_graph`, `AgentState`, `Configuration`.

---

## File tree after Increment 1

```
src/deep_research/
    __init__.py            # public API re-exports + logging setup
    configuration.py
    logging_config.py
    models.py
    prompts.py
    state.py
    graph/
        __init__.py
        graph.py           # build_graph(), deep_researcher
        model.py           # configurable_model (deferred factory)
    nodes/
        __init__.py
        brief.py
        report.py
        researcher.py      # two-node subgraph
    tools/
        __init__.py
        registry.py        # get_all_tools()
        search/
            __init__.py
            base.py        # BaseSearchTool ABC
            tavily.py
tests/
    __init__.py
    test_integration.py
    test_nodes.py
    test_search.py
```

## TODOs

- Model config resolver for provider-aware param handling (configuration.py)
- Extract dedup-by-URL into `BaseSearchTool._deduplicate()` (tavily.py)
- Refine tavily_search tool description when multiple tools added (tavily.py)
- Consider passing original user messages alongside brief for researcher context (brief.py)
- Cache tool list across researcher rounds (researcher.py)

## What this increment does NOT include (deferred)
- Researcher reflection loop (Increment 2)
- Research compression (Increment 2)
- Coordinator / multi-topic (Increment 3)
- Separate ResearcherState for message isolation (Increment 3)
- Question stage: clarification + complexity assessment + brief reform (Increment 4)
- Multi-provider model support + config resolver (Increment 6)
- Token-limit handling (Increment 5)
- Dead-end / contradiction detection (Increment 5)
- Retryable vs permanent tool error distinction (Increment 5)
- CLI interface (Increment 6)
