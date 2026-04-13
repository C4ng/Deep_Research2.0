# Increment 1 — Minimal End-to-End
**Goal**: Question in → researched markdown report out.
**Graph**: `write_brief` → `researcher` → `final_report` (linear, no loops)

## Steps

Each step produces a reviewable, testable, committable change.

---

### Step 1 — Bootstrap the package

Create the minimal Python package skeleton with dependencies.

**Files created**:
```
pyproject.toml             # package metadata, dependencies
src/deep_research/
    __init__.py            # package init, version
```

**Dependencies** (initial):
- `langgraph`
- `langchain-google-genai` (Gemini provider)
- `langchain-core`
- `tavily-python`
- `pydantic`

**Verify**: `pip install -e .` succeeds, `import deep_research` works.

---

### Step 2 — Configuration

Minimal config: which model to use, API keys from env.

**Files created**:
```
src/deep_research/configuration.py
```

**What it contains**:
- `Configuration` Pydantic class with:
  - `research_model: str` (default: `"google_genai:gemini-2.0-flash"`)
  - `summarization_model: str` (default: same or cheaper Gemini)
  - `search_api: str` (default: `"tavily"`)
  - `max_search_results: int` (default: `5`)
- `from_env()` classmethod that reads `GOOGLE_API_KEY`, `TAVILY_API_KEY` from env
- Keep it small — this grows in Increment 4

**Verify**: Instantiate config, confirm env vars load.

---

### Step 3 — State contract

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

**Verify**: Can instantiate state dict, type-checks pass.

---

### Step 4 — Pydantic models

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

**Verify**: Can instantiate each model with sample data.

---

### Step 5 — Model helper

The model provider abstraction — one function, one provider for now.

**Files created**:
```
src/deep_research/helpers/__init__.py
src/deep_research/helpers/model.py
```

**What it contains**:
```python
def get_model(model_name: str, *, max_tokens: int | None = None, **kwargs) -> BaseChatModel
```
- Uses `langchain.chat_models.init_chat_model()` under the hood
- For Increment 1, just wraps the call. Later increments add provider routing, API key lookup per provider.
- Reads API key from configuration/env

**Verify**: `get_model("google_genai:gemini-2.0-flash")` returns a callable model. A simple `.invoke()` test confirms Gemini connectivity.

---

### Step 6 — Search tool: base interface

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

    @abstractmethod
    async def summarize_content(self, content: str) -> WebpageSummary:
        """Summarize raw webpage content."""
        ...
```

- Uses `SearchResult` and `WebpageSummary` from `models.py`
- No implementation yet — just the contract

**Verify**: Cannot instantiate (abstract). Import succeeds.

---

### Step 7 — Search tool: Tavily implementation

Concrete search provider inheriting from the base.

**Files created**:
```
src/deep_research/tools/search/tavily.py
```

**What it contains**:
- `TavilySearchTool(BaseSearchTool)`:
  - `__init__(self, api_key: str, summarization_model: BaseChatModel)` — takes its dependencies explicitly
  - `search()`: calls Tavily async API, deduplicates by URL, summarizes each result's raw content
  - `summarize_content()`: uses the summarization model with a prompt to produce `WebpageSummary`
- Also exports a `@tool`-decorated `tavily_search` function (LangChain tool for binding to the researcher agent). This function internally instantiates or receives a `TavilySearchTool`.
- `summarize_content` helper lives here (not in utils) since it's search-specific

**Verify**: Call `tavily_search` with a test query, confirm results come back with summaries.

---

### Step 8 — Prompts

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

We write our own prompts, informed by but not copied from the reference.

**Verify**: Format each prompt with sample values, confirm they read well.

---

### Step 9 — Node: write_brief

First graph node. Transforms user question into structured research brief.

**Files created**:
```
src/deep_research/nodes/__init__.py
src/deep_research/nodes/brief.py
```

**What it contains**:
```python
async def write_research_brief(state: AgentState, config: RunnableConfig) -> dict:
```
- Reads `messages` from state
- Calls model with `research_brief_prompt` + `.with_structured_output(ResearchBrief)`
- Returns `{"research_brief": brief_as_string}`

**Verify**: Call with a sample state containing a user message. Confirm it returns a well-formed brief.

---

### Step 10 — Node: researcher (single-pass)

The researcher node — searches and accumulates notes. No loop yet (that's Increment 2).

**Files created**:
```
src/deep_research/nodes/researcher.py
```

**What it contains**:
```python
async def researcher(state: AgentState, config: RunnableConfig) -> dict:
```
- Reads `research_brief` from state
- Invokes model with `research_system_prompt` + search tools bound via `.bind_tools()`
- Executes tool calls (search), collects results
- Returns `{"notes": formatted_findings}`

In this increment, the researcher is a single ReAct-style pass: model calls search tool(s), gets results, model produces notes. No reflection loop, no compression — just one round of search → synthesize.

**Verify**: Call with a state containing a research brief. Confirm it searches and returns notes.

---

### Step 11 — Node: final_report

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

No token-limit retry yet — that's Increment 5.

**Verify**: Call with sample brief + notes. Confirm it produces a structured markdown report.

---

### Step 12 — Graph wiring

Wire all three nodes into a LangGraph and expose it.

**Files created**:
```
src/deep_research/graph.py
```

**What it contains**:
```python
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("write_brief", write_research_brief)
    graph.add_node("researcher", researcher)
    graph.add_node("final_report", final_report_generation)
    graph.add_edge(START, "write_brief")
    graph.add_edge("write_brief", "researcher")
    graph.add_edge("researcher", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile()
```

Linear pipeline — no conditional edges, no subgraphs. Those come in later increments.

**Verify**: `build_graph()` compiles without error.

---

### Step 13 — End-to-end integration test

Run the full pipeline with a real question.

**Files created**:
```
tests/test_e2e.py           # or just a script: src/deep_research/__main__.py
```

**What it does**:
- Creates graph via `build_graph()`
- Invokes with a simple question (e.g., "What are the main causes of coral reef bleaching?")
- Asserts: `final_report` is non-empty, contains markdown structure, references sources
- Print the report for manual review

**Verify**: Full pipeline runs, produces a coherent researched report. Review output quality manually.

---

### Step 14 — utils.py re-export facade

Public API surface for external consumers.

**Files created**:
```
src/deep_research/utils.py
```

**What it contains**:
- Re-exports from internal modules:
  - `get_model` from `helpers.model`
  - `TavilySearchTool`, `tavily_search` from `tools.search.tavily`
  - `BaseSearchTool` from `tools.search.base`
  - `build_graph` from `graph`
- Internal code never imports from here — this is for external users only

**Verify**: `from deep_research.utils import build_graph, get_model` works.

---

## File tree after Increment 1

```
deep_research/
    pyproject.toml
    DESIGN_AND_PLAN.md
    DESIGN_REFERENCE.md
    plans/
        increment_1_minimal_e2e.md
    src/deep_research/
        __init__.py
        graph.py
        state.py
        configuration.py
        prompts.py
        models.py
        utils.py
        nodes/
            __init__.py
            brief.py
            researcher.py
            report.py
        tools/
            __init__.py
            search/
                __init__.py
                base.py
                tavily.py
        helpers/
            __init__.py
            model.py
    tests/
        test_e2e.py
```

## What this increment does NOT include (deferred)
- Researcher reflection loop (Increment 2)
- Research compression (Increment 2)
- Supervisor / multi-topic (Increment 3)
- Clarification node (Increment 4)
- Multi-provider model support (Increment 4)
- Token-limit handling (Increment 5)
- Dead-end / contradiction detection (Increment 5)
- CLI interface (Increment 6)
