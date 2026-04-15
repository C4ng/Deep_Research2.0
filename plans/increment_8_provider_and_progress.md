# Increment 8 — Provider Flexibility + Progress Interface

**Goal**: Add a second search provider to prove the abstraction, multi-provider
model API key routing, and a progress-streaming interface so users see what's
happening during research.
**Status**: Planning
**Depends on**: Increment 7 (Research Quality)

## Overview

Three deliverables, scoped to what adds real value:

1. **Second search provider** — add one provider (e.g., Brave) to prove
   `BaseSearchTool` works and handle the raw_content gap (not all providers
   return full page text).
2. **Model API key routing** — `init_chat_model` already handles multi-provider
   models. The gap is routing the right API key per provider from config/env.
3. **Progress streaming interface** — use `astream(stream_mode="updates")` to
   surface research progress: brief ready, researchers dispatched, searches
   happening, reflections, report writing.

**What we're NOT doing** (scoped out with rationale):
- **Provider-native search (Anthropic, OpenAI, Gemini)** — conflicts with our
  pipeline architecture. Our source store, `[source_id]` tagging, citation
  resolution, `[unverified]` marking, and all prompt field criteria depend on
  controlling search results. Native search bypasses all of this — the model
  consumes results internally. Would require a fundamentally different
  abstraction (response post-processor), not a `BaseSearchTool` subclass.
  See "Research: Search Provider Selection" above for full analysis.
- **Serper (Google Search API)** — snippet-only (single Google excerpt per
  result), no standalone Python client (only `langchain-community` wrappers).
  Brave is strictly better: more content via `extra_snippets`, independent
  client library.
- **Two-tier notes (raw + compressed)** — no use case. LangSmith traces already
  show full messages for debugging. Storing raw notes separately adds state
  bloat with no consumer.
- **Token-limit retry with progressive truncation** — speculative. We haven't
  hit token limits: summarizer truncates raw content at 50k chars, reflection
  loop limits rounds to 3. Add reactively if we observe failures, not
  preemptively. The reference implementation's hardcoded `MODEL_TOKEN_LIMITS`
  dict goes stale immediately and requires manual maintenance per model.
- **Full CLI (`cli.py`)** — the progress streaming interface is the real user
  need. A traditional CLI wrapper around `graph.ainvoke()` adds little value
  over a Python script. If deployment is needed, LangGraph Platform
  (`langgraph.json`) is the standard path.

---

## Research: Search Provider Selection

### Providers evaluated

**External search APIs** (fit our pipeline — return raw results we control):

| Provider | Content per result | raw_content? | Python client |
|----------|-------------------|-------------|---------------|
| **Tavily** (current) | snippet + full page text | Yes (`include_raw_content=True`) | `tavily-python` (direct) |
| **Brave Web Search** | snippet (`description`) + up to 5 `extra_snippets` | No — excerpts only | `brave-search-python-client` or raw HTTP |
| **Serper** (Google) | Google snippet only | No | `langchain-community` only |

**Provider-native search** (model searches internally — evaluated and rejected):

| Provider | Tool | How it works |
|----------|------|-------------|
| Anthropic | `web_search_20250305` | Claude searches server-side, results in response |
| OpenAI | `web_search_preview` | GPT searches server-side, results in `tool_outputs` |
| Google Gemini | `GoogleSearch()` grounding | Gemini searches Google, returns `groundingMetadata` |

### Why Brave over Serper

- Brave returns `description` + up to 5 `extra_snippets` per result — ~6 excerpts
  total. Serper returns only Google's single snippet. More content = better input
  for our summarization pipeline.
- Brave has a standalone Python client (`brave-search-python-client`) and raw HTTP
  API. Serper only has `langchain-community` wrappers (we don't use that package).
- Both lack full page text (`raw_content`), but Brave's extra_snippets partially
  compensate.

### Why not provider-native search

Native search conflicts with our architecture:
- **No source IDs**: our pipeline generates `[source_id]` tags, writes to source
  store, and resolves citations in the final report. Native search bypasses all
  of this — the model consumes results internally, we never see raw results.
- **Prompt dependencies**: `researcher_reflection_prompt` expects `[source_id]`
  tagged findings; `compress_research_prompt` preserves `[source_id]` tags;
  `final_report_prompt` cites via `[source_id]`. All built around our pipeline.
- **No control**: we can't enforce `max_searches_per_round`, can't dedup across
  rounds, can't summarize differently. The model decides everything.
- **Would require a different abstraction**: not a `BaseSearchTool` subclass but a
  response post-processor that reverse-engineers source IDs from grounding metadata.
  Fragile and fights the architecture.

Native search could be a future increment with a separate design if needed.

### Brave API details

- **Endpoint**: `GET https://api.search.brave.com/res/v1/web/search`
- **Auth**: `X-Subscription-Token: <API_KEY>` header
- **Key params**: `q` (query), `count` (max 20), `extra_snippets=true`
- **Response**: `{ web: { results: [{ title, url, description, extra_snippets: [...] }] } }`
- **Free tier**: 2000 requests/month
- **No raw_content**: `extra_snippets` are additional excerpts, not full page text

### raw_content strategy for Brave

Concatenate `description` + `extra_snippets` into `raw_content`. This gives the
summarization LLM ~6 excerpts to work with — enough to produce a structured
summary, though less rich than Tavily's full page text. When `extra_snippets` is
not available, fall back to snippet-only (skip summarization, use `description`
as content).

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Which search provider | **Brave** | Best content depth after Tavily (extra_snippets), standalone Python client, no langchain-community dependency. Serper is snippet-only. Native search conflicts with our pipeline architecture. |
| raw_content gap | **Concatenate extra_snippets** | Combine `description` + `extra_snippets` → treat as `raw_content` for summarization. Falls back to snippet-only when extra_snippets unavailable. |
| API key routing | **Env var convention + config fallback** | Follow LangChain convention: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` from env. Search API keys in Configuration. |
| Progress interface | **`astream` + event mapper** | `astream(stream_mode="updates")` emits `{node_name: update}` per node completion. Map node names to user-friendly progress messages. |
| Deployment | **`langgraph.json` for LangGraph Platform** | Standard LangGraph deployment format. Enables LangGraph Studio web UI for free. |

---

## Design

### Search Provider Abstraction (already in place)

```
BaseSearchTool (ABC)
├── search()           — abstract, provider-specific API call
├── summarize_content() — shared, uses configured summarization model
├── search_and_summarize() — shared, orchestrates search → summarize → format
└── _format_results()  — shared, formats with [source_id] tags

TavilySearchTool(BaseSearchTool)  — existing (tavily-python client)
BraveSearchTool(BaseSearchTool)   — new (raw HTTP or brave-search-python-client)
```

New provider only implements `search()`. The `raw_content` strategy per provider:
- **Tavily**: `include_raw_content=True` → full page text → summarization LLM
- **Brave**: `description` + `extra_snippets` concatenated → ~6 excerpts →
  summarization LLM condenses into structured summary. When `extra_snippets`
  unavailable, `raw_content=None` → skip summarization, use snippet directly.
- **Fallback** (base class, no changes needed): when `raw_content` is None,
  `search_and_summarize` skips the summarization LLM call and uses
  `result.content` (snippet) directly.

### Progress Streaming

```
astream(stream_mode="updates") emits per node:

  clarify          → "Checking if clarification needed..."
  write_brief      → "Research plan ready: {title}"
  coordinator      → (subgraph — internal events not surfaced by default)
  researcher       → (subgraph — internal events not surfaced by default)
  final_report     → "Writing final report..."

For subgraph visibility, use astream_events(version="v2") which emits:
  on_chain_start   → node entering (includes node name)
  on_tool_start    → tool calls (search queries)
  on_chain_end     → node output (reflection results, summaries)
```

Progress event types to surface:
- `brief_ready`: research plan title and approach
- `researcher_dispatched`: topic being researched
- `search_executing`: queries being run
- `reflection_complete`: round N, knowledge_state, gaps remaining
- `researcher_complete`: topic done, knowledge_state
- `report_writing`: final report generation started
- `report_complete`: done, report length

### API Key Routing

```python
# Configuration already has tavily_api_key.
# Add search-provider-specific keys:
brave_api_key: str = Field(default="", ...)

# Model API keys: handled by init_chat_model + env vars.
# LangChain convention: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
# No code needed — init_chat_model reads these automatically.
```

---

## Implementation Steps

### Step 0 — Brave search provider ✍️

**Files**: `src/deep_research/tools/search/brave.py` (new),
`src/deep_research/tools/registry.py`, `src/deep_research/configuration.py`,
`pyproject.toml`

**Provider decision**: Brave (settled — see Research section above).

1. Add `brave-search-python-client` to `pyproject.toml` optional dependencies
   (or use raw `httpx` — evaluate which is cleaner)

2. Implement `BraveSearchTool(BaseSearchTool)`:
   - Only `search()` method — calls Brave Web Search API
   - Request: `GET /res/v1/web/search?q={query}&count={max_results}&extra_snippets=true`
   - Auth: `X-Subscription-Token` header with API key
   - Map response to `SearchResult`:
     - `url` = `result.url`
     - `title` = `result.title`
     - `content` = `result.description` (snippet)
     - `raw_content` = `description` + `"\n\n"` + `"\n\n".join(extra_snippets)`
       (concatenated excerpts for summarization; `None` if no extra_snippets)
   - Dedup across queries by URL (same pattern as Tavily)

3. Create the `@tool` wrapper function (same pattern as `tavily_search`):
   ```python
   @tool(description=BRAVE_SEARCH_DESCRIPTION)
   async def brave_search(queries, max_results, config):
       configurable = Configuration.from_runnable_config(config)
       search_tool = BraveSearchTool(api_key=configurable.brave_api_key, config=config)
       return await search_tool.search_and_summarize(queries, max_results=max_results)
   ```

4. Update registry to route based on `SearchAPI` enum:
   ```python
   class SearchAPI(Enum):
       TAVILY = "tavily"
       BRAVE = "brave"     # new
   ```

5. Add `brave_api_key` to Configuration with `BRAVE_SEARCH_API_KEY` env var

**Tests**:
- Unit: `BraveSearchTool.search()` with mocked HTTP responses — verify
  `SearchResult` mapping, extra_snippets concatenation, URL dedup
- Integration: `search_and_summarize()` with real Brave API — verify formatting,
  source IDs, summarization of concatenated snippets

---

### Step 1 — Model API key routing

**Files**: `src/deep_research/configuration.py`, `src/deep_research/graph/model.py`

Currently `init_chat_model` auto-reads env vars for model API keys. The gap:
when users want to pass keys via config (e.g., LangGraph Platform deployment).

1. Add `api_key` resolution to `Configuration.from_runnable_config()`:
   - Check env var by provider convention (already works for most cases)
   - Allow override via `configurable.api_key` for deployment scenarios

2. Update nodes that call `configurable_model.with_config()` to pass `api_key`
   from the resolved config (currently only `model`, `max_tokens`, `temperature`,
   `thinking_budget` are passed).

3. Verify provider swapping works end-to-end:
   - `google_genai:gemini-2.5-flash` (current default)
   - `anthropic:claude-sonnet-4-20250514`
   - `openai:gpt-4.1-mini`

**Tests**:
- Unit: Configuration resolves correct API key per provider prefix
- Integration: Run brief generation with a non-default provider

---

### Step 2 — Progress streaming interface

**Files**: `src/deep_research/progress.py` (new),
`src/deep_research/graph/graph.py`

1. Define progress event types:
   ```python
   @dataclass
   class ProgressEvent:
       stage: str        # "brief", "research", "report"
       message: str      # user-friendly description
       detail: dict      # structured data (topic, round, knowledge_state, etc.)
   ```

2. Implement `run_with_progress()` — async generator that wraps
   `graph.astream_events(version="v2")` and yields `ProgressEvent`s:
   - Map `on_chain_start/end` events by node name to progress messages
   - Extract structured data from node outputs (brief title, researcher
     topic, reflection round, knowledge_state)
   - Filter noise — only surface events the user cares about

3. Add a simple console runner that consumes the generator:
   ```python
   async for event in run_with_progress(query, config):
       print(f"[{event.stage}] {event.message}")
   ```

4. Add `langgraph.json` for LangGraph Platform deployment (enables
   LangGraph Studio web UI).

**Tests**:
- Unit: event mapper produces correct ProgressEvent for each node type
- Integration: full pipeline streams expected event sequence

---

### Step 3 — Integration testing + polish

Run full end-to-end tests:

1. **Search provider swap**: Run same query with Tavily and Brave, compare
   report quality (does snippet-only fallback produce acceptable results?)
2. **Model provider swap**: Run brief generation with Gemini, Anthropic,
   OpenAI — verify structured output works across providers
3. **Progress streaming**: Verify complete event sequence for both simple
   (single researcher) and complex (coordinator) queries
4. **Graceful degradation**: Missing API key for non-default provider shows
   clear error, not a cryptic traceback

---

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Add brave client dependency (optional or httpx) |
| `src/deep_research/tools/search/brave.py` | BraveSearchTool implementation — `search()` with extra_snippets concatenation |
| `src/deep_research/tools/registry.py` | Route search tools by SearchAPI enum (TAVILY / BRAVE) |
| `src/deep_research/configuration.py` | Add `brave_api_key`, `SearchAPI.BRAVE` enum value |
| `src/deep_research/progress.py` | Progress event types + astream wrapper |
| `src/deep_research/graph/graph.py` | Expose builder for langgraph.json |
| `langgraph.json` | LangGraph Platform deployment config |
| `tests/test_search_providers.py` | Brave search unit tests (mocked HTTP) + integration tests |
| `tests/test_progress.py` | Progress streaming tests |

## Not incorporated (rationale in Research section and scoping above)

- **Provider-native search** — conflicts with pipeline architecture (source store, citation tracking)
- **Serper** — snippet-only, no standalone client, Brave is strictly better
- **Two-tier notes** — no consumer, LangSmith covers debugging
- **Token-limit retry** — not observed as a problem, add reactively
- **Full CLI** — progress interface covers the real user need
