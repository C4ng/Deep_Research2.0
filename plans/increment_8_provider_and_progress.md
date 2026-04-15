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

**Verified API response format** (from live test call):
```json
{
  "web": {
    "results": [
      {
        "title": "LangChain overview - Docs by LangChain",
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "description": "Different providers have unique APIs...",
        "extra_snippets": [
          "LangChain · LangGraph · Integrations...",
          "If you don't need these capabilities...",
          "Different providers have unique APIs...",
          "Join us May 13th & May 14th..."
        ]
      }
    ]
  }
}
```
Note: `description` may contain `<strong>` HTML tags. `extra_snippets` returns
up to 4-5 entries per result (not always 5). Some snippets are navigation/
boilerplate — the summarization LLM will filter those naturally.

#### 0a. Dependency: use `httpx` (already available via `tavily-python`)

`tavily-python` depends on `httpx`, so it's already in our environment. Use
raw `httpx.AsyncClient` for Brave API calls — avoids adding a new dependency
for a simple REST API (one endpoint, one GET request).

No `pyproject.toml` change needed.

#### 0b. `BraveSearchTool(BaseSearchTool)` in `brave.py`

Only implements `search()` — same pattern as `TavilySearchTool`:

```python
class BraveSearchTool(BaseSearchTool):
    def __init__(self, api_key: str, config: RunnableConfig | None = None):
        super().__init__(config=config)
        self._api_key = api_key

    async def search(self, queries: list[str], *, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            # Fan out queries concurrently (same as Tavily)
            tasks = [
                client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results, "extra_snippets": "true"},
                    headers={
                        "X-Subscription-Token": self._api_key,
                        "Accept": "application/json",
                    },
                )
                for query in queries
            ]
            responses = await asyncio.gather(*tasks)

        # Dedup by URL across queries (same pattern as Tavily)
        seen_urls: dict[str, SearchResult] = {}
        for response in responses:
            response.raise_for_status()
            data = response.json()
            for result in data.get("web", {}).get("results", []):
                url = result["url"]
                if url in seen_urls:
                    continue

                description = result.get("description", "")
                extra_snippets = result.get("extra_snippets", [])

                # Strip HTML tags from description (Brave returns <strong> etc)
                clean_description = _strip_html(description)

                # Concatenate description + extra_snippets for raw_content
                if extra_snippets:
                    raw_content = clean_description + "\n\n" + "\n\n".join(extra_snippets)
                else:
                    raw_content = None  # fallback: skip summarization, use snippet

                seen_urls[url] = SearchResult(
                    url=url,
                    title=result.get("title", ""),
                    content=clean_description,
                    raw_content=raw_content,
                )
        return list(seen_urls.values())
```

Helper: `_strip_html()` — simple regex to remove `<strong>`, `<em>`, etc.
from Brave's description field. Not a full HTML parser, just tag stripping.

#### 0c. `@tool` wrapper function (same pattern as `tavily_search`)

```python
BRAVE_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

@tool(description=BRAVE_SEARCH_DESCRIPTION)
async def brave_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    configurable = Configuration.from_runnable_config(config)
    search_tool = BraveSearchTool(
        api_key=configurable.brave_api_key, config=config
    )
    return await search_tool.search_and_summarize(
        queries, max_results=max_results
    )
```

#### 0d. Configuration: add `brave_api_key` + `SearchAPI.BRAVE`

In `configuration.py`:
```python
class SearchAPI(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"

# In Configuration class:
brave_api_key: str = Field(
    default="",
    description="Brave Search API key (loaded from BRAVE_SEARCH_API_KEY env var)",
)
```

No other config changes — `from_runnable_config()` already handles env var
resolution via `os.environ.get(field_name.upper())`, so `BRAVE_API_KEY` will
be picked up automatically.

#### 0e. Registry: route by SearchAPI enum

In `registry.py`:
```python
from deep_research.tools.search.brave import brave_search

async def get_search_tools(config: RunnableConfig | None = None) -> list:
    configurable = Configuration.from_runnable_config(config)
    if configurable.search_api == SearchAPI.TAVILY:
        return [tavily_search]
    if configurable.search_api == SearchAPI.BRAVE:
        return [brave_search]
    return []
```

#### 0f. Tests

**Unit** (`tests/test_search_providers.py`):
- `test_brave_search_maps_results` — mock `httpx.AsyncClient.get` with a
  canned Brave response → verify `SearchResult` fields (url, title, content,
  raw_content with concatenated snippets)
- `test_brave_search_strips_html` — description with `<strong>` tags →
  content has clean text
- `test_brave_search_dedup_urls` — two queries returning overlapping URLs →
  deduplicated by URL
- `test_brave_search_no_extra_snippets` — result without `extra_snippets` →
  `raw_content=None`, `content` = description

**Integration** (`tests/test_search_providers.py`, `@pytest.mark.integration`):
- `test_brave_search_and_summarize` — real Brave API call → verify formatted
  output has `[source_id]` tags, summaries are generated from concatenated
  snippets

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
