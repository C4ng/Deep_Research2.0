# Deep Research

An agentic deep research system built on [LangGraph](https://github.com/langchain-ai/langgraph). Given a research question, it decomposes the topic, dispatches parallel researchers with iterative search-and-reflect loops, resolves contradictions across sources, and produces a cited final report.

Built iteratively with [Claude Code](https://claude.ai/claude-code) across 10 increments — from a minimal single-researcher prototype to a full multi-agent system with a web UI. Each increment's plan is in `plans/`.

📺 [**Watch the demo**](docs/demo.mp4) — the system researching a question end-to-end, from clarification through parallel research to the final cited report.

## Features

- **Human-in-the-loop scoping** — optional clarification questions and research brief review before research begins
- **Two-level research architecture** — a coordinator decomposes complex questions into focused subtopics, each investigated by an independent researcher with its own search-reflect loop
- **Iterative reflection** — researchers assess their own progress after each search round, identifying findings, gaps, and contradictions to guide follow-up queries
- **Contradiction handling** — researchers reason about why sources disagree (temporal variation, scope mismatch, precision, etc.), resolve what they can, and only flag genuinely unresolved conflicts
- **Dead-end detection** — when gaps persist unfilled across rounds, the system reformulates queries or exits gracefully instead of looping
- **Citation tracking** — each source gets a short hash ID (e.g., `[a1b2c3d4]`) that researchers reference throughout their notes. At report time, these are automatically replaced with numbered references `[1]`, `[2]`, etc. and a Sources section is appended
- **Source deduplication** — same URL produces the same source ID across all researchers and rounds, skipping redundant summarization
- **Structured final report** — metadata-driven report generation with contradiction analysis, coverage assessment, and honest gap disclosure
- **Modular providers** — swap search APIs and LLM providers via configuration strings, no code changes
- **Web UI** — three-column React interface (activity log, research state, chat) with real-time SSE streaming and resizable panels
- **Fail-fast on errors** — tool failures and empty results trigger early exit with LLM knowledge fallback instead of wasting iterations

## Quick Start

### Prerequisites

- Python 3.11+
- A search API key (Tavily recommended)
- An LLM API key (Google Gemini by default)

### Installation

```bash
git clone https://github.com/C4ng/Deep_Research2.0.git
cd Deep_Research2.0

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core package
pip install -e .

# For additional LLM providers
pip install -e ".[all-providers]"

# For the web UI
pip install -e ".[web]"
cd web/frontend && npm install && cd ../..
```

### Environment

```bash
# .env
GOOGLE_API_KEY=your-key          # Default LLM provider
TAVILY_API_KEY=your-key          # Default search provider

# Optional: additional providers
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
BRAVE_API_KEY=your-key
SERPER_API_KEY=your-key

# Optional: LangSmith tracing (traces every LLM call, tool call, and node execution)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=deep-research
```

### Running

**CLI:**
```bash
python scripts/run.py "What are the key differences between agent frameworks?"
```

**Web UI:**
```bash
python web/dev.py
# Backend:  http://localhost:8000
# Frontend: http://localhost:5173
```

**LangGraph Studio:**
```bash
# Open langgraph.json in LangGraph Studio for visual debugging
```

## Supported Providers

### Search APIs

| Provider | Content Quality | Setup |
|----------|----------------|-------|
| **Tavily** (default) | Full page text via `raw_content` | `TAVILY_API_KEY` |
| **Brave** | Snippets + extra snippets | `BRAVE_API_KEY` |
| **Serper** | Google search snippets only | `SERPER_API_KEY` |

### LLM Providers

Any provider supported by LangChain's `init_chat_model`:

| Provider | Prefix | API Key Env Var |
|----------|--------|-----------------|
| Google Gemini | `google_genai:` | `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic:` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:` | `OPENAI_API_KEY` |

The system uses two model roles:
- **Research model** — reasoning, tool calling, reflection, report writing (default: `gemini-2.5-flash`)
- **Summarization model** — webpage compression, mechanical extraction (default: `gemini-2.5-flash-lite`)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `research_model` | `google_genai:gemini-2.5-flash` | Model for research tasks |
| `summarization_model` | `google_genai:gemini-2.5-flash-lite` | Model for webpage summarization |
| `search_api` | `tavily` | Search provider |
| `max_search_results` | 5 | Results per search query |
| `max_searches_per_round` | 3 | Search calls per research round |
| `max_research_iterations` | 3 | Max search-reflect cycles per researcher |
| `max_research_topics` | 5 | Max subtopics the coordinator decomposes into |
| `max_coordinator_iterations` | 2 | Max coordinator follow-up rounds |
| `allow_clarification` | `true` | Ask clarifying questions before research |
| `allow_human_review` | `true` | Pause for user review of research brief |

Configuration resolves from: environment variables > runtime config > defaults.

## Project Structure

```
src/deep_research/
    configuration.py         # Config schema, SearchAPI enum, defaults
    models.py                # Pydantic schemas (ResearchBrief, ResearchReflection, etc.)
    prompts.py               # All prompt templates
    state.py                 # State TypedDicts with reducers

    graph/
        graph.py             # Main graph: clarify -> brief -> coordinator -> report
        model.py             # Provider-aware model factory

    nodes/
        clarify.py           # Optional clarification questions
        brief.py             # Research brief generation and review
        report.py            # Final report synthesis with citation resolution

        coordinator/
            coordinator.py   # Topic decomposition + researcher dispatch
            reflect.py       # Cross-topic completeness assessment
            tools.py         # dispatch_research tool (runs researcher subgraph)

        researcher/
            researcher.py    # LLM call with search tools bound
            reflect.py       # Per-round reflection with dead-end detection
            summarizer.py    # Compress raw results into concise notes

    tools/
        registry.py          # Dynamic tool assembly
        search/
            base.py          # BaseSearchTool ABC with shared summarization
            tavily.py        # Tavily provider
            brave.py         # Brave provider
            serper.py        # Serper provider

    helpers/
        errors.py            # Error detection helpers
        source_store.py      # Citation infrastructure (source store, dedup, resolution)

web/
    api.py                   # FastAPI backend with SSE streaming
    event_mapper.py          # Graph events -> UI events (chat/activity/state)
    dev.py                   # Dev server (backend + frontend)
    frontend/src/
        App.tsx              # Three-column resizable layout
        components/          # ChatPanel, ActivityPanel, StatePanel, Splitter
        hooks/               # SSE connection and state accumulation
        types.ts             # TypeScript interfaces

tests/                       # Unit, integration, search, citation tests
plans/                       # Increment plans (1-10) with design rationale
```

---

# Design Document

## Why This Architecture

A naive approach — search once, summarize results — misses depth. It can't follow up on gaps, cross-reference conflicting sources, or decide when it has enough information. A single-agent loop improves on this but struggles with complex questions that span multiple angles.

This system uses a **two-level coordinator-researcher architecture** because research has two distinct cognitive tasks:

1. **Decomposition** — breaking "compare agent frameworks" into specific, searchable subtopics. This requires understanding the full question and deciding what angles matter.
2. **Investigation** — deep-diving into each subtopic with iterative search-and-reflect cycles. This requires focus on a single topic without distraction from the broader question.

Separating these into coordinator and researcher roles means each can be optimized independently. Researchers don't need to track the overall research plan — they focus on their assigned topic. The coordinator doesn't need to search — it reads structured results and decides what's missing.

The iterative reflection loop is what makes this a "deep" search rather than a scraper. After each search round, the researcher produces a structured assessment: what did I learn, what's still missing, do my sources contradict each other, and are follow-up searches likely to help? This drives targeted follow-up queries instead of redundant broad searches. Dead-end detection prevents wasted iterations when gaps turn out to be unsearchable. Contradiction resolution goes beyond listing conflicts — the system reasons about *why* sources disagree and resolves temporal, scope, and precision differences before flagging genuine conflicts for the final report.

## Data Flow

```
User Query
    |
    v
+------------------+
|     Clarify      |  Optional: ask clarifying questions if query is ambiguous
+------------------+  Routes to __end__ for user input, or proceeds
    |
    v
+------------------+
|   Write Brief    |  Generate structured research plan (title, question, approach)
+------------------+  User reviews and approves before research begins
    |
    v
+=============================================+
|              COORDINATOR                     |
|  Decomposes brief into 3-5 focused subtopics |
|  Dispatches parallel researchers              |
|                                              |
|  +----------+  +----------+  +----------+   |
|  |Researcher|  |Researcher|  |Researcher|   |
|  |          |  |          |  |          |   |
|  | search   |  | search   |  | search   |   |
|  |   |      |  |   |      |  |   |      |   |
|  | reflect  |  | reflect  |  | reflect  |   |
|  |   |      |  |   |      |  |   |      |   |
|  | search   |  | summarize|  | search   |   |
|  |   |      |  |          |  |   |      |   |
|  | reflect  |  +----------+  | reflect  |   |
|  |   |      |                |   |      |   |
|  | summarize|                | summarize|   |
|  +----------+                +----------+   |
|                                              |
|  Collect results, assess cross-topic coverage |
|  If gaps remain -> dispatch follow-up round   |
+=============================================+
    |
    v
+------------------+
|  Final Report    |  Synthesize findings, resolve citations [id] -> [N]
+------------------+  Analyze contradictions, disclose gaps
    |
    v
  Cited Report
```

### Researcher Loop (per subtopic)

Each researcher runs an independent search-reflect cycle:

```
Round 1: LLM generates search queries
            |
         Execute searches (parallel)
            |
         Reflect: what did we learn? what's missing?
            |
         knowledge_state = "partial", 2 gaps remaining
            |
Round 2: LLM generates targeted follow-up queries
            |
         Execute searches
            |
         Reflect: gaps filled, knowledge_state = "sufficient"
            |
         Summarize: compress raw results into concise notes
            |
         Return ResearchResult to coordinator
```

Stop conditions: `knowledge_state == "sufficient"`, `should_continue == false`, max iterations reached, or dead-end detected (gaps persist unfilled across rounds).

### Coordinator Reflection

After collecting all researcher results, the coordinator assesses cross-topic completeness:
- Are there coverage gaps in the original brief?
- Do different researchers contradict each other?
- Is follow-up research worth the cost?

If significant gaps remain, it dispatches a targeted follow-up round. Otherwise, it merges all notes and metadata for the final report.

## Design Decisions

### Human-in-the-Loop: Question Scoping

The system uses LangGraph's `Command(goto="__end__")` pattern instead of `interrupt()`. User-facing nodes (clarify, write_brief) route themselves to `__end__` when user input is needed. On resume, a conditional router (`_route_start`) skips already-completed stages. This keeps the graph structure simple and stateless between invocations.

The clarification step prevents wasted research on ambiguous queries. The brief review step gives users control over the research strategy — they can adjust scope, angle, or emphasis — without micromanaging individual searches.

### Research Brief: Shaping the Search Strategy

The brief isn't just a summary of the user's question — it's strategic guidance for the coordinator: what kind of research this needs, whether to prioritize breadth or depth, which angles matter most. The coordinator reads this to decide how to decompose the topic. The brief uses a structured `list[str]` for approach points, ensuring clean formatting regardless of LLM output variance.

### Two-Level Architecture: Coordinator and Researchers

Complex questions need both decomposition and depth. The coordinator handles decomposition (breaking "compare agent frameworks" into specific subtopics like "LangChain architecture", "AutoGen multi-agent patterns"), while each researcher handles depth (iterative search-reflect loops within its subtopic). Researchers run as isolated subgraphs with their own state — they don't need to know about each other. The coordinator handles cross-topic synthesis.

### Topic Decomposition

The coordinator LLM decomposes the research brief into up to `max_research_topics` (default: 5) focused subtopics using the `dispatch_research` tool. Each tool call specifies a topic and context explaining the angle to investigate. Researchers run concurrently via `asyncio.gather`, and results are collected as structured `ResearchResult` objects with metadata (findings, gaps, contradictions, knowledge state).

### Structured Reflection and Knowledge Accumulation

Reflection uses Pydantic structured output (`ResearchReflection`) rather than free-form text. This gives the system programmatic access to:
- `key_findings` — accumulated across rounds via an append reducer (knowledge grows monotonically)
- `contradictions` — overwritten each round (LLM outputs the full canonical list)
- `current_gaps` — overwritten each round (latest assessment only)
- `knowledge_state` — drives routing decisions (`insufficient` / `partial` / `sufficient` / `unavailable`)
- `prior_gaps_filled` — enables dead-end detection (0 gaps filled across rounds triggers reformulation)

### Contradiction Resolution

Rather than simply listing conflicting sources, the system follows a three-stage pipeline:

1. **Researcher reflection** — before flagging a contradiction, the LLM reasons about *why* sources disagree. Common resolvable patterns:
   - *Temporal variation*: same metric from different dates — prefer the most recent, note the date
   - *Scope mismatch*: sources describing different things under the same name — label each specifically
   - *Precision difference*: approximate vs exact figures — keep the most precise value
   - *Different aspects*: statements about different dimensions — present as complementary findings
   
   If a conflict can't be resolved from existing data, a targeted `next_query` is added to search for a current authoritative source.

2. **Coordinator reflection** — identifies cross-topic contradictions between different researchers, applying the same resolution logic.

3. **Final report** — presents remaining genuinely unresolved contradictions with both sides cited, analysis of why they may differ, and guidance on which claim has stronger support.

### Stop Conditions

The system decides when to stop at two levels:

**Researcher level:** stop when `knowledge_state == "sufficient"`, `should_continue == false`, max iterations reached, or dead-end detected. Dead-end detection: if `prior_gaps_filled == 0` for two consecutive rounds, the system first injects reformulation guidance (try synonyms, different angles). If gaps persist after reformulation, it forces exit.

**Coordinator level:** stop when the coordinator reflection reports `knowledge_state` as `sufficient` or `unavailable`, `should_continue == false`, or max coordinator iterations reached. The coordinator won't dispatch follow-up research for minor gaps — only significant coverage holes justify another round.

### Final Report Structure

The report node receives:
- **Combined notes** — merged researcher findings with topic headers
- **Report metadata** — per-topic coverage, contradictions, gaps + cross-topic signals from coordinator reflection
- **Source map** — `{source_id: {url, title}}` for citation resolution

The LLM generates a structured markdown report using the metadata to surface contradictions (under `### Conflicting Evidence` subheadings), disclose gaps honestly, and note which topics relied on LLM knowledge rather than live search. After generation, `resolve_citations()` programmatically replaces `[source_id]` hex tags with sequential `[N]` references and appends a deterministic Sources section. Unknown IDs become `[unverified]`.

### Citation Tracking System

Source IDs are 8-character hex strings from `MD5(url)[:8]`. These survive LLM summarization far better than full URLs. The pipeline:

1. Search tool generates source ID, writes source file to disk (first-write-wins dedup)
2. Tool results include `[source_id]` tags inline with content
3. Researchers reference these tags in findings and contradictions
4. Summarizer preserves tags through compression
5. At report time, `resolve_citations()` replaces hex IDs with sequential `[N]` and appends Sources
6. Unknown IDs become `[unverified]` — the reader sees which claims lost their citation chain

Cross-researcher dedup: same URL produces the same source ID regardless of which researcher found it, so the same source is summarized only once across the entire research session.

### Context Engineering

Each node sees only what it needs — not the full state:
- Researchers get their topic + reflection guidance, not the conversation history
- The coordinator gets research metadata (findings, gaps, states) but not raw notes
- The final report gets full notes + metadata + source map
- Prior round messages are not carried forward — reflection captures key knowledge into state accumulators

This keeps LLM context windows focused and reduces token costs.

### Modular Provider Architecture

**LLM providers** use LangChain's `init_chat_model` with configurable fields. Swap providers by changing a string:
```
google_genai:gemini-2.5-flash → anthropic:claude-sonnet-4-20250514 → openai:gpt-4.1
```

A unified `thinking_budget` parameter maps automatically to each provider's native format (Gemini's `thinking_budget`, Anthropic's `thinking`, OpenAI's `reasoning`). Provider-specific quirks (e.g., OpenAI o-series models don't support `temperature`) are handled in `build_model_config()`.

**Search providers** inherit from `BaseSearchTool` and register in a simple map. Adding a new provider: implement `search()`, add one entry to `_SEARCH_TOOL_MAP`. Shared logic (summarization, source store writes, result formatting) lives in the base class.

## Development Process

Built incrementally with **Claude Code** as the primary development tool across 10 increments. Each increment added a specific capability, with a plan document written before implementation:

1. Minimal end-to-end (single researcher, one search, report)
2. Researcher reflection loop (iterative search-reflect cycles)
3. Coordinator + multi-topic decomposition
4. Question stage (clarification + research brief review)
5. Citation system (source store, ID tracking, dedup)
6. Final report redesign (metadata-driven, programmatic citation resolution)
7. Research quality (dead-end detection, contradiction dedup, `[unverified]` marking)
8. Provider flexibility (multi-provider search/LLM, API key routing)
9. Testing and fixes (fail-fast on errors, knowledge state validation)
10. Web UI (three-column React + FastAPI with SSE streaming)

Plans are in `plans/` — they capture the reasoning behind each architectural decision.

## Further Improvements

This is a working prototype. Areas for further development:

**Prompt Calibration**
- The current prompts work but haven't been systematically tuned against a benchmark dataset
- Reflection prompts could be calibrated for different research domains (technical, market, academic)
- The contradiction resolution heuristics could be evaluated against ground-truth contradiction sets

**Domain-Specific Search**
- Different research domains benefit from different search strategies — academic research could use Semantic Scholar or arXiv APIs, market research could use financial data APIs
- The `BaseSearchTool` abstraction supports this: implement a new provider and register it
- Search tool selection could be dynamic based on the research brief's domain

**Error Handling and Resilience**
- The prototype has fail-fast on tool errors and LLM knowledge fallback, but doesn't retry transient failures
- Rate limiting across concurrent researchers could be more sophisticated (currently relies on provider-side limits)
- No circuit breaker pattern for persistent API failures across sessions

**Streaming and UX**
- Researcher internals run via `ainvoke` (not `astream`), so token-level streaming isn't available for intermediate steps — only log-level events stream in real-time
- The web UI state panel could show a timeline view of the research process
- Report generation could stream progressively instead of appearing all at once

**Evaluation**
- No automated quality metrics for research output yet
- Could add LLM-as-judge evaluation on report completeness, citation accuracy, and contradiction handling
- A/B testing different prompt strategies requires instrumentation not yet in place

**Scale**
- The in-memory checkpointer (`MemorySaver`) doesn't persist across server restarts — production use needs a persistent store
- Source files are written to a temp directory — production needs durable storage
- Concurrent research sessions share a single graph instance — works for prototype, needs session isolation at scale
