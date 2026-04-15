# Increment 10 — Web UI

**Goal**: Three-column web interface for the research pipeline — chat,
activity feed, and state viewer. Replace the CLI runner with a visual
experience.

**Depends on**: Increment 9 (Testing & Fixes)

---

## Architecture

```
Browser (React)                    Server (FastAPI)
+--------------------+----------+     +-------------------+
| Chat               | Activity |<--->| SSE stream        |
| - user input       | - dispatch    | | POST /run         |
| - brief review     |   researchers | | POST /resume      |
| - final report     | - search      | | GET  /stream/{id} |
|                    |   queries     | +-------------------+
|                    | - reflection  |        |
|                    |   results     |   build_graph()
|                    | - compression |   astream_events()
+--------------------+----------+
```

**Backend**: FastAPI app wrapping `build_graph()` with `astream_events`
streaming. Categorizes events into two channels (chat, activity) and
sends them as SSE to the frontend.

**Frontend**: Vite + React + Tailwind. Two-panel layout with an SSE
listener that routes events to the correct column.

---

## Step 1 — Backend: FastAPI streaming server

**File**: `web/api.py`

Thin FastAPI app with three endpoints:

```python
POST /api/run          # Start a new research session
POST /api/resume       # Resume after HITL pause (send user message)
GET  /api/stream/{id}  # SSE stream of events for a session
```

Core logic:

```python
async def _run_and_stream(graph, input_state, config, thread_id):
    """Run the graph and yield SSE events categorized by channel."""
    async for event in graph.astream_events(input_state, config, version="v2"):
        kind = event["event"]
        
        # Chat channel: AI messages to the user (clarification, brief, report)
        if kind == "on_chain_end" and event["name"] in ("clarify", "write_brief", "final_report"):
            yield sse("chat", extract_ai_message(event))
        
        # Activity channel: node starts, tool calls, log messages
        if kind == "on_chain_start":
            yield sse("activity", {"node": event["name"], "status": "started"})
        if kind == "on_tool_start":
            yield sse("activity", {"tool": event["name"], "input": event["data"]["input"]})
        
        # State channel: state snapshots after each node
        if kind == "on_chain_end":
            yield sse("state", extract_state_snapshot(event))
```

Key decisions:
- `astream_events(version="v2")` gives fine-grained events (node start/end,
  tool start/end, LLM tokens). We filter and categorize.
- `MemorySaver` checkpointer for session persistence (same as CLI runner).
- HITL: when the graph exits to `__end__` mid-pipeline (brief review),
  the stream sends a `{"channel": "chat", "type": "needs_input"}` event.
  The frontend shows the input box. On submit, the frontend POSTs to
  `/api/resume` which re-invokes the graph and opens a new SSE stream.
- Thread ID = session ID. One thread per research session.

**File**: `web/__init__.py` — empty, makes it a package.

**Dependencies**: `fastapi`, `uvicorn`, `sse-starlette` added to
`pyproject.toml` under `[project.optional-dependencies] web = [...]`.

### Event categorization

Map graph events to the three UI channels:

| Graph event | Channel | UI display |
|-------------|---------|------------|
| clarify AI message | chat | Show clarification question |
| write_brief AI message | chat | Show brief for review |
| final_report complete | chat | Render markdown report |
| coordinator dispatching researchers | activity | "Dispatching 5 researchers..." |
| dispatch_research tool call | activity | "Researching: {topic}" |
| researcher tool_call (search) | activity | "Searching: {queries}" |
| reflect result | activity | "Reflection: {knowledge_state}" |
| summarize complete | activity | "Compressed findings" |
| coordinator_reflect result | activity | "Coordinator: {assessment}" |

### HITL flow

The graph uses `Command(goto="__end__")` for HITL pauses. The SSE stream
detects this and sends a special event:

```json
{"channel": "chat", "type": "needs_input", "data": {"prompt": "Review the brief above..."}}
```

The frontend shows the input box. On submit, it calls `POST /api/resume`
with the user's message, which calls `graph.ainvoke()` with the new
HumanMessage and opens a new SSE stream.

---

## Step 2 — Frontend: three-column React app

**Directory**: `web/frontend/`

Scaffold with Vite + React + TypeScript + Tailwind:

```
web/frontend/
  src/
    App.tsx              # Two-column layout
    components/
      ChatPanel.tsx      # Left: chat messages + input
      ActivityPanel.tsx  # Right: live activity feed
    hooks/
      useEventStream.ts  # SSE connection + event routing
    types.ts             # Event type definitions
  index.html
  package.json
  vite.config.ts
  tailwind.config.js
```

### ChatPanel (left column, ~55%)

- Message list: user messages (right-aligned), AI messages (left-aligned)
- Markdown rendering for AI messages (brief, report) — use `react-markdown`
  with `remark-gfm` for tables/headings
- Input box at the bottom — disabled during research, enabled during HITL
- Final report gets special styling (full-width, proper typography)

### ActivityPanel (right column, ~45%)

- Scrolling feed of activity events, newest at bottom
- Each event is a compact card:
  - "Dispatching 5 researchers" with topic list
  - "Searching: solid-state battery electrolyte innovations 2025"
  - "Reflection round 1: knowledge_state=partial, 3 gaps"
  - "Compressed 4441 chars → 834 chars (19%)"
- Color-coded by type (dispatch=blue, search=green, reflection=yellow)
- Auto-scroll to bottom as events arrive

### Layout

```
+-------------------------+----------------------+
|       Chat (55%)        |   Activity (45%)     |
|                         |                      |
| [User message]          | Dispatching 5        |
|                         |  researchers...      |
| [AI: Here's the brief]  |                      |
|                         | Searching:           |
|                         |  "solid-state..."    |
|                         |                      |
|                         | Reflection R1:       |
|                         |  partial, 3 gaps     |
|                         |                      |
| [input box]             | Coordinator: done    |
+-------------------------+----------------------+
```

Responsive: on narrow screens, activity panel collapses to a toggleable
drawer or moves below the chat.

---

## Step 3 — Wire streaming events

The trickiest part: mapping `astream_events` output to clean UI events.

**File**: `web/api.py` — event mapper functions

`astream_events(version="v2")` produces events like:

```python
{"event": "on_chain_start", "name": "coordinator", "data": {...}}
{"event": "on_tool_start", "name": "dispatch_research", "data": {"input": {"topic": "..."}}}
{"event": "on_chain_end", "name": "reflect", "data": {"output": {...}}}
```

The mapper extracts user-friendly messages:

```python
def map_event(event) -> dict | None:
    """Map a raw astream_events event to a UI event, or None to skip."""
    kind, name = event["event"], event["name"]
    
    if kind == "on_tool_start" and name == "dispatch_research":
        return {"channel": "activity", "type": "dispatch",
                "data": {"topic": event["data"]["input"]["topic"]}}
    
    if kind == "on_tool_start" and name in SEARCH_TOOL_NAMES:
        return {"channel": "activity", "type": "search",
                "data": {"queries": event["data"]["input"]["queries"]}}
    
    # ... etc
    return None  # skip events we don't care about
```

Most events are skipped — we only surface ~10 event types out of hundreds.

---

## Step 4 — Dev setup and scripts

**File**: `web/dev.py` — convenience script to run both backend and frontend

```python
# Runs: uvicorn web.api:app + cd web/frontend && npm run dev
# With CORS configured for local dev
```

**File**: `pyproject.toml` — add `web` optional dependency group:

```toml
[project.optional-dependencies]
web = ["fastapi>=0.115", "uvicorn>=0.34", "sse-starlette>=2.0"]
```

**File**: `web/frontend/vite.config.ts` — proxy `/api` to FastAPI backend.

---

## Step 5 — Tests

**File**: `tests/test_web_api.py`

1. `test_run_creates_session` — POST /api/run returns session ID
2. `test_stream_emits_events` — GET /api/stream/{id} produces SSE events
   (mock the graph to emit a known sequence)
3. `test_resume_sends_message` — POST /api/resume adds HumanMessage to state
4. `test_event_categorization` — verify map_event categorizes correctly

No frontend tests initially — visual verification.

---

## Risks and decisions

- **`astream_events` verbosity**: produces hundreds of events per run.
  The mapper must be selective. Start with too few events and add more,
  rather than flooding the UI.
- **Subgraph events**: researcher subgraph events are nested. Need to
  check if `astream_events` surfaces them or if we need
  `include_names`/`include_tags` filters.
- **HITL timing**: the SSE stream ends when the graph exits to `__end__`.
  The frontend needs to detect this and switch to "waiting for input" mode.
  On resume, a new SSE stream opens.
- **Concurrent sessions**: MemorySaver is per-process. Fine for single-user
  dev. Production would need a persistent checkpointer.
