"""FastAPI backend for the deep research web UI.

Wraps the LangGraph research pipeline with SSE streaming,
categorizing events into chat and activity channels.
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from deep_research.graph.graph import build_graph

from web.event_mapper import map_stream_event

logger = logging.getLogger(__name__)

# Global graph instance with checkpointer
_checkpointer = MemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


# --- Activity log capture ---
# We capture log messages from the research pipeline and forward them
# as activity events to the frontend via SSE.

class ActivityCollector(logging.Handler):
    """Logging handler that collects activity-relevant log messages."""

    def __init__(self):
        super().__init__()
        self.events: asyncio.Queue = asyncio.Queue()

    def emit(self, record: logging.LogRecord):
        # Only capture INFO+ from our pipeline modules
        if record.levelno < logging.INFO:
            return
        if not record.name.startswith("deep_research"):
            return
        try:
            self.events.put_nowait({
                "channel": "activity",
                "type": "log",
                "data": {
                    "module": record.name.split(".")[-1],
                    "message": record.getMessage(),
                    "level": record.levelname.lower(),
                },
            })
        except asyncio.QueueFull:
            pass


# --- Request/response models ---

class RunRequest(BaseModel):
    query: str


class ResumeRequest(BaseModel):
    thread_id: str
    message: str


# --- App setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Deep Research", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return HTMLResponse("<h3>Deep Research API</h3><p>Use the frontend at port 5173</p>")


@app.post("/api/run")
async def run(request: RunRequest):
    """Start a new research session. Returns thread_id for streaming."""
    thread_id = str(uuid.uuid4())
    return {"thread_id": thread_id, "query": request.query}


@app.get("/api/stream/{thread_id}")
async def stream(thread_id: str, query: str | None = None, resume_message: str | None = None):
    """SSE stream for a research session.

    On first call: pass ?query=... to start research.
    On resume: pass ?resume_message=... to continue after HITL pause.
    """
    return EventSourceResponse(
        _generate_events(thread_id, query=query, resume_message=resume_message),
        media_type="text/event-stream",
    )


@app.post("/api/resume")
async def resume(request: ResumeRequest):
    """Resume after HITL pause. Returns the same thread_id for re-streaming."""
    return {"thread_id": request.thread_id, "message": request.message}


async def _generate_events(thread_id: str, query: str | None = None, resume_message: str | None = None):
    """Run the graph and yield SSE events."""
    config = {"configurable": {"thread_id": thread_id}}

    # Set up activity log capture
    collector = ActivityCollector()
    root_logger = logging.getLogger("deep_research")
    root_logger.addHandler(collector)

    try:
        # Build input
        if query:
            input_state = {"messages": [HumanMessage(content=query)]}
        elif resume_message:
            input_state = {"messages": [HumanMessage(content=resume_message)]}
        else:
            return

        # Track AI message contents sent during streaming to avoid duplicates
        sent_contents: set[str] = set()

        # Stream graph execution
        yield _sse({"channel": "activity", "type": "status", "data": {"status": "running"}})

        async for chunk in _graph.astream(input_state, config, stream_mode="updates"):
            # Each chunk is {node_name: state_update}
            for node_name, update in chunk.items():
                # Map node updates to UI events
                events = map_stream_event(node_name, update)
                for event in events:
                    if event.get("channel") == "chat" and event.get("type") == "ai_message":
                        sent_contents.add(event["data"].get("content", ""))
                    yield _sse(event)

            # Drain any activity logs that accumulated
            while not collector.events.empty():
                try:
                    log_event = collector.events.get_nowait()
                    yield _sse(log_event)
                except asyncio.QueueEmpty:
                    break

        # After graph completes, check final state for messages not yet sent
        final_state = await _graph.aget_state(config)
        values = final_state.values if final_state else {}

        all_messages = values.get("messages", [])
        for msg in all_messages:
            if isinstance(msg, AIMessage) and msg.content and msg.content not in sent_contents:
                yield _sse({
                    "channel": "chat",
                    "type": "ai_message",
                    "data": {"content": msg.content},
                })
                sent_contents.add(msg.content)

        # Check if we have a final report
        final_report = values.get("final_report", "")
        if final_report:
            yield _sse({
                "channel": "chat",
                "type": "report",
                "data": {"content": final_report},
            })
            yield _sse({"channel": "activity", "type": "status", "data": {"status": "complete"}})
        else:
            # Graph paused for HITL — needs user input
            yield _sse({
                "channel": "chat",
                "type": "needs_input",
                "data": {"prompt": "Waiting for your response..."},
            })
            yield _sse({"channel": "activity", "type": "status", "data": {"status": "waiting"}})

    finally:
        root_logger.removeHandler(collector)


def _sse(data: dict) -> dict:
    """Format an event for SSE."""
    return {"data": json.dumps(data)}
