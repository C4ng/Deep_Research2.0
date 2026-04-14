# Deep Research — Design & Implementation Plan
*Consolidated from DESIGN_REFERENCE.md analysis + open_deep_research_reference study*

## Architectural Decisions (Locked In)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **LangGraph** | Supervisor-researcher maps naturally to subgraphs; get checkpointing/streaming free |
| Primary LLM | **Gemini** | First target. Architecture must support swapping to Claude/OpenAI later via clean model abstraction |
| Search API | **Tavily** | First target. Must be swappable — define a base search interface, Tavily inherits from it |
| Tool calling | **LangChain tools** | Native LangGraph integration, `.bind_tools()` / `@tool` decorator |
| State design | **Grow incrementally** | Start with minimal AgentState, extend per increment as needs emerge |
| Project structure | **`src/` layout, modular from day 1** | Separate files per node, separate interface from implementation. Designed to grow. |
| Interface | **Python API first** | CLI layer added later on top |
| MCP | **Discarded** | Not needed for this project |

## Design Patterns Borrowed (from DESIGN_REFERENCE.md & reference code)

### Built into architecture from the start
- **Hierarchical subgraphs**: Main graph → supervisor subgraph → researcher subgraph (from open_deep_research)
- **Context isolation**: Each researcher receives only its assigned topic, not full conversation history
- **Research compression**: Dedicated compression step before returning findings to supervisor
- **Webpage summarization**: Structured extraction (key_info, relevant_excerpts, source_url) using cheaper model

### Reflection strategy (hybrid approach)
- **Researcher level → Structured reflection** (Pydantic schema). Enables programmatic routing, dead-end detection, contradiction tracking. The inner loop is tight (3-5 iterations) and needs system-controlled decisions.
- **Supervisor level → Free-form reasoning**. Higher-level strategic decisions ("assign more researchers?") are less amenable to fixed schema. Let the LLM reason naturally before choosing tools.

### Built in later increments
- **Dead-end detection** (DESIGN_REFERENCE Section 6): Compare `missing_info` across reflection rounds — if unchanged, trigger reformulated query from different angle. Key differentiator.
- **Contradiction resolution**: `contradictions` field non-empty → next queries target resolution
- **Supervisor low-confidence handling**: Researcher returns with low confidence → supervisor reassigns or accepts partial
- **Light review pass on report**: Check claims against sources, flag unsupported (simplified from gpt-researcher)
- **URL authority scoring**: +weight for .gov/.edu sources (from foreveryh), lightweight quality signal
- **Two-tier notes**: Raw + compressed (from open_deep_research), adds debugging visibility
- **Token-limit retry with progressive truncation**: Detect token errors, truncate by 10%, retry

### Explicitly skipped
- Constraint extraction (LearningCircuit) — too specialized for open-ended research
- Full quality pipeline: editor → reviewer → revisor → writer (gpt-researcher) — too heavy
- MCP integration
- Middleware chain (deer-flow) — overkill

## Modular / Swappable Design Points

These components have clean interfaces from the start, even if only one implementation exists initially.

### 1. Model provider abstraction
```python
get_model(provider: str, model_name: str, **kwargs) -> BaseChatModel
```
- Increment 1: Gemini only
- Later: add Claude, OpenAI by extending provider enum + adding API key routing

### 2. Search tool interface
```python
class BaseSearchTool(ABC):
    async def search(self, queries: list[str], topic: str) -> list[SearchResult]: ...

class TavilySearchTool(BaseSearchTool):
    async def search(self, queries: list[str], topic: str) -> list[SearchResult]: ...
```
- Increment 1: Tavily implementation
- Later: Brave, SearXNG, etc. Selected via configuration

### 3. Configuration
- Pydantic `Configuration` class, loaded from env vars
- Model selection, search API selection, iteration limits — all configurable
- Start with hardcoded defaults, expose configuration gradually

## Structured Reflection Design

Used at the **researcher level** for programmatic routing. Replaces the reference's stateless `think_tool`.

```python
class Reflection(BaseModel):
    key_findings: list[str]      # what we learned this round
    missing_info: list[str]      # gaps still remaining
    contradictions: list[str]    # conflicting information found
    confidence: float            # 0.0 - 1.0
    should_continue: bool        # model's recommendation
    next_queries: list[str]      # what to search next if continuing
```

**Routing logic** (conditional edge):
- `should_continue=False` OR `confidence >= 0.8` OR `iterations >= max` → exit to compress
- `missing_info` unchanged from prior round → dead-end detected → reformulate (Increment 5)
- `contradictions` non-empty → next queries target resolution (Increment 5)

**Why not `think_tool`**: `think_tool` is a no-op that returns `"Reflection recorded: {text}"`. The system never inspects the content — routing depends entirely on which tool the LLM picks next. Structured reflection gives us typed fields to route on, compare across rounds, and log/debug.

## State Design (Incremental)

### Increment 1-2: Minimal
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_brief: str
    notes: str           # accumulated research findings
    final_report: str
```

### Increment 3+: Add supervisor/researcher separation
```python
class ResearcherState(TypedDict):
    messages: Annotated[list, add_messages]
    research_topic: str
    iterations: int
    findings: str        # compressed output from this researcher

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]  # supervisor's own message history
    research_iterations: int
    notes: list[str]     # collected findings from all researchers
```

### Later: Extend AgentState
```python
    raw_notes: list[str]     # two-tier notes (uncompressed)
    notes: list[str]         # compressed notes
```

## Project Structure

Modular from the start. Each node is its own file. Interface separated from implementation.

### Increment 1
```
src/deep_research/
    __init__.py
    graph.py               # main graph wiring
    state.py               # AgentState
    configuration.py       # minimal config (model, API keys from env)
    prompts.py             # prompt templates
    models.py              # Pydantic schemas (SearchResult, ResearchBrief, etc.)
    utils.py               # public re-export facade (like reference's utils.py)
    nodes/
        __init__.py
        brief.py           # write_research_brief
        researcher.py      # researcher node (helpers live here, not in utils)
        report.py          # final_report_generation
    tools/
        __init__.py
        search/
            __init__.py
            base.py        # BaseSearchTool ABC + SearchResult schema
            tavily.py      # TavilySearchTool(BaseSearchTool)
    helpers/
        __init__.py
        model.py           # get_model() — model provider abstraction
```

**Import pattern** (following reference): Internal code imports directly from where things live (`helpers.model`, `tools.search.tavily`). `utils.py` is a re-export facade for external consumers only. Helper functions stay in the module that uses them — no dumping ground.

### Increment 2 — adds
```
    models.py              # + Reflection schema
    nodes/
        compress.py        # compress_research node
```

### Increment 3 — adds
```
    state.py               # + ResearcherState, SupervisorState
    nodes/
        supervisor.py      # supervisor node + supervisor_tools
    graph.py               # restructured: main graph + researcher subgraph + supervisor subgraph
```

### Increment 4 — adds
```
    nodes/
        clarify.py         # clarify_with_user node
    models.py              # + ClarifyOutput; ResearchBrief simplified (single question + is_simple)
    prompts.py             # + clarify_prompt; research_brief_prompt rewritten; coordinator_system_prompt updated
    state.py               # + is_simple to AgentState
    graph/
        graph.py           # + clarify node, conditional routing (simple → researcher, else → coordinator)
```

### Increment 5 — adds
```
    nodes/
        review.py          # light review pass
    helpers/
        tokens.py          # token-limit detection
```

### Increment 6 — adds
```
    tools/
        search/
            brave.py       # BraveSearchTool(BaseSearchTool)
    helpers/
        config.py          # API key routing per provider
    cli.py                 # CLI interface
```

## Incremental Implementation Plan

### Increment 1 — Minimal End-to-End
**Goal**: Question in → researched markdown report out.

**Graph**: `write_brief` → `researcher` → `final_report`
- Single researcher, single search pass (no loop yet)
- Tavily search: send queries, get results, summarize webpages
- Minimal `AgentState` (messages, research_brief, notes, final_report)
- Gemini model, hardcoded config from env vars
- No supervisor, no reflection, no compression

**Delivers**: Proof that plumbing works — LangGraph, Gemini, Tavily all wired together.

### Increment 2 — Researcher Reflection Loop
**Goal**: Researcher iterates based on what it learns.

- Add `Reflection` Pydantic schema (structured output)
- Researcher becomes a loop: search → reflect → (continue or exit)
- Conditional edge: route on `should_continue` + `confidence` + iteration count
- Add `compress_research` node before returning
- Iteration limit as safety net (default: 5)

**Delivers**: The core value-add — iterative deepening based on identified gaps.

### Increment 3 — Supervisor + Multi-Topic Decomposition
**Goal**: Complex questions decomposed and researched across topics.

- Supervisor node with `ConductResearch` / `ResearchComplete` tools
- Free-form reasoning at supervisor level (not structured reflection)
- Researcher becomes a subgraph invoked per topic
- Context isolation: each researcher gets only its assigned topic
- Sequential execution first, parallel (`asyncio.gather`) later
- Supervisor reviews returned research, can delegate follow-up rounds
- State extended with `ResearcherState` and `SupervisorState`

**Delivers**: Multi-faceted research capability.

### Increment 4 — Question Stage (Clarification + Scoping)
**Goal**: Well-formed research questions through user interaction, with adaptive routing.

- `clarify_with_user` node: resolve ambiguity before research begins (optional, config-gated)
- Brief reform: single well-articulated research question (drop premature decomposition into subtopics — that's the coordinator's job)
- Simple question routing: `is_simple` flag bypasses coordinator, goes direct to single researcher
- Coordinator prompt update: reason deliberately about research strategy before decomposing (no fixed type enum — the coordinator thinks freely about approach)

**Delivers**: Adaptive system — simple questions get fast answers, complex questions get full multi-topic research, coordinator reasons about strategy instead of always decomposing into 5 topics.

### Increment 5 — Dead-End Handling + Quality (Differentiators)
**Goal**: The features from DESIGN_REFERENCE.md Section 6 that set us apart.

- Dead-end detection: compare `missing_info` across reflection rounds → reformulate
- Contradiction resolution: non-empty `contradictions` → targeted follow-up searches
- Coordinator handles low-confidence researcher returns (reassign or accept partial)
- Light review pass on final report (claims vs sources check)
- Token-limit retry with progressive truncation

**Delivers**: Robust, quality-aware research.

### Increment 6 — Extensions + Provider Flexibility
- Multi-provider model support (Claude, OpenAI) + config resolver
- API key routing per provider
- Second search provider: `BraveSearchTool(BaseSearchTool)`
- URL authority scoring on search results
- Two-tier notes (raw + compressed)
- CLI interface (`cli.py`)
