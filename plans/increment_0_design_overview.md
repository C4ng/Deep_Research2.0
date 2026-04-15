# Deep Research — Design Overview

Initial architectural decisions and design patterns that guided the incremental build. Each increment has its own detailed plan:

1. [Increment 1 — Minimal End-to-End](increment_1_minimal_e2e.md)
2. [Increment 2 — Researcher Reflection Loop](increment_2_reflection_loop.md)
3. [Increment 3 — Coordinator + Multi-Topic](increment_3_coordinator.md)
4. [Increment 4 — Question Stage (Clarification + Scoping)](increment_4_question_stage.md)
5. [Increment 5 — Citation System](increment_5_citations.md)
6. [Increment 6 — Final Report Redesign](increment_6_final_report.md)
7. [Increment 7 — Research Quality](increment_7_research_quality.md)
8. [Increment 8 — Provider Flexibility + Progress](increment_8_provider_and_progress.md)
9. [Increment 9 — Testing & Fixes](increment_9_testing_fixes.md)
10. [Increment 10 — Web UI](increment_10_web_ui.md)

## Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | LangGraph | Supervisor-researcher maps naturally to subgraphs; checkpointing/streaming built-in |
| LLM | Gemini (default) | Swappable to Claude/OpenAI via configuration |
| Search | Tavily (default) | Swappable to Brave/Serper via configuration |
| Tool calling | LangChain tools | Native LangGraph integration |
| Reflection | Structured (Pydantic) at researcher level | Enables programmatic routing, dead-end detection, contradiction tracking |
| State | Grown incrementally | Started minimal, extended per increment as needs emerged |

## Core Design Patterns

- **Hierarchical subgraphs**: Main graph → coordinator → researcher subgraphs
- **Context isolation**: Each researcher receives only its assigned topic
- **Structured reflection**: Typed fields for routing decisions, not free-form text
- **Research compression**: Dedicated step before returning findings to coordinator
- **Dead-end detection**: Persistent gaps across rounds trigger reformulation or exit
- **Contradiction resolution**: Reason about why sources disagree before flagging
