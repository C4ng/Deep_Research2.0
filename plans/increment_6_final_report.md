# Increment 6 — Final Report Redesign
**Goal**: A report node that uses full research metadata and produces verifiable, programmatically resolved citations.
**Status**: Complete
**Depends on**: Increment 5 (Citation System) — stable `[sN]` IDs and source store on disk

## Overview

The report node currently receives only `research_brief` (string) and `notes` (string). All intermediate metadata — key_findings, contradictions, gaps, knowledge_state — is discarded at the coordinator/adapter boundary. Citations are improvised by the LLM with no source map. The Increment 5 source store and `[sN]` IDs exist but the report node doesn't use them.

This increment:
1. Passes research metadata (contradictions, gaps, knowledge_state) to the report
2. Gives the report LLM the source map so it can reference sources accurately
3. Programmatically resolves `[sN]` tags to clean `[N] Title: URL` citations
4. Surfaces contradictions and limitations honestly in the report

---

## Current State (Problems)

```
Report node sees:
  research_brief: "Title: ...\nQuestion: ...\nApproach: ..."  (string)
  notes: "## Topic: X\n{compressed notes with [sN] tags}\n---\n..."  (string)

Report node does NOT see:
  key_findings       — per-researcher (discarded at coordinator exit)
  contradictions     — per-researcher + cross-topic (discarded)
  coverage_gaps      — coordinator's identified blind spots (discarded)
  knowledge_state    — per-topic completeness (discarded)
  source store       — all source URLs with content (not wired)

Citation prompt (current):
  "Assign each unique URL a sequential citation number [1], [2], [3]..."
  → The LLM invents or guesses URLs. No source map. No verification.
```

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Metadata format | **Formatted string** (`report_metadata`) | LLM consumes this in a prompt. Multiple list fields on AgentState for data only the report uses is unnecessary. Format at exit, pass as one string. |
| What metadata to pass | **Coordinator reflection + per-researcher contradictions & knowledge_state** | Coordinator reflection synthesizes cross-topic issues. Per-researcher key_findings are redundant with notes (already prioritized by summarizer). Per-researcher contradictions and knowledge_state add topic-level attribution the coordinator doesn't capture. |
| Citation resolution | **Programmatic post-processing** | LLM writes report with `[sN]` tags (familiar from notes). After generation, regex replaces `[sN]` → `[N]` and appends a deterministic Sources section. No hallucinated URLs. |
| Citation format in output | **Sequential `[N]` with footnotes** | `[a1b2c3d4]` is useful during research but reader-unfriendly. Post-processing maps to `[1]`, `[2]`, etc. with a Sources section at the end. |
| `[id1, id2]` handling | **Parse comma-separated, output as `[N, M]`** | Observed LLM behavior from Increment 5 testing — LLMs prefer comma-separated in single bracket. Resolver handles both formats. |
| Post-generation verification | **Yes, lightweight** | Regex + dict lookup. Log warnings for unresolvable IDs. Don't block report generation — remove unresolvable tags gracefully. |
| Report structure | **Prescribe contradictions + further research sections; rest is flexible** | Different questions need different body shapes. But contradictions and research gaps should always be surfaced explicitly when they exist. |

---

## Design

### Metadata Flow

**Coordinator path** (complex questions):
```
ResearchResult per researcher:
  key_findings, contradictions, knowledge_state, missing_info
  ↓
coordinator_reflect on exit:
  CoordinatorReflection: overall_assessment, cross_topic_contradictions,
                         coverage_gaps, knowledge_state
  ↓
_format_report_metadata() builds string from both:
  - Per-topic: contradictions, knowledge_state, missing_info (persistent gaps)
  - Cross-topic: contradictions, coverage_gaps, overall_assessment
  ↓
AgentState.report_metadata (new field)
```

**Simple path** (single researcher):
```
ResearcherState on exit:
  accumulated_findings, accumulated_contradictions, current_gaps,
  final_knowledge_state
  ↓
run_single_researcher() formats into report_metadata
  ↓
AgentState.report_metadata
```

### Gap Types in Metadata

The metadata distinguishes three types of incomplete coverage. The report
prompt instructs the LLM to present each differently under "Areas for
Further Research":

| Gap type | Source | Meaning | How to present |
|----------|--------|---------|----------------|
| **Coverage gaps** | `CoordinatorReflection.coverage_gaps` | Topic not investigated at all — no researcher was assigned | "X was not investigated in this research" |
| **Persistent gaps** | `ResearchResult.missing_info` | Researcher searched but info not found in available sources | "Information on Y was sought but not found in available sources" |
| **Partial coverage** | `ResearchResult.knowledge_state == "partial"` | Topic researched but major angles still missing | "Z was covered at a high level; deeper analysis of [aspect] may be valuable" |

Coverage gaps are scope limitations (we chose not to look). Persistent
gaps are search limitations (we looked but couldn't find). Partial
coverage is depth limitations (we found some but not enough). These carry
different implications for the reader who wants to follow up.

### Citation Resolution Pipeline

```
LLM generates report with [sN] tags (8-char hex IDs from notes)
  ↓
resolve_citations(report_text, source_map):
  1. Regex finds all [sN] references (both [id] and [id1, id2] formats)
  2. Collect unique cited source IDs, ordered by first appearance
  3. Assign sequential numbers: a1b2c3d4 → 1, e5f6a7b8 → 2, ...
  4. Replace in text: [a1b2c3d4] → [1], [a1b2c3d4, e5f6a7b8] → [1, 2]
  5. Remove any existing Sources/References section the LLM may have written
  6. Append deterministic Sources section:
       ## Sources
       [1] Quantum Computing Market 2025: https://example.com/quantum-report
       [2] IBM Press Release: https://ibm.com/quantum
  7. Log warnings for any source IDs not found in store (but don't crash)
```

### Report Prompt Redesign

The prompt receives:
- `brief` — the research plan (existing)
- `notes` — compressed findings with `[sN]` references (existing)
- `report_metadata` — contradictions, gaps, knowledge_state (new)
- `source_map` — formatted `[sN] Title — URL` lookup table (new)

Key prompt instructions:
1. Write the main body from `notes`, citing sources with their `[sN]` tags
2. When contradictions exist: include a "Conflicting Evidence" section presenting both sides with source IDs — do not silently pick one side
3. When the metadata contains gaps: include an "Areas for Further Research" section — the metadata labels indicate why each gap exists, let the LLM reason about how to present them
4. Use knowledge_state signals to calibrate confidence language (sufficient → assertive, partial → hedged)
5. Do NOT write a Sources section — it will be added programmatically
6. Do NOT invent source IDs — only use `[sN]` tags that appear in the notes

---

## Implementation Steps

### Step 0 — Add `report_metadata` to AgentState + CoordinatorState

**File**: `src/deep_research/state.py`

Added `report_metadata: str` to both `AgentState` and `CoordinatorState`.
CoordinatorState needs the field so LangGraph propagates it from subgraph
output to parent state via matching key names (verified working).

```python
class AgentState(TypedDict):
    # ... existing fields ...
    report_metadata: str
    """Formatted research metadata for the report: contradictions, gaps, knowledge states."""
```

Default to `""` — report node handles empty gracefully.
Also updated `INITIAL_STATE` in `tests/test_integration.py`.

### Step 1 — Format metadata at exit points

Two places produce `report_metadata`:

**File**: `src/deep_research/nodes/coordinator/reflect.py`

Add `_format_report_metadata()` that builds a string from:
- Per-researcher: topic, knowledge_state, contradictions (from `ResearchResult`)
- Coordinator reflection: overall_assessment, cross_topic_contradictions, coverage_gaps

Called at coordinator exit (in the `should_stop` branch), written to state.

The coordinator exit currently writes to `CoordinatorState` which maps to `AgentState` — need to verify that `report_metadata` propagates through the subgraph boundary. If CoordinatorState → AgentState mapping drops unknown keys, add `report_metadata` to CoordinatorState as well.

```python
def _format_report_metadata(
    results: list[ResearchResult],
    reflection: CoordinatorReflection,
) -> str:
    parts = []

    # Per-topic signals
    for r in results:
        section = [f"### {r.topic}"]
        section.append(f"Coverage: {r.knowledge_state}")
        if r.contradictions:
            section.append("Contradictions:")
            section.extend(f"- {c}" for c in r.contradictions)
        if r.missing_info:
            section.append("Persistent gaps (searched but not found):")
            section.extend(f"- {g}" for g in r.missing_info)
        parts.append("\n".join(section))

    # Cross-topic signals from coordinator reflection
    if reflection.cross_topic_contradictions:
        parts.append("\n### Cross-Topic Contradictions")
        parts.extend(f"- {c}" for c in reflection.cross_topic_contradictions)

    if reflection.coverage_gaps:
        parts.append("\n### Coverage Gaps (not investigated)")
        parts.extend(f"- {g}" for g in reflection.coverage_gaps)

    parts.append(f"\n### Overall Assessment\n{reflection.overall_assessment}")

    return "\n\n".join(parts)
```

**File**: `src/deep_research/nodes/researcher/adapter.py`

For simple questions, format from the single researcher's state:
```python
async def run_single_researcher(state, config) -> dict:
    result = await researcher_subgraph.ainvoke(initial_state, config)

    metadata_parts = []
    knowledge = result.get("final_knowledge_state", "")
    if knowledge:
        metadata_parts.append(f"Coverage: {knowledge}")
    if result.get("accumulated_contradictions"):
        metadata_parts.append("### Contradictions")
        metadata_parts.extend(f"- {c}" for c in result["accumulated_contradictions"])
    if result.get("current_gaps"):
        metadata_parts.append("### Persistent Gaps (searched but not found)")
        metadata_parts.extend(f"- {g}" for g in result["current_gaps"])

    return {
        "notes": result["notes"],
        "report_metadata": "\n".join(metadata_parts),
    }
```

### Step 2 — Citation resolution helpers

**File**: `src/deep_research/helpers/source_store.py` (extend existing)

Add two functions:

```python
def resolve_citations(report: str, source_map: dict[str, dict]) -> tuple[str, list[str]]:
    """Replace [source_id] tags with sequential [N] and append Sources section.

    Handles both [id] and [id1, id2] formats.

    Returns (resolved_report, warnings).
    Warnings list contains any source IDs referenced but not in store.
    """

def format_source_map_for_prompt(source_map: dict[str, dict]) -> str:
    """Format source map as a lookup table for the report prompt.

    Output:
        [a1b2c3d4] Quantum Computing Market 2025 — https://example.com/...
        [e5f6a7b8] IBM Press Release — https://ibm.com/...
    """
```

**`resolve_citations` detail:**
1. Regex pattern: `\[([0-9a-f]{8}(?:\s*,\s*[0-9a-f]{8})*)\]`
   - Matches `[a1b2c3d4]` and `[a1b2c3d4, e5f6a7b8]` and `[id1, id2, id3]`
2. First pass: collect all unique source IDs, ordered by first appearance
3. Build mapping: `{source_id: sequential_number}`
4. Second pass: replace each match with sequential numbers
   - `[a1b2c3d4]` → `[1]`
   - `[a1b2c3d4, e5f6a7b8]` → `[1, 2]`
   - Unknown IDs → removed from the bracket (logged as warning)
5. Strip any LLM-generated Sources/References section (regex for `## Sources` or `## References` at end of report)
6. Append deterministic Sources section:
   ```
   ## Sources

   [1] Quantum Computing Market 2025: https://example.com/quantum-report
   [2] IBM Press Release: https://ibm.com/quantum
   ```
7. Return `(resolved_text, warning_list)`

### Step 3 — Redesign report prompt

**File**: `src/deep_research/prompts.py`

Replace `final_report_prompt`:

```python
final_report_prompt = """\
Create a comprehensive research report based on the findings below.

<research_brief>
{brief}
</research_brief>

<findings>
{notes}
</findings>

<research_metadata>
{report_metadata}
</research_metadata>

<source_map>
{source_map}
</source_map>

Today's date is {date}.

<instructions>
1. Structure with markdown headings (# title, ## sections, ### subsections).
2. Write substantive sections — each should contain specific facts, data, and
   analysis from the findings, not brief mentions.
3. Cite sources inline using their [source_id] tags exactly as they appear in
   the findings (e.g., [a1b2c3d4]). Do NOT invent source IDs — only use IDs
   that appear in the findings or source map.
4. When research_metadata lists contradictions: include a "Conflicting Evidence"
   section presenting both sides with their source IDs. Do not silently pick
   one side.
5. When research_metadata contains gaps, include an "Areas for Further Research"
   section. The metadata labels indicate why each gap exists (not investigated,
   searched but not found, partial coverage) — use these signals to explain
   what remains unknown and why.
6. Use the coverage/knowledge_state signals in research_metadata to calibrate
   confidence in the main body. Topics with "partial" coverage should use
   hedging language. Topics with "sufficient" coverage can be more assertive.
7. Do NOT write a Sources or References section — it will be added
   programmatically after generation.
8. Write in the same language as the original user query.
</instructions>

Do not refer to yourself or comment on the writing process."""
```

### Step 4 — Update report node

**File**: `src/deep_research/nodes/report.py`

Changes:
1. Import `build_source_map`, `format_source_map_for_prompt`, `resolve_citations`, `get_sources_dir`
2. At start: build source map from store
3. Format prompt with `report_metadata`, `source_map`
4. After LLM generation: run `resolve_citations()` to post-process
5. Log any citation warnings

```python
async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    # ... model setup unchanged ...

    # Build source map for citation resolution
    sources_dir = get_sources_dir()
    source_map = build_source_map(sources_dir)
    source_map_text = format_source_map_for_prompt(source_map)

    prompt = final_report_prompt.format(
        brief=state["research_brief"],
        notes=state["notes"],
        report_metadata=state.get("report_metadata", ""),
        source_map=source_map_text,
        date=...,
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])
    report = response.text or ""

    # Post-process: resolve [sN] → [N] + append Sources section
    if report and source_map:
        report, warnings = resolve_citations(report, source_map)
        for w in warnings:
            logger.warning("Citation warning: %s", w)

    # ... fallback handling unchanged ...
    return {"final_report": report}
```

### Step 5 — Handle subgraph state propagation

**Resolved in Step 0**: Added `report_metadata` to both `CoordinatorState`
and `AgentState`. LangGraph propagates matching keys from subgraph output
to parent state — verified working in integration tests. No workaround
needed (unlike `sources_dir` in Increment 5 which required module-level cache).

### Step 6 — Tests

**File**: `tests/test_source_store.py` (extend)

12 new unit tests for citation resolution and formatting:
- `test_format_source_map_for_prompt` — correct formatting
- `test_format_source_map_empty` — empty map returns placeholder
- `test_resolve_single_citation` — `[a1b2c3d4]` → `[1]`
- `test_resolve_multi_citation` — `[a1b2c3d4, e5f6a7b8]` → `[1, 2]`
- `test_resolve_repeated_citation` — same ID cited twice → same number
- `test_resolve_ordering_by_first_appearance` — sequential numbers follow appearance order
- `test_resolve_unknown_citation` — ID not in store → removed, warning returned
- `test_resolve_mixed_known_unknown` — known IDs kept, unknown dropped in same bracket
- `test_resolve_appends_sources_section` — Sources section at end with correct URLs
- `test_resolve_strips_llm_sources_section` — removes LLM-generated `## Sources` before appending
- `test_resolve_no_citations` — report with no source IDs returned unchanged
- `test_resolve_empty_source_map` — empty source map returns report unchanged

Note: metadata formatting unit tests (`_format_report_metadata`) deferred — the
function is tested implicitly via integration tests. Add if the formatting logic
gets more complex.

**File**: `tests/test_integration.py` (extend existing citation test)

Extended `test_citation_tracking_end_to_end` with 3 new assertions:
- Report contains sequential `[N]` citations (not raw `[sN]` hex IDs)
- Report has `## Sources` section
- No raw hex source IDs remain in the report

---

## Files Changed

| File | Change |
|------|--------|
| `src/deep_research/state.py` | Add `report_metadata: str` to AgentState and CoordinatorState |
| `src/deep_research/nodes/coordinator/reflect.py` | Add `_format_report_metadata()`, write to state on exit |
| `src/deep_research/nodes/researcher/adapter.py` | Format metadata from single researcher state |
| `src/deep_research/helpers/source_store.py` | Add `resolve_citations()`, `format_source_map_for_prompt()` |
| `src/deep_research/prompts.py` | Rewrite `final_report_prompt` with metadata, source map, [sN] instructions |
| `src/deep_research/nodes/report.py` | Wire source map + metadata + post-processing |
| `tests/test_source_store.py` | 12 new citation resolution unit tests |
| `tests/test_integration.py` | Extend citation e2e test for report-level verification |

## Files NOT Changed

| File | Why not |
|------|---------|
| `source_store.py` (existing functions) | `write_source`, `read_source`, `build_source_map` unchanged — already correct |
| `tools/search/base.py` | Search tool unchanged — [sN] format already in place from Increment 5 |
| `prompts.py` (other prompts) | Only `final_report_prompt` changes. Compress/reflect prompts already use [sN] |
| `researcher/reflect.py` | Researcher reflection unchanged — it already accumulates metadata |
| `coordinator/coordinator.py` | Coordinator LLM invocation unchanged |
| `models.py` | No new Pydantic schemas — `report_metadata` is a string, not structured |

## Observations from Integration Testing

End-to-end runs on "latest Python version" (simple path, 3 sources) and
"health effects of intermittent fasting" (coordinator path, 5 researchers,
112 sources):

**What works well:**

- **Citation resolution**: All `[sN]` hex tags replaced with sequential `[N]`.
  Zero raw hex IDs remaining in final reports. Deterministic Sources section
  appended with real URLs from the store.
- **Hallucinated citation detection**: LLM invented source IDs not in the store
  (6 in the fasting report, 2 in the Python report). The resolver caught and
  removed all of them, logging warnings. Validates the programmatic approach
  over trusting the LLM with URLs.
- **Contradictions surfaced**: The fasting report produced multiple "Conflicting
  Evidence" subsections (weight loss efficacy, cholesterol levels, thyroid
  hormones, ghrelin levels, cardiovascular risk). Both sides cited with sources.
- **Gap types distinguished**: The LLM correctly used metadata labels to explain
  why gaps exist. "Persistent gap in large-scale long-term human clinical
  studies" (searched but not found) vs "detailed considerations for adolescents
  were not extensively covered" (partial coverage).
- **Confidence calibration**: Topics with `knowledge_state: partial` got hedging
  ("still developing", "most evidence comes from short- to medium-term studies").
  Topics with `sufficient` were more assertive.
- **Subgraph state propagation**: `report_metadata` propagates from
  CoordinatorState → AgentState via matching key names. No workaround needed.

**Known issues for later development:**

1. **Contradictions buried inline without clear hierarchy**: The LLM placed
   contradictions as inline `**Conflicting Evidence: ...**` subsections within
   topic sections. Contextually correct but visually hard to scan — no clear
   structural separation. Will improve naturally when Increment 7 adds
   contradiction resolution with analysis (richer content warrants proper
   subsections). Not worth fixing with a prompt tweak while contradictions are
   still raw "A says X, B says Y."

2. **No summary "Areas for Further Research" section**: Gaps are noted inline
   within relevant topic sections (good for context) but there's no top-level
   summary collecting all gaps for a quick scan. Consider prompting for both:
   inline context + summary section. Defer to Increment 7 when the report
   prompt gets further refinement alongside contradiction resolution.

3. **Duplicate contradictions in metadata**: `accumulated_contradictions` uses
   an append reducer (`operator.add`), so if round 2 and round 3 both identify
   the same contradiction, it appears twice in the metadata. Root cause is in
   reflection accumulation, not in report formatting. Fix in Increment 7 when
   reworking the contradiction pipeline (resolution, dedup, trust scoring).

4. **LLM still hallucinates source IDs**: Despite the prompt saying "Do NOT
   invent source IDs," the LLM creates ones not in the store (6 out of ~90
   citations in the fasting report). The programmatic resolver catches and
   removes them, but this means some claims in the report lose their citation
   silently. Consider: flag removed citations in the report text (e.g.,
   "[citation needed]") rather than silently removing.

## Deferred

- **Contradiction resolution** (Increment 7): This increment presents unresolved contradictions transparently. Resolution mechanism (follow-up searches, trust scoring, structured analysis) is Increment 7 scope. Will also improve report structure for contradictions.
- **Contradiction dedup** (Increment 7): Deduplicate `accumulated_contradictions` at reflection or metadata formatting level.
- **Report structure refinement** (Increment 7): Summary sections for gaps and contradictions alongside inline context. Better visual hierarchy.
- **URL authority scoring**: Would help report calibrate source credibility. Increment 7/8 scope.
- **Report length control**: No token budget or length targeting yet. If reports are too long/short, address separately.
- **Hallucinated citation handling**: Consider marking removed citations as `[citation needed]` instead of silent removal.
