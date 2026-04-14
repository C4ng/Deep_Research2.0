# Increment 6 — Final Report Redesign
**Goal**: A report node that uses the full research metadata and produces verifiable citations.
**Status**: Draft
**Depends on**: Increment 5 (Citation System) — stable source IDs and source store

## Overview

The report node is currently underdesigned. It receives only `research_brief` (string) and `notes` (string). All intermediate metadata — key_findings, contradictions, gaps, knowledge_state — is discarded at the coordinator → AgentState boundary. The report is written blind, and citations are reconstructed from whatever text survived compression.

With the citation system from Increment 5 in place (stable `[sN]` IDs, source store on disk), the report can now:
- Resolve `[sN]` references to actual URLs from the store
- Verify every citation against the store (no hallucinated URLs)
- Use accumulated metadata to structure the report (contradictions, gaps, findings)

---

## Current State (Problems)

```
Report node sees:
  research_brief: "Title: ...\nQuestion: ...\nApproach: ..."  (string)
  notes: "## Topic: X\n{compressed notes}\n---\n## Topic: Y\n..."  (string)

Report node does NOT see:
  key_findings       — per-researcher accumulated findings (discarded)
  contradictions     — per-researcher + cross-topic (discarded)
  coverage_gaps      — known blind spots from coordinator (discarded)
  knowledge_state    — per-topic completeness (discarded)
  source store       — all source URLs with content (not wired)
```

The report prompt is generic — "create a comprehensive report" with basic citation rules. It doesn't know about contradictions to surface, gaps to flag as limitations, or which findings were strongest vs weakest.

---

## Proposed Design

### Metadata Flow to Report

Pass accumulated metadata from the coordinator to the report node via AgentState. New fields:

```python
class AgentState(TypedDict):
    # ... existing fields ...
    report_metadata: str    # formatted string with findings, contradictions, gaps
```

**Why a formatted string, not structured fields?** The report node passes this to an LLM prompt. Adding multiple list fields to AgentState for data that only the report uses is unnecessary — format the metadata into a readable string at the coordinator exit, pass it as one field.

**What the coordinator includes on exit (in `_merge_notes` or a new helper):**
- Per-topic key_findings (from `ResearchResult.key_findings`)
- Per-topic contradictions (from `ResearchResult.contradictions`)
- Per-topic knowledge_state (from `ResearchResult.knowledge_state`)
- Cross-topic contradictions (from `CoordinatorReflection.cross_topic_contradictions`)
- Coverage gaps (from `CoordinatorReflection.coverage_gaps`)
- Overall assessment (from `CoordinatorReflection.overall_assessment`)

### Report Node Receives

After this increment, the report node sees:
- `research_brief` — the research plan (existing)
- `notes` — compressed findings with `[sN]` source references (existing, improved by Increment 5)
- `report_metadata` — formatted findings, contradictions, gaps, knowledge states (new)
- Source store path (via config) — for building the Sources section

### Report Prompt Redesign

The report prompt should instruct the LLM to:
1. Write the main body from `notes`, using `[sN]` references
2. Surface contradictions as an "Open Questions" or "Conflicting Evidence" section
3. Surface coverage gaps as a "Limitations" or "Areas for Further Research" section
4. Use knowledge_state signals to weight confidence (well-covered vs partial)
5. Build a Sources section by resolving `[sN]` → URL from the store

### Citation Resolution at Report Time

The report node reads the source store and builds a source map:
```python
source_map = {}
for path in sources_dir.glob("*.md"):
    meta = parse_frontmatter(path)
    source_map[meta["source_id"]] = {"url": meta["url"], "title": meta["title"]}
```

This map is injected into the prompt so the LLM can write `[Title](URL)` with correct URLs:
```
<sources>
[s1] Quantum Computing Market 2025: https://example.com/quantum-report
[s2] IBM Press Release: https://ibm.com/quantum
...
</sources>
```

### Optional: Post-Generation Verification

After the report is generated, programmatically check that every `[sN]` cited in the report exists in the source store. Flag any that don't. This is cheap (regex + dict lookup) and catches hallucinated citations.

---

## Open Questions

1. **How much metadata is too much?** Passing all per-researcher key_findings, contradictions, and gaps could be a lot of text. Should we pass only the coordinator's final reflection (which already synthesizes cross-topic issues), or also per-researcher detail?

2. **Report structure flexibility**: Should the prompt prescribe a fixed structure (findings → contradictions → limitations → sources), or let the LLM adapt structure to the content? Different questions benefit from different report shapes.

3. **Post-generation verification**: Programmatic check that cited `[sN]` IDs exist in the store — is this worth building, or is providing the source map to the LLM sufficient?

4. **Simple question path**: For `is_simple=True` (single researcher, no coordinator), there's no coordinator reflection. How does metadata flow? The single researcher's accumulated findings/contradictions need to be formatted for the report directly.

---

## Proposed Steps (Draft)

### Step 0 — Add report_metadata to AgentState + format at coordinator exit

Add `report_metadata: str` to AgentState. At coordinator exit (in `coordinator_reflect`), format key_findings, contradictions, gaps, knowledge_state into a readable string and write to state.

### Step 1 — Wire source store path to report node

Pass the source store directory path through config so the report node can read source files. Build the source map (source_id → url, title) at report time.

### Step 2 — Redesign report prompt

Rewrite `final_report_prompt` to use notes with `[sN]` references, source map for URL resolution, and report_metadata for contradictions/gaps/findings. Add sections for open questions and limitations.

### Step 3 — Handle simple question path

For `is_simple=True`, format the single researcher's accumulated metadata into `report_metadata` before reaching the report node. Adapter in `run_single_researcher` or a shared helper.

### Step 4 — Optional citation verification

Post-generation step: extract all `[sN]` references from the report, check each against the source store, log warnings for any missing.

### Step 5 — Tests

Unit tests for metadata formatting, source map building, citation verification. Integration test verifying the report contains proper citations that match the store.
