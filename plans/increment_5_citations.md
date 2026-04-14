# Increment 5 — Citation System
**Goal**: Structured citation tracking from search through compression, with stable source IDs and verifiable references.
**Status**: Complete
**Depends on**: Increment 4 (Question Stage) — completed

## Overview

Citations are currently untracked text. URLs enter as structured `SearchResult` objects, get formatted into plain text (`--- SOURCE 1: Title ---\nURL: ...`), then survive (or don't) through two LLM compression steps. There is no source registry, no dedup across rounds, and no way to verify a URL in the final report was actually returned by search.

This increment builds the citation infrastructure. The next increment (6 — Final Report) uses it.

---

## Current Citation Flow (Problems)

```
Level 1: Search Results
  SearchResult(url, title, content, raw_content)
  → Citation: inherent. 1:1 with URL. Structured.
  → PROBLEM: dedup only within a single tavily_search() call.
    Same URL can appear in round 1 and round 2 of same researcher,
    and across different researchers.

Level 2: Webpage Summary
  summarize_content(raw_content) → WebpageSummary(summary, key_excerpts)
  → Citation: trivial 1:1. One summary = one URL.
  → PROBLEM: summarization is topic-agnostic. Same URL produces same
    summary regardless of which query found it or what the researcher
    is investigating. Missed opportunity for focused extraction.

Level 3: Formatted Tool Output
  _format_results() → "--- SOURCE 1: Title ---\nURL: ...\n{summary}\n"
  → Citation: positional. SOURCE N → URL is clear.
  → THIS IS WHERE STRUCTURE IS LOST. After this, URLs are just text.

Level 4: Summarizer Compression (per researcher)
  All ToolMessages → compress → notes string
  → The prompt says "preserve citations and source URLs."
  → PROBLEM: synthesizes across sources. When information from multiple
    sources is merged, which URL to attribute? LLM best-effort, no guarantee.
  → PROBLEM: SOURCE numbers are per-tool-call, collide across rounds.
    Round 1 has SOURCE 1-5, round 2 has SOURCE 1-5 — different URLs.

Level 5: Reflection Metadata (per researcher round)
  key_findings: ["companies raised $3.77B (Source 2, 13)"]
  contradictions: ["Source 2 says X, Source 22 says Y"]
  → PROBLEM: references SOURCE numbers, not URLs. Numbers collide
    across rounds and are meaningless outside the tool call.
```

### Summary of issues

| Issue | Where | Impact |
|-------|-------|--------|
| No cross-round URL dedup | Tavily search across rounds | Duplicate summaries, wasted tokens |
| No cross-researcher URL dedup | Different researchers, same URL | Same URL summarized multiple times |
| Topic-agnostic summarization | BaseSearchTool.summarize_content | Generic summaries miss relevant aspects |
| SOURCE number collision | Per-tool-call numbering | References ambiguous across rounds |
| Structure lost at formatting | _format_results() | After this point, URL-content link is text-only |

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage approach | **File-based source store** | Subsumes in-memory registry. Content preserved on disk, no state bloat, natural dedup, debug-friendly. |
| URL dedup | **Same URL = same file, first write wins** | Deterministic source_id from URL hash. File exists → skip write AND skip re-summarization (saves API call). |
| Topic-aware summarization | **Deferred** | Generic summaries are usually good enough — Tavily returns query-relevant pages, researcher LLM handles topic filtering. Add later if quality issues observed. TODO(enhancement). |
| Store lifecycle | **Caller manages** | `sources_dir` in Configuration. If not set, source store creates and caches a temp directory. Caller can pass a persistent path for inspection. |
| Source ID format | **`[sN]` where N = md5(url)[:8]** | Deterministic, global, no collision across rounds/researchers. Short enough for LLMs to preserve through compression. |

---

## Design

### Source store on disk

Each URL gets one file on disk, written at search time:

```
sources/
  a1b2c3d4.md    ← md5(url)[:8]
  e5f6a7b8.md
  ...
```

File format:
```markdown
---
source_id: a1b2c3d4
url: https://example.com/quantum-report
title: "Quantum Computing Market 2025"
---
<summary>
Concise summary of the webpage content...
</summary>

<key_excerpts>
Important quotes or data points extracted verbatim...
</key_excerpts>
```

### How [sN] IDs flow through the pipeline

```
Search tool output (to researcher LLM):
  --- [a1b2c3d4] Quantum Computing Market 2025 ---
  URL: https://example.com/quantum-report
  {summary content}

Compressed notes (summarizer output):
  Quantum computing market reached $3.77B [a1b2c3d4]. IBM announced
  a 1000-qubit processor [e5f6a7b8], contradicting Google's timeline [a1b2c3d4][f9c0d1e2].

Reflection metadata:
  key_findings: ["Market reached $3.77B [a1b2c3d4]"]
  contradictions: ["[e5f6a7b8] says X, [f9c0d1e2] says Y"]

Report time (Increment 6):
  Read source store → build source_id → URL map → resolve [sN] to [Title](URL)
```

### Cross-round dedup saves API calls

```python
# In search_and_summarize, before calling summarize_content():
source_id = generate_source_id(result.url)
existing = read_source(sources_dir, source_id)
if existing:
    # URL already summarized in a prior round — reuse stored summary
    summary_text = existing["content"]
    # Skip the summarization LLM call entirely
else:
    summary = await self.summarize_content(result.raw_content)
    write_source(sources_dir, source_id, result.url, result.title, summary)
    summary_text = format_summary(summary)
```

---

## Implementation Steps

### Step 0 — Source store helpers

**New file**: `src/deep_research/helpers/source_store.py`

**Functions:**

```python
def generate_source_id(url: str) -> str:
    """Deterministic source ID from URL. md5(url)[:8]."""

def write_source(
    sources_dir: Path,
    source_id: str,
    url: str,
    title: str,
    summary: str,
    key_excerpts: str,
) -> bool:
    """Write a source file if it doesn't exist. Returns True if written, False if skipped (dedup)."""

def read_source(sources_dir: Path, source_id: str) -> dict | None:
    """Read a source file. Returns {source_id, url, title, content} or None if not found."""

def build_source_map(sources_dir: Path) -> dict[str, dict]:
    """Read all source files. Returns {source_id: {url, title}} for report-time resolution."""

def get_sources_dir(config: RunnableConfig) -> Path:
    """Get or create the sources directory from config.
    If sources_dir is set in Configuration, use it.
    If not, create a temp directory (cached per process)."""
```

**File format**: YAML-ish frontmatter (`---` delimited) with url, title, source_id. Body is the summary content. No external YAML library — simple string parsing (frontmatter is 3 fields, always the same format).

**Changes to existing files**: None. Pure new module.

### Step 1 — Wire sources_dir through Configuration

**File**: `src/deep_research/configuration.py`

Add one field:
```python
sources_dir: str = Field(
    default="",
    description="Directory for source files. If empty, auto-created temp directory.",
)
```

The `get_sources_dir()` helper in source_store.py resolves this — reads from config, falls back to a cached temp dir if empty.

### Step 2 — Modify search tool to register sources + use [sN] IDs

**File**: `src/deep_research/tools/search/base.py`

This is the main change. The `search_and_summarize()` method currently does:
1. `search()` → list of SearchResult
2. For each result, `summarize_content(raw_content)` → WebpageSummary
3. `_format_results(results, summaries)` → formatted string

After this step:
1. `search()` → list of SearchResult (unchanged)
2. For each result:
   - `generate_source_id(result.url)`
   - Check source store: file exists? → load stored summary, skip LLM call
   - File doesn't exist? → `summarize_content(raw_content)` → write to store
3. `_format_results(results, summaries, source_ids)` → formatted string with `[sN]` IDs

**Specific changes to `search_and_summarize()`** (base.py:86-118):

```python
async def search_and_summarize(self, queries, *, max_results=5) -> str:
    results = await self.search(queries, max_results=max_results)
    if not results:
        return "No search results found. Try different search queries."

    sources_dir = get_sources_dir(self._config)
    source_ids = []
    summaries = []

    async def _summarize_one(result: SearchResult) -> tuple[str, str | None]:
        source_id = generate_source_id(result.url)

        # Cross-round dedup: check store before summarizing
        existing = read_source(sources_dir, source_id)
        if existing:
            logger.info("Source %s already in store, skipping summarization", source_id)
            return source_id, existing["content"]

        if not result.raw_content:
            return source_id, None

        try:
            summary = await self.summarize_content(result.raw_content)
            content = (
                f"<summary>\n{summary.summary}\n</summary>\n\n"
                f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
            )
            write_source(sources_dir, source_id, result.url, result.title,
                         summary.summary, summary.key_excerpts)
            return source_id, content
        except Exception as e:
            logger.warning("Summarization failed for %s: %s", result.url, e)
            return source_id, None

    pairs = await asyncio.gather(*[_summarize_one(r) for r in results])
    source_ids = [p[0] for p in pairs]
    summaries = [p[1] for p in pairs]

    return self._format_results(results, summaries, source_ids)
```

**Specific changes to `_format_results()`** (base.py:120-132):

```python
@staticmethod
def _format_results(
    results: list[SearchResult],
    summaries: list[str | None],
    source_ids: list[str],
) -> str:
    output = "Search results:\n\n"
    for result, summary, sid in zip(results, summaries, source_ids):
        content = summary if summary else result.content
        output += f"\n--- [{sid}] {result.title} ---\n"
        output += f"URL: {result.url}\n\n"
        output += f"{content}\n\n"
        output += "-" * 80 + "\n"
    return output
```

Before: `--- SOURCE 1: Title ---` (positional, collides across rounds)
After: `--- [a1b2c3d4] Title ---` (stable, global, deterministic)

### Step 3 — Update prompts for [sN] references

**File**: `src/deep_research/prompts.py`

**compress_research_prompt** — the summarizer's instructions change from "preserve all citations and source URLs" to using `[sN]` IDs:

```python
compress_research_prompt = """\
Compress raw research findings into a concise synthesis for a report-writing agent. \
Today's date is {date}.

<research_topic>
{research_topic}
</research_topic>

<raw_findings>
{tool_results}
</raw_findings>

<instructions>
1. Reference each finding by its source ID (e.g., [a1b2c3d4]). Source IDs
   appear in the raw findings as [source_id] tags — preserve them exactly.
   Do NOT invent or modify source IDs.
2. Deduplicate overlapping information across sources.
3. Keep specific facts, data points, statistics, and direct quotes.
4. Remove boilerplate, navigation text, and irrelevant content.
5. Group related information together by subtopic.
6. Target roughly 30% of the original length.
</instructions>"""
```

Key change: instruction 1 replaces "preserve all citations and source URLs" with "reference by source ID." The LLM preserves short IDs far more reliably than full URLs.

**researcher_reflection_prompt** — add guidance in `<field_criteria>` for key_findings and contradictions to use `[sN]` IDs:

```
key_findings:
- Include specific facts and data points discovered this round.
- Reference sources using their [source_id] tags from the search results
  (e.g., "Market reached $3.77B [a1b2c3d4]").
- ...

contradictions:
- Cite which sources disagree using their [source_id] tags
  (e.g., "[a1b2c3d4] says X, [e5f6a7b8] says Y").
```

These are additions to existing field_criteria entries, not replacements. The rest of the prompt stays the same.

### Step 4 — Tests

**New file**: `tests/test_source_store.py`

Unit tests for source store helpers:
- `test_generate_source_id_deterministic` — same URL → same ID
- `test_generate_source_id_different_urls` — different URLs → different IDs
- `test_write_source_creates_file` — file created with correct frontmatter + content
- `test_write_source_dedup` — second write same URL → returns False, file unchanged
- `test_read_source_exists` — reads back what was written
- `test_read_source_not_found` — returns None
- `test_build_source_map` — reads all files into {source_id: {url, title}} dict

Integration-level test (can be a unit test with mocked search):
- `test_format_results_uses_source_ids` — verify output contains `[sN]` not `SOURCE N`

---

## Files Changed

| File | Change |
|------|--------|
| `src/deep_research/helpers/source_store.py` | **New** — source store read/write/dedup helpers |
| `src/deep_research/configuration.py` | Add `sources_dir` field |
| `src/deep_research/tools/search/base.py` | `search_and_summarize()` writes to store + dedup; `_format_results()` uses `[sN]` IDs |
| `src/deep_research/prompts.py` | `compress_research_prompt` + `researcher_reflection_prompt` updated for `[sN]` references |
| `tests/test_source_store.py` | **New** — unit tests for source store |

## Files NOT Changed

| File | Why not |
|------|---------|
| `tavily.py` | Tavily-specific search logic unchanged. Within-call dedup stays (still useful). Cross-round dedup handled by base class via store. |
| `researcher.py` | Researcher LLM naturally sees `[sN]` in tool output and uses them — no prompt change needed. |
| `reflect.py` | Reflection prompt updated (Step 3) but reflection node code unchanged — it already passes tool results to the prompt. |
| `summarizer.py` | Node code unchanged — prompt does the work. The summarizer still joins ToolMessages and compresses. |
| `state.py` | No new state fields. Source store is on disk, not in graph state. |
| `coordinator/` | Coordinator doesn't touch citations directly. `[sN]` IDs flow through researcher → notes → coordinator → report. |

## Observations from Integration Testing

End-to-end run on "ocean plastic pollution" (19 sources, 2 research rounds):

**What works well:**
- **Summarizer attribution**: Nearly every claim in compressed notes has source IDs. Multi-source corroboration occurs naturally (e.g., `[9f5a3ca9, ce77b549]` for a finding confirmed by two sources).
- **Key findings**: All 19 accumulated findings reference specific source IDs. The LLM consistently tags facts with the sources they came from.
- **Contradictions**: Source IDs used to cite exactly which sources disagree (e.g., different tonnage estimates attributed to `[d8329a64]` vs `[46719a5f, 7b035619]` vs `[e2e6b229]`).
- **Short IDs preserved through compression**: 8-char hex IDs survive the summarizer far better than full URLs ever would.

**Issue: LLM citation format**
The LLM prefers comma-separated IDs in a single bracket: `[id1, id2]` rather than separate brackets `[id1][id2]`. This means:
- Report-time citation resolver (Increment 6) must parse `[id1, id2]` format, not just `[id1]`
- Regex for counting/verifying citations needs to handle both formats

**Coverage**: Not all source IDs survive compression — some are lost when the summarizer deduplicates overlapping info (expected at high compression ratios). The source store preserves originals on disk regardless.

## Deferred

- **Topic-aware summarization**: TODO(enhancement). Inject research topic into `summarize_webpage_prompt` so extraction focuses on relevant aspects. Not needed for citation tracking — add if quality issues observed.
- **Report-time citation resolution**: Increment 6 scope. Source store provides the data; report node consumes it.
