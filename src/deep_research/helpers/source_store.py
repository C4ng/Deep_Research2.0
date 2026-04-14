"""File-based source store for citation tracking.

Each URL encountered during research gets one file on disk with metadata
(url, title, source_id) and content (summary + key excerpts). Source IDs
are deterministic (URL hash), providing natural dedup across rounds and
researchers.

The store is the single source of truth for citations. At report time,
the store resolves [sN] references to actual URLs.
"""

import hashlib
import logging
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Cached temp directory for when sources_dir is not configured
_default_sources_dir: Path | None = None


def generate_source_id(url: str) -> str:
    """Generate a deterministic source ID from a URL.

    Uses md5 hash truncated to 8 hex chars. Deterministic: same URL
    always produces the same ID, across rounds and researchers.
    """
    return hashlib.md5(url.encode()).hexdigest()[:8]


def write_source(
    sources_dir: Path,
    source_id: str,
    url: str,
    title: str,
    summary: str,
    key_excerpts: str,
) -> bool:
    """Write a source file if it doesn't already exist.

    Returns True if the file was written, False if it already existed (dedup).
    """
    path = sources_dir / f"{source_id}.md"
    if path.exists():
        logger.debug("Source %s already exists, skipping write", source_id)
        return False

    content = (
        f"---\n"
        f"source_id: {source_id}\n"
        f"url: {url}\n"
        f"title: {title}\n"
        f"---\n"
        f"<summary>\n{summary}\n</summary>\n\n"
        f"<key_excerpts>\n{key_excerpts}\n</key_excerpts>\n"
    )
    path.write_text(content, encoding="utf-8")
    logger.debug("Wrote source %s: %s", source_id, url)
    return True


def read_source(sources_dir: Path, source_id: str) -> dict | None:
    """Read a source file and return its metadata and content.

    Returns a dict with keys: source_id, url, title, content.
    Returns None if the file doesn't exist.
    """
    path = sources_dir / f"{source_id}.md"
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    meta, content = _parse_frontmatter(text)
    return {
        "source_id": meta.get("source_id", source_id),
        "url": meta.get("url", ""),
        "title": meta.get("title", ""),
        "content": content,
    }


def build_source_map(sources_dir: Path) -> dict[str, dict]:
    """Read all source files and return a mapping of source_id to metadata.

    Returns {source_id: {url, title}} for citation resolution at report time.
    """
    source_map: dict[str, dict] = {}
    if not sources_dir.exists():
        return source_map

    for path in sources_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        meta, _ = _parse_frontmatter(text)
        sid = meta.get("source_id", path.stem)
        source_map[sid] = {
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
        }
    return source_map


def get_sources_dir() -> Path:
    """Get or create the sources directory.

    If sources_dir is set in Configuration, use it.
    Otherwise, create a temp directory (cached per process).
    """
    global _default_sources_dir

    if _default_sources_dir is not None:
        return _default_sources_dir

    _default_sources_dir = Path(
        tempfile.mkdtemp(prefix="deep_research_sources_")
    )
    logger.info("Created sources directory: %s", _default_sources_dir)
    return _default_sources_dir


def set_sources_dir(path: str | Path) -> Path:
    """Set the sources directory explicitly.

    Call before graph invocation to control where source files are stored.
    LangGraph strips custom configurable keys through subgraphs, so this
    uses a module-level cache instead of config propagation.
    """
    global _default_sources_dir
    _default_sources_dir = Path(path)
    _default_sources_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Sources directory set to: %s", _default_sources_dir)
    return _default_sources_dir


def reset_sources_dir() -> None:
    """Reset the sources directory cache. Used between runs/tests."""
    global _default_sources_dir
    _default_sources_dir = None


def format_source_map_for_prompt(source_map: dict[str, dict]) -> str:
    """Format source map as a lookup table for the report prompt.

    Produces one line per source so the LLM knows what sources exist
    and can reference them by their [source_id] tags.
    """
    if not source_map:
        return "No sources available."
    lines = []
    for sid, meta in source_map.items():
        title = meta.get("title", "Untitled")
        url = meta.get("url", "")
        lines.append(f"[{sid}] {title} — {url}")
    return "\n".join(lines)


# Regex for source ID references: [id] or [id1, id2, id3]
_CITATION_PATTERN = re.compile(
    r"\[([0-9a-f]{8}(?:\s*,\s*[0-9a-f]{8})*)\]"
)

# Regex to strip LLM-generated Sources/References section at end of report
_SOURCES_SECTION_PATTERN = re.compile(
    r"\n##\s+(?:Sources|References)\s*\n.*",
    re.DOTALL,
)


def resolve_citations(
    report: str, source_map: dict[str, dict]
) -> tuple[str, list[str]]:
    """Replace [source_id] tags with sequential [N] and append a Sources section.

    Handles both [id] and [id1, id2] formats (observed LLM behavior).

    Returns (resolved_report, warnings).
    Warnings list contains messages about source IDs referenced but not in store.
    """
    if not source_map:
        return report, []

    warnings: list[str] = []

    # First pass: collect all cited source IDs in order of first appearance
    cited_ids: list[str] = []
    for match in _CITATION_PATTERN.finditer(report):
        ids_str = match.group(1)
        for sid in re.split(r"\s*,\s*", ids_str):
            if sid not in cited_ids:
                cited_ids.append(sid)

    if not cited_ids:
        return report, []

    # Build source_id → sequential number mapping (only for known IDs)
    id_to_num: dict[str, int] = {}
    num = 1
    for sid in cited_ids:
        if sid in source_map:
            id_to_num[sid] = num
            num += 1
        else:
            warnings.append(f"Source ID [{sid}] referenced in report but not found in store")

    # Second pass: replace each citation match with sequential numbers
    def _replace_match(match: re.Match) -> str:
        ids_str = match.group(1)
        sids = re.split(r"\s*,\s*", ids_str)
        nums = [str(id_to_num[sid]) for sid in sids if sid in id_to_num]
        if not nums:
            return ""  # all IDs unknown — remove the bracket
        return "[" + ", ".join(nums) + "]"

    resolved = _CITATION_PATTERN.sub(_replace_match, report)

    # Strip any LLM-generated Sources/References section
    resolved = _SOURCES_SECTION_PATTERN.sub("", resolved).rstrip()

    # Append deterministic Sources section
    if id_to_num:
        source_lines = ["\n\n## Sources\n"]
        for sid, n in id_to_num.items():
            meta = source_map[sid]
            title = meta.get("title", "Untitled")
            url = meta.get("url", "")
            source_lines.append(f"[{n}] {title}: {url}")
        resolved += "\n".join(source_lines) + "\n"

    return resolved, warnings


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse simple YAML-ish frontmatter from a source file.

    Expects format:
        ---
        key: value
        key: value
        ---
        body content

    No external YAML library needed — frontmatter is always 3 fixed fields.
    """
    if not text.startswith("---"):
        return {}, text

    # Find the closing ---
    end = text.find("---", 3)
    if end == -1:
        return {}, text

    frontmatter_str = text[3:end].strip()
    body = text[end + 3:].strip()

    meta: dict[str, str] = {}
    for line in frontmatter_str.splitlines():
        line = line.strip()
        if not line:
            continue
        colon = line.find(":")
        if colon == -1:
            continue
        key = line[:colon].strip()
        value = line[colon + 1:].strip()
        meta[key] = value

    return meta, body
