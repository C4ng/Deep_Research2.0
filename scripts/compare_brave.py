"""Quick comparison: run the full pipeline with Brave search.

Compare the LangSmith trace with prior Tavily runs.
Usage: python scripts/compare_brave.py
"""

import asyncio
import re
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from deep_research.graph.graph import build_graph
from deep_research.helpers.source_store import (
    build_source_map,
    reset_sources_dir,
    set_sources_dir,
)


QUERY = "What is the current state of quantum computing in 2025?"

CONFIG = {
    "configurable": {
        "allow_clarification": False,
        "allow_human_review": False,
        "search_api": "brave",
    }
}

INITIAL_STATE = {
    "messages": [HumanMessage(content=QUERY)],
    "research_brief": "",
    "is_simple": False,
    "notes": "",
    "report_metadata": "",
    "final_report": "",
}


async def main():
    with tempfile.TemporaryDirectory() as sources_dir:
        set_sources_dir(sources_dir)
        try:
            graph = build_graph()

            print(f"Query: {QUERY}")
            print(f"Search API: brave")
            print(f"Sources dir: {sources_dir}")
            print("=" * 70)
            print("Running pipeline...\n")

            result = await graph.ainvoke(INITIAL_STATE, config=CONFIG)

            # --- Summary ---
            report = result.get("final_report", "")
            notes = result.get("notes", "")
            brief = result.get("research_brief", "")
            metadata = result.get("report_metadata", "")

            source_map = build_source_map(Path(sources_dir))
            note_ids = set(re.findall(r"\[([0-9a-f]{8})\]", notes))
            seq_citations = re.findall(r"\[(\d+)\]", report)
            unverified = report.count("[unverified]")

            print("=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)
            print(f"Brief length:       {len(brief)} chars")
            print(f"Notes length:       {len(notes)} chars")
            print(f"Report length:      {len(report)} chars")
            print(f"Sources in store:   {len(source_map)}")
            print(f"Source IDs in notes: {len(note_ids)}")
            print(f"Citations in report: {len(seq_citations)}")
            print(f"[unverified] marks: {unverified}")
            print(f"Has ## Sources:     {'## Sources' in report}")
            print()

            # Show first 2000 chars of report
            print("=" * 70)
            print("REPORT PREVIEW (first 2000 chars)")
            print("=" * 70)
            print(report[:2000])
            if len(report) > 2000:
                print(f"\n... ({len(report) - 2000} more chars)")

        finally:
            reset_sources_dir()


if __name__ == "__main__":
    asyncio.run(main())
