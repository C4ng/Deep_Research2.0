import { useState } from "react";
import type { CoordinatorState, ResearcherState } from "../types";

interface StatePanelProps {
  researchers: Map<string, ResearcherState>;
  coordinator: CoordinatorState;
  isRunning: boolean;
}

const KS_COLORS: Record<string, string> = {
  sufficient: "text-green-600 dark:text-green-400",
  partial: "text-yellow-600 dark:text-yellow-400",
  insufficient: "text-red-500 dark:text-red-400",
  unavailable: "text-gray-500 dark:text-gray-500",
};

const KS_BADGES: Record<string, string> = {
  sufficient: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400",
  partial: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
  insufficient: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400",
  unavailable: "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400",
};

function KnowledgeBadge({ state }: { state: string }) {
  if (!state) return null;
  return (
    <span
      className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${KS_BADGES[state] || KS_BADGES.partial}`}
    >
      {state}
    </span>
  );
}

function CollapsibleList({
  title,
  items,
  color = "text-gray-600 dark:text-gray-400",
  icon,
}: {
  title: string;
  items: string[];
  color?: string;
  icon: string;
}) {
  const [open, setOpen] = useState(false);
  if (items.length === 0) return null;

  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(!open)}
        className={`text-xs ${color} flex items-center gap-1 hover:opacity-80`}
      >
        <span className="text-[10px]">{open ? "▼" : "▶"}</span>
        <span>{icon}</span>
        <span>
          {title} ({items.length})
        </span>
      </button>
      {open && (
        <ul className="ml-5 mt-0.5 space-y-0.5">
          {items.map((item, i) => (
            <li key={i} className={`text-xs ${color}`}>
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ResearcherCard({ researcher }: { researcher: ResearcherState }) {
  const [expanded, setExpanded] = useState(true);
  const latestRound = researcher.rounds[researcher.rounds.length - 1];
  const isComplete = !!researcher.finalState;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 space-y-2">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1.5 text-left min-w-0"
        >
          <span className="text-xs text-gray-400 shrink-0">
            {expanded ? "▼" : "▶"}
          </span>
          <span className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
            {researcher.topic}
          </span>
        </button>
        <div className="flex items-center gap-1.5 shrink-0">
          {isComplete ? (
            <KnowledgeBadge state={researcher.finalState!} />
          ) : (
            <span className="text-[10px] text-blue-500 animate-pulse">
              researching...
            </span>
          )}
        </div>
      </div>

      {expanded && (
        <div className="space-y-2 ml-4">
          {researcher.rounds.map((round) => (
            <div key={round.round} className="space-y-0.5">
              {/* Round header */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-gray-400">
                  R{round.round}
                </span>
                {round.knowledge_state && (
                  <KnowledgeBadge state={round.knowledge_state} />
                )}
              </div>

              {/* Search queries */}
              {round.queries.length > 0 && (
                <div className="space-y-0.5">
                  {round.queries.map((q, i) => (
                    <div
                      key={i}
                      className="text-xs text-gray-500 dark:text-gray-400 flex items-start gap-1.5"
                    >
                      <span className="text-blue-400 shrink-0 mt-px">🔍</span>
                      <span className="font-mono text-[11px]">{q}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Findings / Gaps / Contradictions */}
              <CollapsibleList
                title="Findings"
                items={round.key_findings}
                icon="✓"
                color="text-green-600 dark:text-green-400"
              />
              <CollapsibleList
                title="Gaps"
                items={round.missing_info}
                icon="?"
                color="text-yellow-600 dark:text-yellow-400"
              />
              <CollapsibleList
                title="Contradictions"
                items={round.contradictions}
                icon="⚡"
                color="text-red-500 dark:text-red-400"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CoordinatorCard({ coordinator }: { coordinator: CoordinatorState }) {
  if (coordinator.dispatches.length === 0) return null;

  const latestReflection =
    coordinator.reflections[coordinator.reflections.length - 1];

  return (
    <div className="border border-purple-200 dark:border-purple-800 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
          Coordinator
        </span>
        {latestReflection && (
          <KnowledgeBadge state={latestReflection.knowledge_state} />
        )}
      </div>

      {/* Rounds */}
      {coordinator.dispatches.map((dispatch, i) => {
        const results = coordinator.results[i];
        const reflection = coordinator.reflections[i];

        return (
          <div
            key={i}
            className="border-l-2 border-purple-300 dark:border-purple-700 pl-2.5 space-y-1"
          >
            <div className="text-xs text-purple-600 dark:text-purple-400 font-medium">
              Round {i + 1} — {dispatch.count} researchers
            </div>

            {/* Result summary per topic */}
            {results &&
              results.results.map((r, j) => (
                <div
                  key={j}
                  className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400"
                >
                  <span
                    className={`${KS_COLORS[r.knowledge_state] || ""}`}
                  >
                    {r.knowledge_state === "sufficient"
                      ? "✓"
                      : r.knowledge_state === "unavailable"
                        ? "✗"
                        : "◐"}
                  </span>
                  <span className="truncate">{r.topic}</span>
                  <span className="text-[10px] text-gray-400 shrink-0">
                    {r.findings_count}f {r.gaps_count}g{" "}
                    {r.contradictions_count > 0 &&
                      `${r.contradictions_count}c`}
                  </span>
                </div>
              ))}

            {/* Reflection */}
            {reflection && (
              <div className="space-y-1 mt-1">
                <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                  {reflection.overall_assessment}
                </p>
                <CollapsibleList
                  title="Coverage gaps"
                  items={reflection.coverage_gaps}
                  icon="?"
                  color="text-yellow-600 dark:text-yellow-400"
                />
                <CollapsibleList
                  title="Cross-topic contradictions"
                  items={reflection.cross_topic_contradictions}
                  icon="⚡"
                  color="text-red-500 dark:text-red-400"
                />
                {reflection.should_continue && (
                  <div className="text-[10px] text-purple-500">
                    → dispatching round {i + 2}...
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function StatePanel({
  researchers,
  coordinator,
  isRunning,
}: StatePanelProps) {
  const researcherList = Array.from(researchers.values());
  const hasContent = researcherList.length > 0 || coordinator.dispatches.length > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
          Research State
        </h2>
        {isRunning && (
          <span className="inline-flex items-center gap-1.5 text-xs text-blue-600 dark:text-blue-400">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            Live
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {!hasContent && (
          <p className="text-sm text-gray-400 text-center mt-8">
            Research state will appear here once researchers are dispatched.
          </p>
        )}

        {/* Coordinator overview */}
        <CoordinatorCard coordinator={coordinator} />

        {/* Researcher cards */}
        {researcherList.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              Researchers ({researcherList.length})
            </h3>
            {researcherList.map((r) => (
              <ResearcherCard key={r.topic} researcher={r} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
