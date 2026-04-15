import { useEffect, useRef } from "react";
import type { ActivityItem } from "../types";

interface ActivityPanelProps {
  items: ActivityItem[];
  isRunning: boolean;
}

const TYPE_STYLES: Record<string, string> = {
  status: "border-l-gray-400",
  log: "border-l-gray-300",
  dispatch: "border-l-blue-500",
  researcher_complete: "border-l-green-500",
  results_collected: "border-l-emerald-500",
  coordinator_done: "border-l-purple-500",
  report_generated: "border-l-amber-500",
};

const TYPE_ICONS: Record<string, string> = {
  status: "\u25CF",
  log: "\u25CB",
  dispatch: "\u25B6",
  researcher_complete: "\u2713",
  results_collected: "\u25A0",
  coordinator_done: "\u25C6",
  report_generated: "\u2605",
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function ActivityPanel({ items, isRunning }: ActivityPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
          Activity
        </h2>
        {isRunning && (
          <span className="inline-flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Running
          </span>
        )}
      </div>

      {/* Event list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
        {items.length === 0 && (
          <p className="text-sm text-gray-400 text-center mt-8">
            Activity will appear here once research starts.
          </p>
        )}

        {items.map((item) => (
          <div
            key={item.id}
            className={`border-l-2 pl-3 py-1.5 ${TYPE_STYLES[item.type] || "border-l-gray-300"}`}
          >
            <div className="flex items-start gap-2">
              <span className="text-xs mt-0.5 opacity-60">
                {TYPE_ICONS[item.type] || "\u25CB"}
              </span>
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm whitespace-pre-wrap break-words ${
                    item.level === "warning"
                      ? "text-amber-600 dark:text-amber-400"
                      : item.level === "error"
                        ? "text-red-600 dark:text-red-400"
                        : "text-gray-700 dark:text-gray-300"
                  }`}
                >
                  {item.message}
                </p>
                <span className="text-[10px] text-gray-400">
                  {formatTime(item.timestamp)}
                </span>
              </div>
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
