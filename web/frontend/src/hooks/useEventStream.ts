import { useCallback, useRef, useState } from "react";
import type {
  ActivityItem,
  ChatMessage,
  CoordinatorDispatchEvent,
  CoordinatorReflectionEvent,
  CoordinatorResultsEvent,
  CoordinatorState,
  ResearcherReflectionEvent,
  ResearcherSearchEvent,
  ResearcherState,
  SSEEvent,
} from "../types";

let idCounter = 0;
const nextId = () => String(++idCounter);

function formatActivityMessage(event: SSEEvent): string | null {
  if (event.channel !== "activity") return null;
  const d = event.data as Record<string, unknown>;

  switch (event.type) {
    case "status":
      if (d.status === "running") return "Research started...";
      if (d.status === "complete") return "Research complete.";
      if (d.status === "waiting") return "Waiting for your input.";
      return `Status: ${d.status}`;

    case "log":
      return `${d.message}`;

    case "dispatch": {
      const topics = d.topics as string[];
      const lines = topics.map((t) => `  - ${t}`);
      return `Dispatching ${d.count} researchers:\n${lines.join("\n")}`;
    }

    case "researcher_complete":
      return `Researcher finished: ${d.name}`;

    case "results_collected": {
      const results = d.results as Array<{
        topic: string;
        knowledge_state: string;
      }>;
      const lines = results.map(
        (r) => `  - ${r.topic} (${r.knowledge_state})`
      );
      return `Collected ${d.count} results:\n${lines.join("\n")}`;
    }

    case "coordinator_done":
      return `Coordinator round ${d.iteration} complete.`;

    case "report_generated":
      return `Report generated (${d.length} chars).`;

    default:
      return null;
  }
}

export function useEventStream() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [activityItems, setActivityItems] = useState<ActivityItem[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [needsInput, setNeedsInput] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // State panel: accumulated researcher and coordinator state
  const [researchers, setResearchers] = useState<Map<string, ResearcherState>>(
    new Map()
  );
  const [coordinator, setCoordinator] = useState<CoordinatorState>({
    round: 0,
    dispatches: [],
    results: [],
    reflections: [],
  });

  const addChat = useCallback((msg: ChatMessage) => {
    setChatMessages((prev) => [...prev, msg]);
  }, []);

  const addActivity = useCallback((item: ActivityItem) => {
    setActivityItems((prev) => [...prev, item]);
  }, []);

  const handleStateEvent = useCallback((event: SSEEvent) => {
    if (event.channel !== "state") return;
    const d = event.data as Record<string, unknown>;

    switch (event.type) {
      case "researcher_search": {
        const e = d as unknown as ResearcherSearchEvent;
        setResearchers((prev) => {
          const next = new Map(prev);
          const existing = next.get(e.topic) || {
            topic: e.topic,
            rounds: [],
          };
          // Find or create the current round entry
          const lastRound = existing.rounds[existing.rounds.length - 1];
          if (lastRound && lastRound.key_findings.length === 0) {
            // Still in the same round (no reflection yet), append queries
            lastRound.queries = [...lastRound.queries, ...e.queries];
          } else {
            // New round
            existing.rounds.push({
              round: existing.rounds.length + 1,
              queries: e.queries,
              knowledge_state: "",
              key_findings: [],
              missing_info: [],
              contradictions: [],
            });
          }
          next.set(e.topic, { ...existing });
          return next;
        });
        break;
      }

      case "researcher_reflection": {
        const e = d as unknown as ResearcherReflectionEvent;
        setResearchers((prev) => {
          const next = new Map(prev);
          const existing = next.get(e.topic) || {
            topic: e.topic,
            rounds: [],
          };
          const lastRound = existing.rounds[existing.rounds.length - 1];
          if (lastRound) {
            lastRound.knowledge_state = e.knowledge_state;
            lastRound.key_findings = e.key_findings;
            lastRound.missing_info = e.missing_info;
            lastRound.contradictions = e.contradictions;
          }
          if (!e.should_continue) {
            existing.finalState = e.knowledge_state;
          }
          next.set(e.topic, { ...existing });
          return next;
        });
        break;
      }

      case "coordinator_dispatch": {
        const e = d as unknown as CoordinatorDispatchEvent;
        setCoordinator((prev) => ({
          ...prev,
          round: prev.dispatches.length + 1,
          dispatches: [...prev.dispatches, e],
        }));
        break;
      }

      case "coordinator_results": {
        const e = d as unknown as CoordinatorResultsEvent;
        setCoordinator((prev) => ({
          ...prev,
          results: [...prev.results, e],
        }));
        break;
      }

      case "coordinator_reflection": {
        const e = d as unknown as CoordinatorReflectionEvent;
        setCoordinator((prev) => ({
          ...prev,
          reflections: [...prev.reflections, e],
        }));
        break;
      }
    }
  }, []);

  const handleEvent = useCallback(
    (event: SSEEvent) => {
      // State channel
      if (event.channel === "state") {
        handleStateEvent(event);
        return;
      }

      if (event.channel === "chat") {
        if (event.type === "ai_message") {
          addChat({
            id: nextId(),
            role: "assistant",
            content: event.data.content || "",
          });
        } else if (event.type === "report") {
          addChat({
            id: nextId(),
            role: "assistant",
            content: event.data.content || "",
            isReport: true,
          });
        } else if (event.type === "needs_input") {
          setNeedsInput(true);
        }
      }

      if (event.channel === "activity") {
        if (event.type === "status") {
          const status = event.data.status as string;
          if (status === "complete" || status === "waiting") {
            setIsRunning(false);
          }
        }
        const message = formatActivityMessage(event);
        if (message) {
          addActivity({
            id: nextId(),
            type: event.type,
            message,
            level:
              event.type === "log"
                ? ((event.data.level as "info" | "warning" | "error") || "info")
                : "info",
            timestamp: new Date(),
          });
        }
      }
    },
    [addChat, addActivity, handleStateEvent]
  );

  const connectStream = useCallback(
    (tid: string, params: string) => {
      // Close existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource(`/api/stream/${tid}?${params}`);
      eventSourceRef.current = es;

      es.onmessage = (e) => {
        try {
          const event: SSEEvent = JSON.parse(e.data);
          handleEvent(event);
        } catch {
          // skip malformed events
        }
      };

      es.onerror = () => {
        es.close();
        eventSourceRef.current = null;
        setIsRunning(false);
      };
    },
    [handleEvent]
  );

  const startResearch = useCallback(
    async (query: string) => {
      setIsRunning(true);
      setNeedsInput(false);
      setResearchers(new Map());
      setCoordinator({ round: 0, dispatches: [], results: [], reflections: [] });

      // Add user message to chat
      addChat({ id: nextId(), role: "user", content: query });

      // Create session
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const { thread_id } = await res.json();
      setThreadId(thread_id);

      // Connect SSE stream
      connectStream(thread_id, `query=${encodeURIComponent(query)}`);
    },
    [addChat, connectStream]
  );

  const resumeResearch = useCallback(
    async (message: string) => {
      if (!threadId) return;

      setIsRunning(true);
      setNeedsInput(false);

      addChat({ id: nextId(), role: "user", content: message });

      // Connect new SSE stream for resume
      connectStream(
        threadId,
        `resume_message=${encodeURIComponent(message)}`
      );
    },
    [threadId, addChat, connectStream]
  );

  return {
    chatMessages,
    activityItems,
    isRunning,
    needsInput,
    startResearch,
    resumeResearch,
    researchers,
    coordinator,
  };
}
