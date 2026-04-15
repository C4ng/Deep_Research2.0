export type Channel = "chat" | "activity" | "state";

export interface ChatEvent {
  channel: "chat";
  type: "ai_message" | "needs_input" | "report";
  data: {
    content?: string;
    node?: string;
    prompt?: string;
  };
}

export interface ActivityEvent {
  channel: "activity";
  type:
    | "status"
    | "log"
    | "dispatch"
    | "researcher_complete"
    | "results_collected"
    | "coordinator_done"
    | "report_generated";
  data: Record<string, unknown>;
}

export interface StateEvent {
  channel: "state";
  type:
    | "researcher_search"
    | "researcher_reflection"
    | "coordinator_dispatch"
    | "coordinator_results"
    | "coordinator_reflection";
  data: Record<string, unknown>;
}

export type SSEEvent = ChatEvent | ActivityEvent | StateEvent;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isReport?: boolean;
}

export interface ActivityItem {
  id: string;
  type: string;
  message: string;
  level?: "info" | "warning" | "error";
  timestamp: Date;
}

// --- State panel types ---

export interface ResearcherSearchEvent {
  topic: string;
  queries: string[];
}

export interface ResearcherReflectionEvent {
  topic: string;
  round: number;
  knowledge_state: string;
  key_findings: string[];
  missing_info: string[];
  contradictions: string[];
  should_continue: boolean;
  next_queries: string[];
}

export interface CoordinatorDispatchEvent {
  topics: string[];
  count: number;
}

export interface CoordinatorResultItem {
  topic: string;
  knowledge_state: string;
  findings_count: number;
  gaps_count: number;
  contradictions_count: number;
  key_findings: string[];
  missing_info: string[];
  contradictions: string[];
}

export interface CoordinatorResultsEvent {
  results: CoordinatorResultItem[];
  count: number;
}

export interface CoordinatorReflectionEvent {
  round: number;
  knowledge_state: string;
  overall_assessment: string;
  cross_topic_contradictions: string[];
  coverage_gaps: string[];
  should_continue: boolean;
}

/** Accumulated state for a single researcher across rounds */
export interface ResearcherState {
  topic: string;
  rounds: {
    round: number;
    queries: string[];
    knowledge_state: string;
    key_findings: string[];
    missing_info: string[];
    contradictions: string[];
  }[];
  finalState?: string;
}

/** Accumulated coordinator state */
export interface CoordinatorState {
  round: number;
  dispatches: CoordinatorDispatchEvent[];
  results: CoordinatorResultsEvent[];
  reflections: CoordinatorReflectionEvent[];
}
