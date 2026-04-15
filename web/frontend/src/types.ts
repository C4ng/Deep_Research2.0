export type Channel = "chat" | "activity";

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

export type SSEEvent = ChatEvent | ActivityEvent;

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
