/* ── WebSocket 事件类型 ── */

export interface TaskEvent {
  type:
    | "node_start"
    | "tool_call"
    | "tool_result"
    | "validation_result"
    | "human_review_required"
    | "task_complete"
    | "task_error"
    | "ping";
  timestamp: string;
  node?: string;
  tool?: string;
  params?: Record<string, unknown>;
  success?: boolean;
  summary?: string;
  layer?: string;
  passed?: boolean;
  errors?: string[];
  review_type?: string;
  details?: Record<string, unknown>;
  status?: string;
  duration?: number;
  token_usage?: number;
  error?: string;
  data?: Record<string, unknown>;
}

/* ── REST API 类型 ── */

export interface TaskCreateRequest {
  query: string;
  project_root: string;
  auto_mode?: boolean;
  max_retries?: number;
}

export interface TaskStatus {
  task_id: string;
  state: "pending" | "running" | "waiting_review" | "completed" | "failed" | "cancelled";
  progress: number;
  current_step: string | null;
  errors: string[];
}

export interface DecisionRequest {
  decision: "approve" | "reject" | "modify";
  modifications?: Record<string, unknown> | null;
}

export interface TaskReport {
  task_id: string;
  plan: unknown[];
  changes: Record<string, unknown>[];
  validation_results: unknown[];
  duration: number;
  token_usage: number;
  error?: string;
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T | null;
  error?: string | null;
}

/* ── 本地历史记录类型 ── */

export interface HistoryEntry {
  task_id: string;
  query: string;
  timestamp: string;
  state: string;
}

/* ── 分组后的日志条目 ── */

export interface LogGroup {
  node: string;
  events: TaskEvent[];
}
