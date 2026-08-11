export type TaskState =
  | "pending"
  | "running"
  | "waiting_review"
  | "completed"
  | "failed"
  | "cancelled";

export interface TaskEvent {
  event_id?: string;
  seq?: number;
  schema_version?: number;
  visibility?: "user" | "internal";
  type:
    | "node_start"
    | "node_complete"
    | "tool_call"
    | "tool_result"
    | "validation_result"
    | "human_review_required"
    | "deviation_detected"
    | "task_complete"
    | "task_error"
    | "task_cancelled"
    | "assistant_message"
    | "memory_recalled"
    | "memory_extracted"
    | "strategy_recalled"
    | "skill_resolved"
    | "transcript_saved"
    | "capability_degraded"
    | "capabilities_resolved"
    | "circuit_state_changed"
    | "permission_denied"
    | "mcp_server_status"
    | "mcp_tool_discovered"
    | "model_retry_scheduled"
    | "model_call_failed"
    | "fallback_activated"
    | "infrastructure_recovered"
    | "run_profile_selected"
    | "tools_selected"
    | "context_selected"
    | "mcp_discovery_deferred"
    | "steering_queued"
    | "steering_applied"
    | "recovery_started"
    | "ping";
  timestamp: string;
  node?: string;
  step_id?: number;
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
  scope?: string;
  duration_ms?: number;
  deviation_count?: number;
  content?: string;
}

export interface TaskCreateRequest {
  query: string;
  project_root?: string;
  auto_mode?: boolean;
  max_retries?: number;
  conversation_history?: { role: "user" | "assistant"; content: string }[];
  response_mode?: "auto" | "execute" | "chat";
  direct_execution?: boolean;
  conversation_id?: string;
  benchmark_instance_id?: string;
  recovered_from_task_id?: string;
}

export interface TaskStatus {
  task_id: string;
  state: TaskState;
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
  status?: "completed" | "failed" | "cancelled";
  recovered_from_task_id?: string | null;
  plan: unknown[];
  changes: Record<string, unknown>[];
  validation_results: Record<string, unknown>[];
  duration: number;
  token_usage: number;
  error?: string;
  assistant_response: string;
  response_mode: "chat" | "execute";
  run_profile?: Record<string, unknown>;
  tool_manifest?: Record<string, unknown>;
  context_manifest?: Record<string, unknown>;
  steering_instructions?: string[];
  memory_hits: Record<string, unknown>[];
  resolved_skills: Record<string, unknown>[];
  warnings: string[];
  reflection?: Record<string, unknown> | null;
  transcript_path?: string | null;
  mcp_servers: Record<string, unknown>[];
  model_runtime: Record<string, unknown>;
  infrastructure_runtime: Record<string, unknown>;
  benchmark_metrics?: Record<string, unknown>;
  artifacts?: {
    artifact_id: string;
    kind: string;
    size?: number;
    download_url: string;
  }[];
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T | null;
  error?: string | null;
}

export interface HistoryEntry {
  conversation_id: string;
  task_id: string;
  query: string;
  timestamp: string;
  state: string;
  workspace_root?: string;
  project_id?: string;
  messages: ChatMessage[];
}

export interface Project {
  project_id: string;
  name: string;
  created_at: string;
  workspace_root: string;
}

export interface ProjectFile {
  path: string;
  name: string;
  size: number;
  modified_at: string;
}

export interface TaskPreview {
  available: boolean;
  entrypoint: string | null;
  entries: string[];
  revision: string;
}

export interface TaskFile {
  path: string;
  name: string;
  size: number;
  modified: number;
  binary: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  task_id?: string;
}

export interface LogGroup {
  node: string;
  events: TaskEvent[];
}

export interface BenchmarkTask {
  instance_id: string;
  repo: string;
  base_commit: string;
  version: string;
  problem_statement: string;
  fail_to_pass: string[];
  pass_to_pass: string[];
  gold_patch_changed_lines: number;
  workspace_path: string;
  workspace_ready: boolean;
}

export interface BenchmarkCatalog {
  tasks: BenchmarkTask[];
  count: number;
}

export interface SystemCapabilities {
  inline_runner: boolean;
  llm_configured: boolean;
  model: string;
  model_resilience?: Record<string, unknown>;
  sandbox_enabled: boolean;
  memory_enabled: boolean;
  vector_memory_enabled: boolean;
  reflection_enabled: boolean;
  tools: string[];
  memory?: { enabled: boolean; stored_count: number; transcript_count: number; detail: string };
  skills: {
    configured: boolean; count: number; resolved_count?: number;
    active?: Record<string, unknown>[]; allowed_tools?: string[];
    warnings?: string[]; detail?: string;
  };
  mcp: {
    enabled: boolean; configured: boolean; available: boolean; detail: string;
    servers: Record<string, unknown>[]; tools?: string[]; blocked_tools?: string[]; warnings?: string[];
  };
  benchmark_tasks: number;
}
