import type {
  ApiResponse,
  TaskCreateRequest,
  TaskStatus,
  DecisionRequest,
  TaskReport,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "";

async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<ApiResponse<T>> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body: ApiResponse<T> = await res.json();
  if (!res.ok && !body.error) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return body;
}

export async function createTask(
  req: TaskCreateRequest,
): Promise<{ task_id: string }> {
  const body = await request<{ task_id: string }>("/api/v1/tasks", {
    method: "POST",
    body: JSON.stringify(req),
  });
  if (!body.success || !body.data) {
    throw new Error(body.error || "Failed to create task");
  }
  return body.data;
}

export async function getTaskStatus(
  taskId: string,
): Promise<TaskStatus> {
  const body = await request<TaskStatus>(`/api/v1/tasks/${taskId}`);
  if (!body.success || !body.data) {
    throw new Error(body.error || "Failed to get task status");
  }
  return body.data;
}

export async function submitDecision(
  taskId: string,
  decision: DecisionRequest,
): Promise<void> {
  const body = await request<void>(`/api/v1/tasks/${taskId}/decision`, {
    method: "POST",
    body: JSON.stringify(decision),
  });
  if (!body.success) {
    throw new Error(body.error || "Failed to submit decision");
  }
}

export async function cancelTask(taskId: string): Promise<void> {
  const body = await request<void>(`/api/v1/tasks/${taskId}`, {
    method: "DELETE",
  });
  if (!body.success) {
    throw new Error(body.error || "Failed to cancel task");
  }
}

export async function getTaskReport(
  taskId: string,
): Promise<TaskReport> {
  const body = await request<TaskReport>(`/api/v1/tasks/${taskId}/report`);
  if (!body.success || !body.data) {
    throw new Error(body.error || "Failed to get task report");
  }
  return body.data;
}
