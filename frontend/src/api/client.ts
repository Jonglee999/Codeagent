import type {
  ApiResponse,
  BenchmarkCatalog,
  DecisionRequest,
  Project,
  ProjectFile,
  SystemCapabilities,
  TaskCreateRequest,
  TaskReport,
  TaskStatus,
  TaskPreview,
  TaskFile,
  HistoryEntry,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "";

async function request<T>(url: string, options?: RequestInit): Promise<ApiResponse<T>> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${url}`, {
      ...(!(options?.body instanceof FormData) ? { headers: { "Content-Type": "application/json" } } : {}),
      ...options,
    });
  } catch (error) {
    throw new Error(error instanceof Error ? `API unavailable: ${error.message}` : "API unavailable", { cause: error });
  }

  const text = await response.text();
  let body: ApiResponse<T> & { detail?: string };
  try {
    body = text ? JSON.parse(text) : { success: response.ok };
  } catch {
    throw new Error(`HTTP ${response.status}: server returned an invalid response`);
  }
  if (!response.ok) {
    throw new Error(body.error || body.detail || `HTTP ${response.status}: ${response.statusText}`);
  }
  return body;
}

export async function createTask(req: TaskCreateRequest): Promise<{ task_id: string; project_root: string; conversation_id?: string }> {
  const body = await request<{ task_id: string; project_root: string; conversation_id?: string }>("/api/v1/tasks", {
    method: "POST",
    body: JSON.stringify(req),
  });
  if (!body.success || !body.data) throw new Error(body.error || "Failed to create task");
  return body.data;
}

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const body = await request<TaskStatus>(`/api/v1/tasks/${taskId}`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to get task status");
  return body.data;
}

export async function submitDecision(taskId: string, decision: DecisionRequest): Promise<void> {
  const body = await request<void>(`/api/v1/tasks/${taskId}/decision`, {
    method: "POST",
    body: JSON.stringify(decision),
  });
  if (!body.success) throw new Error(body.error || "Failed to submit decision");
}

export async function cancelTask(taskId: string): Promise<void> {
  const body = await request<void>(`/api/v1/tasks/${taskId}`, { method: "DELETE" });
  if (!body.success) throw new Error(body.error || "Failed to cancel task");
}

export async function steerTask(taskId: string, instruction: string): Promise<boolean> {
  const body = await request<{ queued: boolean }>(`/api/v1/tasks/${taskId}/steer`, {
    method: "POST",
    body: JSON.stringify({ instruction }),
  });
  if (!body.success || !body.data) throw new Error(body.error || "Failed to steer task");
  return body.data.queued;
}

export async function recoverTask(
  taskId: string,
  options: { instruction?: string; auto_mode?: boolean; max_retries?: number } = {},
): Promise<{ task_id: string; project_root: string; conversation_id?: string; recovered_from_task_id: string }> {
  const body = await request<{
    task_id: string; project_root: string; conversation_id?: string; recovered_from_task_id: string;
  }>(`/api/v1/tasks/${encodeURIComponent(taskId)}/recover`, {
    method: "POST",
    body: JSON.stringify(options),
  });
  if (!body.success || !body.data) throw new Error(body.error || "Failed to recover task");
  return body.data;
}

export async function getTaskReport(taskId: string): Promise<TaskReport> {
  const body = await request<TaskReport>(`/api/v1/tasks/${taskId}/report`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to get task report");
  return body.data;
}

export async function getTaskPreview(taskId: string): Promise<TaskPreview> {
  const body = await request<TaskPreview>(`/api/v1/tasks/${encodeURIComponent(taskId)}/preview`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to load task preview");
  return body.data;
}

export async function listTaskFiles(taskId: string): Promise<TaskFile[]> {
  const body = await request<{ files: TaskFile[] }>(`/api/v1/tasks/${encodeURIComponent(taskId)}/files`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to list task files");
  return body.data.files;
}

export function taskWorkspaceFileUrl(taskId: string, path: string): string {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE}/api/v1/tasks/${encodeURIComponent(taskId)}/files/${encodedPath}`;
}

export function taskPreviewUrl(taskId: string, entrypoint: string, revision?: string): string {
  const encodedPath = entrypoint.split("/").map(encodeURIComponent).join("/");
  const params = revision ? `?revision=${encodeURIComponent(revision)}` : "";
  return `${API_BASE}/api/v1/tasks/${encodeURIComponent(taskId)}/preview/${encodedPath}${params}`;
}

export async function getBenchmarkCatalog(): Promise<BenchmarkCatalog> {
  const body = await request<BenchmarkCatalog>("/api/v1/catalog/tasks");
  if (!body.success || !body.data) throw new Error(body.error || "Failed to load task catalog");
  return body.data;
}

export async function prepareBenchmarkWorkspace(instanceId: string): Promise<{ project_root: string }> {
  const body = await request<{ project_root: string }>(
    `/api/v1/catalog/tasks/${encodeURIComponent(instanceId)}/prepare`,
    { method: "POST" },
  );
  if (!body.success || !body.data) throw new Error(body.error || "Failed to prepare workspace");
  return body.data;
}

export async function getSystemCapabilities(projectRoot?: string, query?: string): Promise<SystemCapabilities> {
  const params = new URLSearchParams();
  if (projectRoot) params.set("project_root", projectRoot);
  if (query) params.set("query", query);
  const suffix = params.size ? `?${params.toString()}` : "";
  const body = await request<SystemCapabilities>(`/api/v1/system/capabilities${suffix}`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to load capabilities");
  return body.data;
}

export async function listProjects(): Promise<Project[]> {
  const body = await request<{ projects: Project[] }>("/api/v1/projects");
  if (!body.success || !body.data) throw new Error(body.error || "Failed to list projects");
  return body.data.projects;
}

export async function createProject(name: string): Promise<Project> {
  const body = await request<Project>("/api/v1/projects", {
    method: "POST", body: JSON.stringify({ name }),
  });
  if (!body.success || !body.data) throw new Error(body.error || "Failed to create project");
  return body.data;
}

export async function deleteProject(projectId: string): Promise<void> {
  const body = await request(`/api/v1/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" });
  if (!body.success) throw new Error(body.error || "Failed to delete project");
}

export async function listProjectFiles(projectId: string): Promise<ProjectFile[]> {
  const body = await request<{ files: ProjectFile[] }>(`/api/v1/projects/${encodeURIComponent(projectId)}/files`);
  if (!body.success || !body.data) throw new Error(body.error || "Failed to list files");
  return body.data.files;
}

export async function uploadProjectFiles(projectId: string, files: File[]): Promise<void> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const body = await request(`/api/v1/projects/${encodeURIComponent(projectId)}/files`, {
    method: "POST", body: form,
  });
  if (!body.success) throw new Error(body.error || "Failed to upload files");
}

export function projectFileDownloadUrl(projectId: string, path: string): string {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE}/api/v1/projects/${encodeURIComponent(projectId)}/download/${encodedPath}`;
}

export async function deleteProjectFile(projectId: string, path: string): Promise<void> {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  const body = await request(`/api/v1/projects/${encodeURIComponent(projectId)}/files/${encodedPath}`, { method: "DELETE" });
  if (!body.success) throw new Error(body.error || "Failed to delete file");
}

export async function deleteConversation(conversationId: string, workspaceRoot?: string): Promise<{ workspace_deleted: boolean; history_deleted: boolean; reason: string }> {
  const body = await request<{ workspace_deleted: boolean; history_deleted: boolean; reason: string }>(
    `/api/v1/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE", body: JSON.stringify({ workspace_root: workspaceRoot || null }) },
  );
  if (!body.success || !body.data) throw new Error(body.error || "Failed to delete conversation");
  return body.data;
}

export async function listConversations(): Promise<HistoryEntry[]> {
  const body = await request<{ conversations: HistoryEntry[] }>("/api/v1/conversations");
  if (!body.success || !body.data) throw new Error(body.error || "Failed to list conversations");
  return body.data.conversations;
}

export async function importConversations(conversations: HistoryEntry[]): Promise<number> {
  const body = await request<{ imported: number }>("/api/v1/conversations/import", {
    method: "POST",
    body: JSON.stringify({ conversations }),
  });
  if (!body.success || !body.data) throw new Error(body.error || "Failed to import conversations");
  return body.data.imported;
}
