import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const taskHarness = vi.hoisted(() => ({
  createTask: vi.fn(),
  selectTask: vi.fn(),
  resetTask: vi.fn(),
  cancelTask: vi.fn(),
  steerTask: vi.fn(),
  recoverTask: vi.fn(),
  handleDecision: vi.fn(),
}));
const taskView = vi.hoisted(() => ({
  taskId: null as string | null,
  status: null as { task_id: string; state: "completed"; progress: number; current_step: null; errors: string[] } | null,
  error: null as string | null,
  loading: false,
  finalReport: null as Record<string, unknown> | null,
  workspaceRoot: null as string | null,
}));
const apiHarness = vi.hoisted(() => ({
  deleteConversation: vi.fn(),
  createProject: vi.fn(),
  getTaskPreview: vi.fn(),
  listConversations: vi.fn(),
  importConversations: vi.fn(),
}));

vi.mock("./hooks/useTask", () => ({
  useTask: () => ({
    ...taskView,
    ...taskHarness,
  }),
}));

const wsView = vi.hoisted(() => ({
  events: [] as Array<{ type: string; content?: string; error?: string; status?: string }>,
  connected: false,
  error: null as string | null,
}));

vi.mock("./hooks/useWebSocket", () => ({
  useWebSocket: () => ({ ...wsView }),
}));

vi.mock("./api/client", () => ({
  getBenchmarkCatalog: vi.fn(async () => ({ tasks: [], count: 0 })),
  getSystemCapabilities: vi.fn(async () => ({
    inline_runner: true,
    llm_configured: true,
    model: "test/model",
    sandbox_enabled: false,
    memory_enabled: true,
    vector_memory_enabled: false,
    reflection_enabled: true,
    tools: ["read_file", "write_file", "delete_file"],
    skills: { configured: true, count: 2 },
    mcp: { enabled: true, configured: false, available: false, servers: [], detail: "未配置" },
    benchmark_tasks: 30,
  })),
  prepareBenchmarkWorkspace: vi.fn(),
  listProjects: vi.fn(async () => []),
  listConversations: apiHarness.listConversations,
  importConversations: apiHarness.importConversations,
  listProjectFiles: vi.fn(async () => []),
  createProject: apiHarness.createProject,
  deleteProject: vi.fn(),
  uploadProjectFiles: vi.fn(),
  deleteProjectFile: vi.fn(),
  deleteConversation: apiHarness.deleteConversation,
  projectFileDownloadUrl: vi.fn(() => "/download"),
  getTaskPreview: apiHarness.getTaskPreview,
  taskPreviewUrl: vi.fn((taskId: string, entrypoint: string, revision?: string) => (
    `/api/v1/tasks/${taskId}/preview/${entrypoint}${revision ? `?revision=${revision}` : ""}`
  )),
}));

describe("AGENT4CODE chat shell", () => {
  beforeEach(() => {
    localStorage.clear();
    taskHarness.createTask.mockReset();
    taskHarness.createTask.mockResolvedValue({ task_id: "task-123", project_root: "workspace/task-123" });
    taskHarness.resetTask.mockReset();
    Object.assign(taskView, {
      taskId: null, status: null, error: null, loading: false,
      finalReport: null, workspaceRoot: null,
    });
    wsView.events = [];
    apiHarness.deleteConversation.mockReset();
    apiHarness.deleteConversation.mockResolvedValue({ workspace_deleted: true, reason: "deleted" });
    apiHarness.createProject.mockReset();
    apiHarness.getTaskPreview.mockReset().mockResolvedValue({
      available: false, entrypoint: null, entries: [], revision: "0",
    });
    apiHarness.listConversations.mockReset().mockResolvedValue([]);
    apiHarness.importConversations.mockReset().mockResolvedValue(0);
  });

  it("sends a query with Enter and leaves projectRoot undefined", async () => {
    const user = userEvent.setup();
    render(<App />);
    const composer = await screen.findByRole("textbox", { name: "向 AGENT4CODE 发送任务" });

    await user.type(composer, "创建一个解析器{enter}");

    await waitFor(() => expect(taskHarness.createTask).toHaveBeenCalledTimes(1));
    expect(taskHarness.createTask).toHaveBeenCalledWith("创建一个解析器", {
      projectRoot: undefined,
      autoMode: true,
      maxRetries: 3,
      conversationHistory: [],
      directExecution: false,
      conversationId: expect.any(String),
    });
  });

  it("opens functional catalog and environment panels", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("说出需求，剩下的交给 Agent。");

    await user.click(screen.getByRole("button", { name: /SWE 任务集/ }));
    expect(screen.getByRole("dialog", { name: "选择 Smoke 任务" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "关闭任务集" }));

    await user.click(screen.getByRole("button", { name: "环境就绪" }));
    expect(screen.getByRole("dialog", { name: "运行环境" })).toBeInTheDocument();
  });

  it("clears the composer when starting a new conversation", async () => {
    const user = userEvent.setup();
    render(<App />);
    const composer = await screen.findByRole("textbox", { name: "向 AGENT4CODE 发送任务" });
    await user.type(composer, "尚未发送的任务");

    await user.click(screen.getByRole("button", { name: /新建会话/ }));

    expect(composer).toHaveValue("");
    expect(taskHarness.resetTask).toHaveBeenCalledOnce();
  });

  it("deletes a stored conversation from the sidebar", async () => {
    const storedConversation = {
      conversation_id: "conversation-1",
      task_id: "task-1",
      query: "待删除会话",
      timestamp: new Date().toISOString(),
      state: "completed",
      messages: [{
        id: "message-1", role: "user", content: "待删除会话",
        timestamp: new Date().toISOString(), task_id: "task-1",
      }],
    };
    localStorage.setItem("codeagent_task_history", JSON.stringify([storedConversation]));
    apiHarness.listConversations.mockResolvedValue([storedConversation]);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "删除会话：待删除会话" }));

    await waitFor(() => expect(apiHarness.deleteConversation).toHaveBeenCalledWith("conversation-1", undefined));
    expect(screen.queryByText("待删除会话")).not.toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("codeagent_task_history") || "[]")).toEqual([]);
  });

  it("creates and opens a durable project from the sidebar", async () => {
    apiHarness.createProject.mockResolvedValue({
      project_id: "abc123",
      name: "发布说明",
      created_at: new Date().toISOString(),
      workspace_root: "D:/projects/project-abc123",
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click((await screen.findAllByRole("button", { name: "新建项目" })).at(-1)!);
    await user.type(screen.getByRole("textbox", { name: "项目名称" }), "发布说明");
    await user.click(screen.getByRole("button", { name: "创建项目" }));

    await waitFor(() => expect(apiHarness.createProject).toHaveBeenCalledWith("发布说明"));
    expect(await screen.findByRole("complementary", { name: "项目：发布说明" })).toBeInTheDocument();

    await user.type(screen.getByRole("textbox", { name: "向 AGENT4CODE 发送任务" }), "生成 changelog{enter}");
    await waitFor(() => expect(taskHarness.createTask).toHaveBeenCalledWith(
      "生成 changelog",
      expect.objectContaining({ projectRoot: "D:/projects/project-abc123" }),
    ));
  });

  it("renders one execution card that auto-expands on a completed task", async () => {
    Object.assign(taskView, {
      taskId: "task-1",
      status: { task_id: "task-1", state: "completed", progress: 100, current_step: null, errors: [] },
      finalReport: {
        task_id: "task-1", plan: [], changes: [], validation_results: [], duration: 1,
        token_usage: 10, assistant_response: "唯一回答", response_mode: "chat",
        memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
        model_runtime: {}, infrastructure_runtime: {},
      },
      workspaceRoot: "workspace/task-1",
    });
    const storedConversation = {
      conversation_id: "conversation-1", task_id: "task-1", query: "你好",
      timestamp: new Date().toISOString(), state: "completed", workspace_root: "workspace/task-1",
      messages: [
        { id: "user-1", role: "user", content: "你好", timestamp: new Date().toISOString(), task_id: "task-1" },
        { id: "assistant-1", role: "assistant", content: "唯一回答", timestamp: new Date().toISOString(), task_id: "task-1" },
      ],
    };
    localStorage.setItem("codeagent_task_history", JSON.stringify([storedConversation]));
    apiHarness.listConversations.mockResolvedValue([storedConversation]);
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(container.querySelector<HTMLButtonElement>(".conversation-select")!);

    await waitFor(() => expect(container.querySelectorAll(".thinking-card")).toHaveLength(1));
    // Completed tasks expand by default so the report and generated files are visible.
    await waitFor(() => expect(container.querySelector(".thinking-toggle")).toHaveAttribute("aria-expanded", "true"));
    expect(container.querySelector(".report-panel")).toBeInTheDocument();
    expect(screen.getAllByText("唯一回答")).toHaveLength(1);

    // The user can still collapse the execution details.
    await user.click(container.querySelector<HTMLButtonElement>(".thinking-toggle")!);
    expect(container.querySelector(".report-panel")).not.toBeInTheDocument();
  });

  it("shows the agent's answer from the live stream when the report is unavailable", async () => {
    Object.assign(taskView, {
      taskId: "task-live",
      status: { task_id: "task-live", state: "completed", progress: 100, current_step: null, errors: [] },
      finalReport: null,
      workspaceRoot: "workspace/task-live",
    });
    wsView.events = [
      { type: "assistant_message", content: "已完成，改动集中在 main.py。" },
      { type: "task_complete", status: "success" },
    ];
    const storedConversation = {
      conversation_id: "conversation-live", task_id: "task-live", query: "写一个脚本",
      timestamp: new Date().toISOString(), state: "completed", workspace_root: "workspace/task-live",
      messages: [{
        id: "user-1", role: "user", content: "写一个脚本",
        timestamp: new Date().toISOString(), task_id: "task-live",
      }],
    };
    localStorage.setItem("codeagent_task_history", JSON.stringify([storedConversation]));
    apiHarness.listConversations.mockResolvedValue([storedConversation]);
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(container.querySelector<HTMLButtonElement>(".conversation-select")!);

    expect(await screen.findByText("已完成，改动集中在 main.py。")).toBeInTheDocument();
  });

  it("does not attach a late report from the previous turn to the current turn", async () => {
    Object.assign(taskView, {
      taskId: "task-new",
      status: { task_id: "task-new", state: "completed", progress: 100, current_step: null, errors: [] },
      finalReport: {
        task_id: "task-old", plan: [], changes: [], validation_results: [], duration: 1,
        token_usage: 1, assistant_response: "stale greeting", response_mode: "chat",
        memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
        model_runtime: {}, infrastructure_runtime: {},
      },
      workspaceRoot: "workspace/shared",
    });
    const storedConversation = {
      conversation_id: "conversation-multi-turn", task_id: "task-new", query: "hello",
      timestamp: new Date().toISOString(), state: "completed", workspace_root: "workspace/shared",
      messages: [
        { id: "user-old", role: "user", content: "hello", timestamp: new Date().toISOString(), task_id: "task-old" },
        { id: "assistant-old", role: "assistant", content: "first answer", timestamp: new Date().toISOString(), task_id: "task-old" },
        { id: "user-new", role: "user", content: "create a generator", timestamp: new Date().toISOString(), task_id: "task-new" },
      ],
    };
    localStorage.setItem("codeagent_task_history", JSON.stringify([storedConversation]));
    apiHarness.listConversations.mockResolvedValue([storedConversation]);
    const user = userEvent.setup();
    const view = render(<App />);

    await user.click(view.container.querySelector<HTMLButtonElement>(".conversation-select")!);
    await waitFor(() => expect(screen.getByText("create a generator")).toBeInTheDocument());
    expect(screen.queryByText("stale greeting")).not.toBeInTheDocument();

    taskView.finalReport = {
      ...taskView.finalReport,
      task_id: "task-new",
      assistant_response: "generator created",
    };
    view.rerender(<App />);

    expect(await screen.findByText("generator created")).toBeInTheDocument();
    expect(screen.queryByText("stale greeting")).not.toBeInTheDocument();
  });

  it("synthesizes a terminal answer when a failed task has no report text", async () => {
    Object.assign(taskView, {
      taskId: "task-fail",
      status: { task_id: "task-fail", state: "failed", progress: 100, current_step: null, errors: ["validation failed"] },
      finalReport: {
        task_id: "task-fail", plan: [], changes: [], validation_results: [], duration: 2,
        token_usage: 0, assistant_response: "", response_mode: "execute",
        memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
        model_runtime: {}, infrastructure_runtime: {},
      },
      workspaceRoot: "workspace/task-fail",
    });
    wsView.events = [{ type: "task_error", error: "validation failed" }];
    const storedConversation = {
      conversation_id: "conversation-fail", task_id: "task-fail", query: "写一个解析器",
      timestamp: new Date().toISOString(), state: "failed", workspace_root: "workspace/task-fail",
      messages: [{
        id: "user-1", role: "user", content: "写一个解析器",
        timestamp: new Date().toISOString(), task_id: "task-fail",
      }],
    };
    localStorage.setItem("codeagent_task_history", JSON.stringify([storedConversation]));
    apiHarness.listConversations.mockResolvedValue([storedConversation]);
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(container.querySelector<HTMLButtonElement>(".conversation-select")!);

    expect(await screen.findByText(/任务未能完成：validation failed/)).toBeInTheDocument();
  });

  it("automatically opens a real HTML preview for the current task", async () => {
    Object.assign(taskView, {
      taskId: "task-preview",
      status: { task_id: "task-preview", state: "completed", progress: 1, current_step: null, errors: [] },
      workspaceRoot: "workspace/task-preview",
    });
    apiHarness.getTaskPreview.mockResolvedValue({
      available: true,
      entrypoint: "index.html",
      entries: ["index.html"],
      revision: "42",
    });

    render(<App />);

    expect(await screen.findByRole("complementary", { name: "HTML 实时预览" })).toBeInTheDocument();
    expect(screen.getByTitle("实时预览：index.html")).toHaveAttribute(
      "src", "/api/v1/tasks/task-preview/preview/index.html?revision=42",
    );
  });

  it("keeps an explicitly closed preview closed for follow-up tasks", async () => {
    Object.assign(taskView, {
      taskId: "task-preview-1",
      status: { task_id: "task-preview-1", state: "completed", progress: 1, current_step: null, errors: [] },
      workspaceRoot: "workspace/shared-preview",
    });
    apiHarness.getTaskPreview.mockResolvedValue({
      available: true,
      entrypoint: "index.html",
      entries: ["index.html"],
      revision: "42",
    });
    const user = userEvent.setup();
    const { rerender } = render(<App />);

    await user.click(await screen.findByRole("button", { name: "关闭页面预览" }));
    expect(screen.queryByRole("complementary", { name: "HTML 实时预览" })).not.toBeInTheDocument();

    apiHarness.getTaskPreview.mockClear();
    Object.assign(taskView, {
      taskId: "task-preview-2",
      status: { task_id: "task-preview-2", state: "completed", progress: 1, current_step: null, errors: [] },
    });
    rerender(<App />);

    await waitFor(() => expect(apiHarness.getTaskPreview).toHaveBeenCalledWith("task-preview-2"));
    expect(screen.queryByRole("complementary", { name: "HTML 实时预览" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "页面预览" }));
    expect(screen.getByRole("complementary", { name: "HTML 实时预览" })).toBeInTheDocument();
  });
});
