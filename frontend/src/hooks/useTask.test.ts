import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useTask } from "./useTask";
import * as api from "../api/client";
import type { TaskReport } from "../types";

vi.mock("../api/client", () => ({
  createTask: vi.fn(),
  getTaskStatus: vi.fn(),
  getTaskReport: vi.fn(),
  cancelTask: vi.fn(),
  steerTask: vi.fn(),
  recoverTask: vi.fn(),
  submitDecision: vi.fn(),
}));

describe("useTask conversation workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getTaskStatus).mockResolvedValue({
      task_id: "task-1",
      state: "completed",
      progress: 100,
      current_step: null,
      errors: [],
    });
    vi.mocked(api.getTaskReport).mockRejectedValue(new Error("not ready"));
  });

  it("reuses the managed workspace for follow-up messages and clears it on reset", async () => {
    vi.mocked(api.createTask)
      .mockResolvedValueOnce({ task_id: "task-1", project_root: "D:/sessions/chat-1" })
      .mockResolvedValueOnce({ task_id: "task-2", project_root: "D:/sessions/chat-1" })
      .mockResolvedValueOnce({ task_id: "task-3", project_root: "D:/sessions/chat-2" });
    const { result } = renderHook(() => useTask());

    await act(async () => {
      await result.current.createTask("first", { autoMode: true, maxRetries: 3 });
    });
    await waitFor(() => expect(result.current.workspaceRoot).toBe("D:/sessions/chat-1"));

    await act(async () => {
      await result.current.createTask("follow up", { autoMode: true, maxRetries: 3 });
    });
    expect(api.createTask).toHaveBeenNthCalledWith(2, expect.objectContaining({
      project_root: "D:/sessions/chat-1",
    }));

    act(() => result.current.resetTask());
    await act(async () => {
      await result.current.createTask("new conversation", { autoMode: true, maxRetries: 3 });
    });
    expect(api.createTask).toHaveBeenNthCalledWith(3, expect.not.objectContaining({
      project_root: expect.anything(),
    }));
  });

  it("retires the previous run while a follow-up task is being created", async () => {
    let resolveCreate!: (value: { task_id: string; project_root: string }) => void;
    vi.mocked(api.createTask).mockImplementation(() => new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    vi.mocked(api.getTaskStatus).mockResolvedValue({
      task_id: "task-old",
      state: "running",
      progress: 50,
      current_step: "execution",
      errors: [],
    });
    const { result } = renderHook(() => useTask());

    act(() => result.current.selectTask("task-old", "D:/sessions/chat-1"));
    await waitFor(() => expect(result.current.taskId).toBe("task-old"));

    let creation!: Promise<unknown>;
    act(() => {
      creation = result.current.createTask("follow up", { autoMode: true, maxRetries: 3 });
    });

    expect(result.current.taskId).toBeNull();
    expect(result.current.status).toBeNull();
    expect(result.current.finalReport).toBeNull();
    expect(result.current.workspaceRoot).toBe("D:/sessions/chat-1");

    await act(async () => {
      resolveCreate({ task_id: "task-new", project_root: "D:/sessions/chat-1" });
      await creation;
    });
    expect(result.current.taskId).toBe("task-new");
  });

  it("queues steering against the active task", async () => {
    vi.mocked(api.createTask).mockResolvedValue({
      task_id: "task-1", project_root: "D:/sessions/chat-1",
    });
    vi.mocked(api.steerTask).mockResolvedValue(true);
    const { result } = renderHook(() => useTask());
    await act(async () => {
      await result.current.createTask("start", { autoMode: true, maxRetries: 3 });
    });

    let queued = false;
    await act(async () => {
      queued = await result.current.steerTask("Keep compatibility");
    });

    expect(queued).toBe(true);
    expect(api.steerTask).toHaveBeenCalledWith("task-1", "Keep compatibility");
  });

  it("starts a recovery run in the existing workspace", async () => {
    vi.mocked(api.createTask).mockResolvedValue({
      task_id: "task-1", project_root: "D:/sessions/chat-1",
    });
    vi.mocked(api.recoverTask).mockResolvedValue({
      task_id: "task-2", project_root: "D:/sessions/chat-1",
      recovered_from_task_id: "task-1",
    });
    const { result } = renderHook(() => useTask());
    await act(async () => {
      await result.current.createTask("start", { autoMode: true, maxRetries: 3 });
    });

    await act(async () => {
      await result.current.recoverTask(undefined, { autoMode: false, maxRetries: 2 });
    });

    expect(api.recoverTask).toHaveBeenCalledWith("task-1", {
      auto_mode: false,
      max_retries: 2,
    });
    expect(result.current.taskId).toBe("task-2");
    expect(result.current.workspaceRoot).toBe("D:/sessions/chat-1");
  });

  it("retries the report after a terminal status so finalReport is not lost", async () => {
    vi.useFakeTimers();
    try {
      const report: TaskReport = {
        task_id: "task-1",
        plan: [],
        changes: [{ file_path: "src/generator.py" }],
        validation_results: [],
        duration: 1.2,
        token_usage: 42,
        assistant_response: "Task completed.",
        response_mode: "execute",
        memory_hits: [],
        resolved_skills: [],
        warnings: [],
        mcp_servers: [],
        model_runtime: {},
        infrastructure_runtime: {},
      };
      // The report lags behind the completed status on the first poll, then becomes available.
      vi.mocked(api.getTaskReport)
        .mockRejectedValueOnce(new Error("report not ready yet"))
        .mockResolvedValueOnce(report);

      const { result } = renderHook(() => useTask());
      act(() => { result.current.selectTask("task-1"); });
      await act(async () => { await Promise.resolve(); });

      // First poll hit the terminal status but the report failed → retry, don't lose it.
      expect(result.current.finalReport).toBeNull();

      await act(async () => { await vi.advanceTimersByTimeAsync(2000); });

      expect(result.current.finalReport).toEqual(report);
      expect(api.getTaskReport).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("ignores a previous task report that resolves after a follow-up task", async () => {
    let resolveOldReport!: (report: TaskReport) => void;
    const oldReportPromise = new Promise<TaskReport>((resolve) => {
      resolveOldReport = resolve;
    });
    const report = (taskId: string, response: string): TaskReport => ({
      task_id: taskId,
      plan: [],
      changes: [],
      validation_results: [],
      duration: 1,
      token_usage: 1,
      assistant_response: response,
      response_mode: "chat",
      memory_hits: [],
      resolved_skills: [],
      warnings: [],
      mcp_servers: [],
      model_runtime: {},
      infrastructure_runtime: {},
    });
    vi.mocked(api.getTaskStatus).mockImplementation(async (taskId) => ({
      task_id: taskId,
      state: "completed",
      progress: 100,
      current_step: null,
      errors: [],
    }));
    vi.mocked(api.getTaskReport).mockImplementation((taskId) => (
      taskId === "task-old"
        ? oldReportPromise
        : Promise.resolve(report("task-new", "new answer"))
    ));

    const { result } = renderHook(() => useTask());
    act(() => result.current.selectTask("task-old"));
    await waitFor(() => expect(api.getTaskReport).toHaveBeenCalledWith("task-old"));

    act(() => result.current.selectTask("task-new"));
    await waitFor(() => expect(result.current.finalReport?.task_id).toBe("task-new"));

    await act(async () => {
      resolveOldReport(report("task-old", "stale greeting"));
      await oldReportPromise;
    });

    expect(result.current.taskId).toBe("task-new");
    expect(result.current.finalReport?.assistant_response).toBe("new answer");
  });
});
