import { afterEach, describe, expect, it, vi } from "vitest";

import { createTask, importConversations, listConversations, recoverTask, steerTask, taskPreviewUrl } from "./client";

describe("task API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("omits project_root for chat-first tasks", async () => {
    let capturedOptions: RequestInit | undefined;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, options?: RequestInit) => {
      capturedOptions = options;
      return new Response(JSON.stringify({
        success: true,
        data: { task_id: "task-1", project_root: "D:/workspaces/chat-1" },
      }), { status: 202, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await createTask({ query: "build it", auto_mode: true });

    expect(JSON.parse(String(capturedOptions?.body))).toEqual({ query: "build it", auto_mode: true });
    expect(result.project_root).toBe("D:/workspaces/chat-1");
  });

  it("lists server-owned conversations", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      success: true,
      data: { conversations: [{ conversation_id: "conversation-1" }] },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    expect(await listConversations()).toEqual([{ conversation_id: "conversation-1" }]);
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/conversations", expect.any(Object));
  });

  it("imports legacy conversations through the migration endpoint", async () => {
    let capturedOptions: RequestInit | undefined;
    vi.stubGlobal("fetch", vi.fn(async (_input, options?: RequestInit) => {
      capturedOptions = options;
      return new Response(JSON.stringify({ success: true, data: { imported: 1 } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }));

    const imported = await importConversations([{
      conversation_id: "conversation-1",
      task_id: "task-1",
      query: "Fix it",
      timestamp: "2026-08-07T00:00:00Z",
      state: "completed",
      messages: [],
    }]);

    expect(imported).toBe(1);
    expect(JSON.parse(String(capturedOptions?.body)).conversations).toHaveLength(1);
  });

  it("builds an encoded cache-busted task preview URL", () => {
    expect(taskPreviewUrl("task 1", "pages/演示.html", "42")).toBe(
      "/api/v1/tasks/task%201/preview/pages/%E6%BC%94%E7%A4%BA.html?revision=42",
    );
  });

  it("queues a bounded runtime steering instruction", async () => {
    let capturedOptions: RequestInit | undefined;
    vi.stubGlobal("fetch", vi.fn(async (_input, options?: RequestInit) => {
      capturedOptions = options;
      return new Response(JSON.stringify({ success: true, data: { queued: true } }), {
        status: 200, headers: { "Content-Type": "application/json" },
      });
    }));

    expect(await steerTask("task-1", "Keep compatibility")).toBe(true);
    expect(JSON.parse(String(capturedOptions?.body))).toEqual({
      instruction: "Keep compatibility",
    });
  });

  it("starts recovery from a terminal task", async () => {
    let capturedOptions: RequestInit | undefined;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, options?: RequestInit) => {
      capturedOptions = options;
      return new Response(JSON.stringify({
        success: true,
        data: {
          task_id: "task-2", project_root: "D:/workspace",
          recovered_from_task_id: "task-1",
        },
      }), { status: 202, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await recoverTask("task-1", { auto_mode: true, max_retries: 2 });

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/tasks/task-1/recover", expect.any(Object));
    expect(JSON.parse(String(capturedOptions?.body))).toEqual({ auto_mode: true, max_retries: 2 });
    expect(result.task_id).toBe("task-2");
  });
});
