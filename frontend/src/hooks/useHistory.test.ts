import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useHistory } from "./useHistory";
import type { HistoryEntry } from "../types";

const api = vi.hoisted(() => ({
  importConversations: vi.fn(),
  listConversations: vi.fn(),
}));

vi.mock("../api/client", () => api);

const entry: HistoryEntry = {
  conversation_id: "conversation-1",
  task_id: "task-1",
  query: "Fix it",
  timestamp: "2026-08-07T00:00:00Z",
  state: "completed",
  workspace_root: "D:/workspace/conversation-1",
  messages: [{
    id: "message-1",
    role: "user",
    content: "Fix it",
    timestamp: "2026-08-07T00:00:00Z",
    task_id: "task-1",
  }],
};

describe("useHistory", () => {
  beforeEach(() => {
    localStorage.clear();
    api.importConversations.mockReset().mockResolvedValue(1);
    api.listConversations.mockReset().mockResolvedValue([entry]);
  });

  it("imports legacy history once and then hydrates from the server", async () => {
    localStorage.setItem("codeagent_task_history", JSON.stringify([entry]));
    const { result } = renderHook(() => useHistory());

    await waitFor(() => expect(api.listConversations).toHaveBeenCalledOnce());
    expect(api.importConversations).toHaveBeenCalledWith([entry]);
    expect(result.current.history).toEqual([entry]);
    expect(localStorage.getItem("codeagent_task_history")).toBeNull();
  });

  it("keeps optimistic state in memory without recreating local storage", async () => {
    const { result } = renderHook(() => useHistory());
    await waitFor(() => expect(api.listConversations).toHaveBeenCalledOnce());

    act(() => result.current.addEntry(entry));
    expect(result.current.history).toEqual([entry]);
    expect(localStorage.getItem("codeagent_task_history")).toBeNull();

    act(() => result.current.removeEntry(entry.conversation_id));
    expect(result.current.history).toEqual([]);
  });
});
