import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useWebSocket } from "./useWebSocket";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close() {}
}

describe("useWebSocket replay behavior", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    FakeWebSocket.instances = [];
  });

  it("deduplicates replayed real events by event_id", async () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const { result } = renderHook(() => useWebSocket("task-1"));
    const socket = FakeWebSocket.instances[0];

    act(() => socket.onopen?.(new Event("open")));
    const payload = JSON.stringify({
      event_id: "event-1", type: "node_start", node: "worker",
      timestamp: new Date().toISOString(),
    });
    act(() => {
      socket.onmessage?.(new MessageEvent("message", { data: payload }));
      socket.onmessage?.(new MessageEvent("message", { data: payload }));
    });

    await waitFor(() => expect(result.current.events).toHaveLength(1));
    expect(result.current.phase).toBe("live");
  });

  it("exposes reconnecting state instead of silently freezing", () => {
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const { result } = renderHook(() => useWebSocket("task-2"));
    const socket = FakeWebSocket.instances[0];

    act(() => {
      socket.onerror?.(new Event("error"));
      socket.onclose?.(new CloseEvent("close", { code: 1006 }));
    });

    expect(result.current.connected).toBe(false);
    expect(result.current.phase).toBe("reconnecting");
    expect(result.current.error).toContain("正在尝试恢复");
  });

  it("keeps internal memory activity out of the user event stream", async () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const { result } = renderHook(() => useWebSocket("task-memory"));
    const socket = FakeWebSocket.instances[0];

    act(() => socket.onopen?.(new Event("open")));
    act(() => {
      socket.onmessage?.(new MessageEvent("message", { data: JSON.stringify({
        event_id: "memory-1", type: "memory_recalled", visibility: "internal",
        timestamp: new Date().toISOString(),
      }) }));
      socket.onmessage?.(new MessageEvent("message", { data: JSON.stringify({
        event_id: "tool-1", type: "tool_call", tool: "read_file",
        timestamp: new Date().toISOString(),
      }) }));
    });

    await waitFor(() => expect(result.current.events).toHaveLength(1));
    expect(result.current.events[0].type).toBe("tool_call");
  });

  it("clears events before connecting to a different task", async () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const { result, rerender } = renderHook(
      ({ taskId }) => useWebSocket(taskId),
      { initialProps: { taskId: "task-old" } },
    );
    const oldSocket = FakeWebSocket.instances[0];

    act(() => oldSocket.onopen?.(new Event("open")));
    act(() => oldSocket.onmessage?.(new MessageEvent("message", { data: JSON.stringify({
      event_id: "old-answer",
      type: "assistant_message",
      content: "old fallback",
      timestamp: new Date().toISOString(),
    }) })));
    await waitFor(() => expect(result.current.events).toHaveLength(1));

    rerender({ taskId: "task-new" });

    await waitFor(() => expect(result.current.taskId).toBe("task-new"));
    expect(result.current.events).toEqual([]);
    expect(result.current.phase).toBe("connecting");

    const newSocket = FakeWebSocket.instances[1];
    act(() => newSocket.onopen?.(new Event("open")));
    expect(result.current.events).toEqual([]);
  });
});
