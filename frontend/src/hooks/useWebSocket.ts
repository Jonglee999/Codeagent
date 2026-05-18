import { useEffect, useRef, useState, useCallback } from "react";
import type { TaskEvent } from "../types";

// vite dev proxy 会把 /api 的 ws 请求转发到后端，生产环境用同域
const WS_BASE = import.meta.env.VITE_WS_URL || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
const MAX_RECONNECT_ATTEMPTS = 5;

interface WebSocketState {
  events: TaskEvent[];
  connected: boolean;
  error: string | null;
}

export function useWebSocket(taskId: string | null): WebSocketState {
  const [state, setState] = useState<WebSocketState>({
    events: [],
    connected: false,
    error: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const retryCount = useRef(0);
  const doneRef = useRef(false);
  const mountedRef = useRef(true);
  // 用于去重：key = "type-timestamp"
  const seenKeysRef = useRef<Set<string>>(new Set());

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      doneRef.current = true;
      cleanup();
    };
  }, [cleanup]);

  useEffect(() => {
    if (!taskId) return;

    doneRef.current = false;
    retryCount.current = 0;
    seenKeysRef.current = new Set();
    setState({ events: [], connected: false, error: null });

    function connect() {
      if (!mountedRef.current || doneRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/api/v1/tasks/${taskId}/stream`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        retryCount.current = 0;
        setState((prev) => ({ ...prev, connected: true, error: null }));
      };

      ws.onmessage = (msg) => {
        if (!mountedRef.current) return;
        try {
          const event: TaskEvent = JSON.parse(msg.data);
          if (event.type === "ping") return;

          // 去重：历史回放和实时推送可能重叠
          const key = `${event.type}-${event.timestamp}`;
          if (seenKeysRef.current.has(key)) return;
          seenKeysRef.current.add(key);

          setState((prev) => ({ ...prev, events: [...prev.events, event] }));

          if (event.type === "task_complete" || event.type === "task_error") {
            doneRef.current = true;
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setState((prev) => ({ ...prev, error: "WebSocket connection error" }));
      };

      ws.onclose = (e) => {
        if (!mountedRef.current) return;
        setState((prev) => ({ ...prev, connected: false }));

        if (!doneRef.current && retryCount.current < MAX_RECONNECT_ATTEMPTS) {
          retryCount.current += 1;
          const delay = Math.min(1000 * 2 ** (retryCount.current - 1), 16000);
          setTimeout(connect, delay);
        } else if (!doneRef.current) {
          setState((prev) => ({
            ...prev,
            error: `WebSocket closed (code ${e.code}). Reconnect exhausted.`,
          }));
        }
      };
    }

    // 立即连接，后端历史回放保证不丢消息
    connect();

    return () => {
      cleanup();
    };
  }, [taskId, cleanup]);

  return state;
}
