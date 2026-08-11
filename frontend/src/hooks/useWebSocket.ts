import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TaskEvent } from "../types";

const WS_BASE = import.meta.env.VITE_WS_URL || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
const MAX_RECONNECT_ATTEMPTS = 5;

interface WebSocketState {
  taskId: string | null;
  events: TaskEvent[];
  connected: boolean;
  error: string | null;
  phase: "idle" | "connecting" | "live" | "reconnecting" | "ended";
  reconnectAttempt: number;
}

const EMPTY_STATE: WebSocketState = {
  taskId: null, events: [], connected: false, error: null,
  phase: "idle", reconnectAttempt: 0,
};

export function useWebSocket(taskId: string | null) {
  const [state, setState] = useState<WebSocketState>(EMPTY_STATE);
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const doneRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seenRef = useRef(new Set<string>());

  const cleanup = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      socket.close();
    }
  }, []);

  useEffect(() => {
    if (!taskId) { cleanup(); return; }
    let active = true;
    retryRef.current = 0;
    doneRef.current = false;
    seenRef.current = new Set();

    const connect = () => {
      if (!active || doneRef.current) return;
      setState((previous) => ({
        taskId,
        events: previous.taskId === taskId ? previous.events : [],
        connected: false,
        error: null,
        phase: retryRef.current ? "reconnecting" : "connecting",
        reconnectAttempt: retryRef.current,
      }));
      const socket = new WebSocket(`${WS_BASE}/api/v1/tasks/${taskId}/stream`);
      socketRef.current = socket;
      socket.onopen = () => {
        retryRef.current = 0;
        setState((previous) => ({
          taskId,
          events: previous.taskId === taskId ? previous.events : [],
          connected: true,
          error: null,
          phase: "live",
          reconnectAttempt: retryRef.current,
        }));
      };
      socket.onmessage = (message) => {
        try {
          const event: TaskEvent = JSON.parse(message.data);
          if (event.type === "ping") return;
          if (
            event.visibility === "internal"
            || ["memory_recalled", "memory_extracted", "strategy_recalled", "transcript_saved"].includes(event.type)
          ) return;
          const key = event.event_id || (event.seq === undefined
            ? JSON.stringify(event)
            : `seq:${event.seq}`);
          if (seenRef.current.has(key)) return;
          seenRef.current.add(key);
          const terminal = event.type === "task_complete" || event.type === "task_error" || event.type === "task_cancelled";
          setState((previous) => ({
            taskId,
            connected: true,
            error: null,
            phase: terminal ? "ended" : "live",
            reconnectAttempt: retryRef.current,
            events: [...(previous.taskId === taskId ? previous.events : []), event].slice(-2000),
          }));
          if (terminal) doneRef.current = true;
        } catch { /* ignore malformed frames */ }
      };
      socket.onerror = () => setState((previous) => ({ ...previous, taskId, error: "实时执行流暂时中断，正在尝试恢复" }));
      socket.onclose = (event) => {
        if (!active) return;
        setState((previous) => ({
          ...previous,
          taskId,
          connected: false,
          phase: doneRef.current ? "ended" : "reconnecting",
        }));
        if (!doneRef.current && retryRef.current < MAX_RECONNECT_ATTEMPTS) {
          retryRef.current += 1;
          timerRef.current = setTimeout(connect, Math.min(1000 * 2 ** (retryRef.current - 1), 16000));
        } else if (!doneRef.current) {
          setState((previous) => ({ ...previous, taskId, error: `Live stream closed (${event.code})` }));
        }
      };
    };

    connect();
    return () => { active = false; cleanup(); };
  }, [cleanup, taskId]);

  return useMemo(
    () => state.taskId === taskId ? state : { ...EMPTY_STATE, taskId },
    [state, taskId],
  );
}
