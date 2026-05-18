import { useState, useCallback, useRef } from "react";
import type { TaskStatus } from "../types";
import * as api from "../api/client";

interface UseTaskReturn {
  taskId: string | null;
  status: TaskStatus | null;
  error: string | null;
  loading: boolean;
  createTask: (query: string, projectRoot: string, autoMode: boolean) => Promise<string | null>;
  cancelTask: () => Promise<void>;
  pollStatus: () => Promise<void>;
  handleDecision: (decision: "approve" | "reject" | "modify", modifications?: Record<string, unknown>) => Promise<void>;
  finalReport: unknown;
}

export function useTask(): UseTaskReturn {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [finalReport, setFinalReport] = useState<unknown>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = useCallback((id: string) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(async () => {
      try {
        const s = await api.getTaskStatus(id);
        setStatus(s);
        if (s.state === "completed" || s.state === "failed" || s.state === "cancelled") {
          if (pollingRef.current) clearInterval(pollingRef.current);
          if (s.state === "completed") {
            try {
              const report = await api.getTaskReport(id);
              setFinalReport(report);
            } catch { /* report may not be immediately available */ }
          }
        }
      } catch { /* poll silently */ }
    }, 2000);
  }, []);

  const createTaskFn = useCallback(
    async (query: string, projectRoot: string, autoMode: boolean): Promise<string | null> => {
      setLoading(true);
      setError(null);
      setFinalReport(null);
      try {
        const result = await api.createTask({
          query,
          project_root: projectRoot,
          auto_mode: autoMode,
        });
        setTaskId(result.task_id);
        startPolling(result.task_id);
        return result.task_id;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to create task";
        setError(msg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [startPolling],
  );

  const cancelTaskFn = useCallback(async () => {
    if (!taskId) return;
    try {
      await api.cancelTask(taskId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel task");
    }
  }, [taskId]);

  const pollStatusFn = useCallback(async () => {
    if (!taskId) return;
    try {
      const s = await api.getTaskStatus(taskId);
      setStatus(s);
    } catch { /* ignore */ }
  }, [taskId]);

  const handleDecisionFn = useCallback(
    async (decision: "approve" | "reject" | "modify", modifications?: Record<string, unknown>) => {
      if (!taskId) return;
      try {
        await api.submitDecision(taskId, { decision, modifications: modifications ?? null });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to submit decision");
      }
    },
    [taskId],
  );

  return {
    taskId,
    status,
    error,
    loading,
    createTask: createTaskFn,
    cancelTask: cancelTaskFn,
    pollStatus: pollStatusFn,
    handleDecision: handleDecisionFn,
    finalReport,
  };
}
