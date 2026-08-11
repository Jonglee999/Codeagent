import { useCallback, useEffect, useRef, useState } from "react";
import type { TaskReport, TaskState, TaskStatus } from "../types";
import * as api from "../api/client";

const TERMINAL_STATES: TaskState[] = ["completed", "failed", "cancelled"];
const MAX_REPORT_RETRIES = 15; // 15 × 2s ≈ 30s of retries before giving up

export function useTask() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [finalReport, setFinalReport] = useState<TaskReport | null>(null);
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reportFetchedRef = useRef(false);
  const reportRetriesRef = useRef(0);
  const taskIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = null;
  }, []);

  const refresh = useCallback(async (id: string) => {
    try {
      const next = await api.getTaskStatus(id);
      if (taskIdRef.current !== id) return; // stale poll for a previous task
      setStatus(next);
      if (TERMINAL_STATES.includes(next.state)) {
        if (!reportFetchedRef.current) {
          try {
            const report = await api.getTaskReport(id);
            // A report request can outlive its task. Do not let a late response
            // overwrite the report (or stop polling) for a newer conversation turn.
            if (taskIdRef.current !== id) return;
            setFinalReport(report);
            reportFetchedRef.current = true;
            stopPolling();
          } catch {
            if (taskIdRef.current !== id) return;
            // Report may lag behind the terminal status (e.g. still being written).
            // Keep polling so a later tick retries; stop only after exhausting retries.
            reportRetriesRef.current += 1;
            if (reportRetriesRef.current >= MAX_REPORT_RETRIES) stopPolling();
          }
        } else {
          stopPolling();
        }
      }
    } catch (refreshError) {
      if (taskIdRef.current !== id) return;
      setError(refreshError instanceof Error ? refreshError.message : "Status refresh failed");
    }
  }, [stopPolling]);

  const selectTask = useCallback((id: string, projectRoot?: string) => {
    stopPolling();
    taskIdRef.current = id;
    reportFetchedRef.current = false;
    reportRetriesRef.current = 0;
    setTaskId(id);
    setStatus(null);
    setFinalReport(null);
    setError(null);
    if (projectRoot) setWorkspaceRoot(projectRoot);
    void refresh(id);
    pollingRef.current = setInterval(() => void refresh(id), 2000);
  }, [refresh, stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  const createTask = useCallback(async (
    query: string,
    options: {
      projectRoot?: string; autoMode: boolean; maxRetries: number;
      conversationHistory?: { role: "user" | "assistant"; content: string }[];
      directExecution?: boolean;
      conversationId?: string;
      benchmarkInstanceId?: string;
    },
  ) => {
    // Retire the previous run before the create request is sent.  Keeping its
    // task id/report alive while the request is in flight lets App associate a
    // late WebSocket answer from that run with the newly submitted user turn.
    stopPolling();
    taskIdRef.current = null;
    reportFetchedRef.current = false;
    reportRetriesRef.current = 0;
    setTaskId(null);
    setStatus(null);
    setFinalReport(null);
    setLoading(true);
    setError(null);
    try {
      const effectiveProjectRoot = options.projectRoot || workspaceRoot || undefined;
      const result = await api.createTask({
        query,
        ...(effectiveProjectRoot ? { project_root: effectiveProjectRoot } : {}),
        auto_mode: options.autoMode,
        max_retries: options.maxRetries,
        conversation_history: options.conversationHistory || [],
        response_mode: "auto",
        direct_execution: options.directExecution || false,
        conversation_id: options.conversationId,
        benchmark_instance_id: options.benchmarkInstanceId,
      });
      setWorkspaceRoot(result.project_root);
      selectTask(result.task_id);
      return result;
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create task");
      return null;
    } finally {
      setLoading(false);
    }
  }, [selectTask, stopPolling, workspaceRoot]);

  const resetTask = useCallback(() => {
    stopPolling();
    setTaskId(null);
    setStatus(null);
    setFinalReport(null);
    setError(null);
    setWorkspaceRoot(null);
  }, [stopPolling]);

  const cancelTask = useCallback(async () => {
    if (!taskId) return;
    try {
      await api.cancelTask(taskId);
      await refresh(taskId);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "Failed to cancel task");
    }
  }, [refresh, taskId]);

  const steerTask = useCallback(async (instruction: string) => {
    if (!taskId) return false;
    try {
      return await api.steerTask(taskId, instruction);
    } catch (steerError) {
      setError(steerError instanceof Error ? steerError.message : "Failed to steer task");
      return false;
    }
  }, [taskId]);

  const recoverTask = useCallback(async (
    instruction?: string,
    options: { autoMode?: boolean; maxRetries?: number } = {},
  ) => {
    if (!taskId) return null;
    setLoading(true);
    setError(null);
    try {
      const result = await api.recoverTask(taskId, {
        ...(instruction?.trim() ? { instruction: instruction.trim() } : {}),
        auto_mode: options.autoMode ?? true,
        max_retries: options.maxRetries ?? 3,
      });
      setWorkspaceRoot(result.project_root);
      selectTask(result.task_id, result.project_root);
      return result;
    } catch (recoverError) {
      setError(recoverError instanceof Error ? recoverError.message : "Failed to recover task");
      return null;
    } finally {
      setLoading(false);
    }
  }, [selectTask, taskId]);

  const handleDecision = useCallback(async (
    decision: "approve" | "reject" | "modify",
    modifications?: Record<string, unknown>,
  ) => {
    if (!taskId) return;
    try {
      await api.submitDecision(taskId, { decision, modifications: modifications ?? null });
    } catch (decisionError) {
      setError(decisionError instanceof Error ? decisionError.message : "Failed to submit decision");
    }
  }, [taskId]);

  return {
    taskId, status, error, loading, finalReport, workspaceRoot,
    createTask, selectTask, resetTask, cancelTask, steerTask, recoverTask, handleDecision,
  };
}
