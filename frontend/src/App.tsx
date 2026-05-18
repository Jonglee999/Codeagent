import { useState, useCallback, useRef, useEffect } from "react";
import type { TaskEvent, TaskCreateRequest } from "./types";
import { useTask } from "./hooks/useTask";
import { useWebSocket } from "./hooks/useWebSocket";
import { useHistory } from "./hooks/useHistory";
import TaskInput from "./components/TaskInput";
import ExecutionLog from "./components/ExecutionLog";
import HumanReviewModal from "./components/HumanReviewModal";
import TaskHistory from "./components/TaskHistory";
import ProgressStepper from "./components/ProgressStepper";
import ReportPanel from "./components/ReportPanel";

export default function App() {
  const [showReview, setShowReview] = useState(false);
  const reviewEventRef = useRef<TaskEvent | null>(null);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  const { history, addEntry, updateState } = useHistory();
  const taskActions = useTask();
  const prevLenRef = useRef(0);

  useEffect(() => {
    prevLenRef.current = 0;
  }, [currentTaskId]);

  const ws = useWebSocket(currentTaskId);

  useEffect(() => {
    if (ws.events.length <= prevLenRef.current) return;
    for (let i = prevLenRef.current; i < ws.events.length; i++) {
      const event = ws.events[i];
      if (event.type === "task_complete" && currentTaskId) {
        updateState(currentTaskId, "completed");
        setTimeout(() => setCurrentTaskId(null), 5000);
      }
      if (event.type === "task_error" && currentTaskId) {
        updateState(currentTaskId, "failed");
        setTimeout(() => setCurrentTaskId(null), 5000);
      }
    }
    prevLenRef.current = ws.events.length;
  }, [ws.events, currentTaskId, updateState]);

  useEffect(() => {
    if (!taskActions.status || !currentTaskId) return;
    const s = taskActions.status.state;
    if (s === "completed" || s === "failed" || s === "cancelled") {
      updateState(currentTaskId, s);
    }
  }, [taskActions.status, currentTaskId, updateState]);

  const handleSubmit = useCallback(
    async (req: TaskCreateRequest) => {
      reviewEventRef.current = null;
      setShowReview(false);
      const id = await taskActions.createTask(req.query, req.project_root, req.auto_mode ?? false);
      if (id) {
        setCurrentTaskId(id);
        addEntry({ task_id: id, query: req.query, timestamp: new Date().toISOString(), state: "pending" });
      }
    },
    [taskActions, addEntry],
  );

  const handleHumanReview = useCallback((event: TaskEvent) => {
    reviewEventRef.current = event;
    setShowReview(true);
  }, []);

  const handleDecision = useCallback(
    async (decision: "approve" | "reject" | "modify", feedback?: string) => {
      if (!currentTaskId) return;
      setShowReview(false);
      if (decision === "modify" && feedback) {
        await taskActions.handleDecision("modify", { feedback });
      } else {
        await taskActions.handleDecision(decision);
      }
    },
    [currentTaskId, taskActions],
  );

  const handleHistorySelect = useCallback((taskId: string) => {
    setCurrentTaskId(null);
    reviewEventRef.current = null;
    setShowReview(false);
    setTimeout(() => setCurrentTaskId(taskId), 200);
  }, []);

  const handleCancel = useCallback(async () => {
    await taskActions.cancelTask();
    if (currentTaskId) updateState(currentTaskId, "cancelled");
    setCurrentTaskId(null);
  }, [taskActions, currentTaskId, updateState]);

  const taskState = taskActions.status?.state ?? (currentTaskId ? "pending" : null);
  const isRunning = taskState === "running" || taskState === "pending";

  // 找最后一个 task_complete 事件用于报告面板
  const completeEvent = ws.events.findLast?.((e) => e.type === "task_complete") ?? null;

  return (
    <div className="min-h-screen bg-[#0f1117] text-slate-200">
      {/* Header */}
      <header className="border-b border-[#2a2d3a] bg-[#1a1d27]">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-indigo-500 flex items-center justify-center text-white font-bold text-sm">C</div>
            <div>
              <span className="font-semibold text-slate-100 text-sm">CodeAgent</span>
              <span className="ml-2 text-xs text-slate-500">AI Coding Assistant</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {taskState && (
              <span className={`text-xs px-2.5 py-1 rounded-full font-medium border ${
                taskState === "completed" ? "bg-green-500/10 text-green-400 border-green-500/20"
                : taskState === "failed"  ? "bg-red-500/10 text-red-400 border-red-500/20"
                : taskState === "running" ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/20 animate-pulse"
                : taskState === "waiting_review" ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                : "bg-slate-500/10 text-slate-400 border-slate-500/20"
              }`}>
                {taskState}
              </span>
            )}
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${ws.connected ? "bg-green-400" : "bg-slate-600"}`} />
              <span className="text-xs text-slate-500">{ws.connected ? "connected" : "offline"}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-6 py-5">
        <div className="flex gap-5">
          {/* Left panel */}
          <div className="w-80 shrink-0 space-y-4">
            {/* Task input */}
            <div className="bg-[#1a1d27] border border-[#2a2d3a] rounded-xl p-4">
              <TaskInput
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                loading={taskActions.loading}
                running={isRunning && !!currentTaskId}
              />
              {taskActions.error && (
                <p className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                  {taskActions.error}
                </p>
              )}
              {ws.error && (
                <p className="mt-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                  {ws.error}
                </p>
              )}
            </div>

            {/* Progress stepper */}
            {(isRunning || taskState === "completed" || taskState === "failed") && (
              <div className="bg-[#1a1d27] border border-[#2a2d3a] rounded-xl p-4">
                <ProgressStepper events={ws.events} taskState={taskState} />
              </div>
            )}

            {/* History */}
            <div className="bg-[#1a1d27] border border-[#2a2d3a] rounded-xl p-4">
              <TaskHistory history={history} onSelect={handleHistorySelect} />
            </div>
          </div>

          {/* Right panel */}
          <div className="flex-1 min-w-0 space-y-4">
            {/* Execution log header */}
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Execution Log</h2>
              {currentTaskId && (
                <span className="text-xs text-slate-600 font-mono">{currentTaskId.slice(0, 8)}…</span>
              )}
            </div>

            <ExecutionLog
              events={ws.events}
              onHumanReview={handleHumanReview}
              isRunning={isRunning && !!currentTaskId}
            />

            {/* Report panel — shown after completion */}
            {(taskState === "completed" || taskState === "failed") && (
              <ReportPanel
                report={taskActions.finalReport}
                completeEvent={completeEvent}
              />
            )}
          </div>
        </div>
      </main>

      {/* Human Review Modal */}
      {showReview && (
        <HumanReviewModal
          event={reviewEventRef.current}
          onDecision={handleDecision}
          onClose={() => setShowReview(false)}
        />
      )}
    </div>
  );
}
