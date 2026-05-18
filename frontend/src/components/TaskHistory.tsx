import { useState } from "react";
import type { HistoryEntry } from "../types";
import * as api from "../api/client";
import type { TaskReport } from "../types";

interface TaskHistoryProps {
  history: HistoryEntry[];
  onSelect: (taskId: string) => void;
}

function StateBadge({ state }: { state: string }) {
  const cls =
    state === "completed" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
    : state === "failed"  ? "bg-red-500/10 text-red-400 border-red-500/20"
    : state === "running" ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
    : state === "cancelled" ? "bg-slate-500/10 text-slate-500 border-slate-500/20"
    : "bg-slate-500/10 text-slate-400 border-slate-500/20";
  return (
    <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded border font-medium ${cls}`}>
      {state}
    </span>
  );
}

function ReportView({ report }: { report: TaskReport }) {
  return (
    <div className="mt-3 space-y-3 animate-slide-in">
      {/* Stats */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-[#0f1117] border border-[#2a2d3a] rounded-lg p-2 text-center">
          <div className="text-lg font-mono text-indigo-400">{report.duration?.toFixed(1) ?? "—"}s</div>
          <div className="text-xs text-slate-500">Duration</div>
        </div>
        <div className="bg-[#0f1117] border border-[#2a2d3a] rounded-lg p-2 text-center">
          <div className="text-lg font-mono text-violet-400">{report.token_usage ?? "—"}</div>
          <div className="text-xs text-slate-500">Tokens</div>
        </div>
      </div>

      {/* Plan */}
      {report.plan && report.plan.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Plan Steps</div>
          <ol className="space-y-1">
            {report.plan.map((step, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                <span className="shrink-0 text-slate-600 font-mono">{i + 1}.</span>
                <span>{typeof step === "string" ? step : (step as Record<string, unknown>)?.description as string ?? JSON.stringify(step)}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Changes */}
      {report.changes && report.changes.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Changed Files</div>
          <ul className="space-y-1">
            {report.changes.map((c, i) => (
              <li key={i} className="text-xs font-mono text-slate-400 bg-[#0f1117] border border-[#2a2d3a] rounded px-2 py-1 truncate">
                {(c as Record<string, unknown>)?.file_path as string ?? JSON.stringify(c)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Error */}
      {report.error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {report.error}
        </div>
      )}
    </div>
  );
}

export default function TaskHistory({ history, onSelect }: TaskHistoryProps) {
  const [reportData, setReportData] = useState<TaskReport | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const handleClick = async (entry: HistoryEntry) => {
    onSelect(entry.task_id);
    if (openId === entry.task_id) {
      setOpenId(null);
      setReportData(null);
      return;
    }
    setOpenId(entry.task_id);
    setLoadingId(entry.task_id);
    try {
      const report = await api.getTaskReport(entry.task_id);
      setReportData(report);
    } catch {
      setReportData(null);
    } finally {
      setLoadingId(null);
    }
  };

  if (history.length === 0) {
    return (
      <div className="text-xs text-slate-600 text-center py-6">
        No task history yet.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Recent Tasks</h3>
      <ul className="space-y-1">
        {history.map((entry) => (
          <li key={entry.task_id}>
            <button
              onClick={() => handleClick(entry)}
              className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                openId === entry.task_id
                  ? "bg-indigo-500/10 border border-indigo-500/20"
                  : "hover:bg-white/5 border border-transparent"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-slate-300 truncate flex-1">
                  {entry.query.length > 38 ? entry.query.slice(0, 38) + "…" : entry.query}
                </span>
                <StateBadge state={entry.state} />
              </div>
              <div className="text-xs text-slate-600 mt-0.5">
                {new Date(entry.timestamp).toLocaleString()}
              </div>
            </button>

            {openId === entry.task_id && (
              <div className="px-3 pb-2">
                {loadingId === entry.task_id ? (
                  <div className="text-xs text-slate-500 py-2 flex items-center gap-1.5">
                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Loading report…
                  </div>
                ) : reportData ? (
                  <ReportView report={reportData} />
                ) : (
                  <div className="text-xs text-slate-600 py-2">Report not available.</div>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
