import type { TaskEvent, TaskReport } from "../types";

interface ReportPanelProps {
  report: unknown;
  completeEvent: TaskEvent | null;
}

export default function ReportPanel({ report, completeEvent }: ReportPanelProps) {
  const r = report as TaskReport | null;

  const duration = r?.duration ?? completeEvent?.duration;
  const tokenUsage = r?.token_usage ?? completeEvent?.token_usage;
  const isError = completeEvent?.type === "task_error";

  return (
    <div className={`border rounded-xl p-4 animate-slide-in ${
      isError
        ? "border-red-500/30 bg-red-500/5"
        : "border-emerald-500/30 bg-emerald-500/5"
    }`}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <span className={`text-base ${isError ? "text-red-400" : "text-emerald-400"}`}>
          {isError ? "✗" : "✓"}
        </span>
        <h3 className={`text-sm font-semibold ${isError ? "text-red-300" : "text-emerald-300"}`}>
          {isError ? "Task Failed" : "Task Completed"}
        </h3>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-[#0f1117]/60 border border-[#2a2d3a] rounded-lg p-3 text-center">
          <div className="text-xl font-mono text-indigo-400">
            {duration != null ? `${duration.toFixed(1)}s` : "—"}
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Duration</div>
        </div>
        <div className="bg-[#0f1117]/60 border border-[#2a2d3a] rounded-lg p-3 text-center">
          <div className="text-xl font-mono text-violet-400">
            {tokenUsage != null ? tokenUsage.toLocaleString() : "—"}
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Tokens Used</div>
        </div>
      </div>

      {/* Error message */}
      {(isError || r?.error) && (
        <div className="mb-4 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {completeEvent?.error ?? r?.error ?? "Unknown error"}
        </div>
      )}

      {/* Plan */}
      {r?.plan && r.plan.length > 0 && (
        <div className="mb-4">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Execution Plan ({r.plan.length} steps)
          </div>
          <ol className="space-y-1.5">
            {r.plan.map((step, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span className="shrink-0 w-4 h-4 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center font-mono text-[10px]">
                  {i + 1}
                </span>
                <span className="text-slate-400 pt-0.5">
                  {typeof step === "string"
                    ? step
                    : (step as Record<string, unknown>)?.description as string
                      ?? (step as Record<string, unknown>)?.action as string
                      ?? JSON.stringify(step)}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Changed files */}
      {r?.changes && r.changes.length > 0 && (
        <div className="mb-4">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Changed Files ({r.changes.length})
          </div>
          <ul className="space-y-1">
            {r.changes.map((c, i) => {
              const change = c as Record<string, unknown>;
              const path = change?.file_path as string ?? change?.path as string ?? JSON.stringify(c);
              const op = change?.operation as string ?? change?.type as string;
              return (
                <li key={i} className="flex items-center gap-2 text-xs">
                  <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                    op === "create" ? "bg-emerald-500/10 text-emerald-400"
                    : op === "delete" ? "bg-red-500/10 text-red-400"
                    : "bg-indigo-500/10 text-indigo-400"
                  }`}>
                    {op ?? "edit"}
                  </span>
                  <code className="text-slate-400 font-mono truncate">{path}</code>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Validation results */}
      {r?.validation_results && r.validation_results.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Validation Results
          </div>
          <ul className="space-y-1">
            {r.validation_results.map((v, i) => {
              const vr = v as Record<string, unknown>;
              const passed = vr?.passed as boolean;
              return (
                <li key={i} className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${
                  passed ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
                }`}>
                  <span>{passed ? "✓" : "✗"}</span>
                  <span>{vr?.layer as string ?? vr?.type as string ?? `Check ${i + 1}`}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
