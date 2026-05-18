import type { TaskEvent } from "../types";

interface ProgressStepperProps {
  events: TaskEvent[];
  taskState: string | null;
}

const STEPS = [
  { key: "context_node",       label: "Context",    icon: "◎" },
  { key: "planning_node",      label: "Planning",   icon: "◈" },
  { key: "execution_node",     label: "Execution",  icon: "◆" },
  { key: "validation_node",    label: "Validation", icon: "◉" },
  { key: "human_review_node",  label: "Review",     icon: "◇" },
];

type StepStatus = "pending" | "running" | "done" | "error";

function getStepStatuses(events: TaskEvent[], taskState: string | null): StepStatus[] {
  const startedNodes = new Set(
    events.filter((e) => e.type === "node_start").map((e) => e.node),
  );
  const hasError = events.some((e) => e.type === "task_error");
  const isDone = taskState === "completed" || taskState === "failed";

  // Find the last started node index
  let lastStartedIdx = -1;
  for (let i = STEPS.length - 1; i >= 0; i--) {
    if (startedNodes.has(STEPS[i].key)) { lastStartedIdx = i; break; }
  }

  return STEPS.map((step, i) => {
    if (!startedNodes.has(step.key)) return "pending";
    if (i < lastStartedIdx) return "done";
    if (i === lastStartedIdx) {
      if (isDone) return hasError ? "error" : "done";
      return "running";
    }
    return "done";
  });
}

export default function ProgressStepper({ events, taskState }: ProgressStepperProps) {
  const statuses = getStepStatuses(events, taskState);

  return (
    <div>
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Progress</h3>
      <div className="space-y-2">
        {STEPS.map((step, i) => {
          const status = statuses[i];
          return (
            <div key={step.key} className="flex items-center gap-2.5">
              {/* Step indicator */}
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs shrink-0 transition-colors ${
                status === "done"    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                : status === "running" ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 animate-pulse"
                : status === "error"   ? "bg-red-500/20 text-red-400 border border-red-500/40"
                : "bg-[#0f1117] text-slate-600 border border-[#2a2d3a]"
              }`}>
                {status === "done"    ? "✓"
                : status === "error"  ? "✗"
                : status === "running" ? (
                  <svg className="animate-spin h-2.5 w-2.5" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : step.icon}
              </div>

              {/* Label */}
              <span className={`text-xs transition-colors ${
                status === "done"    ? "text-emerald-400"
                : status === "running" ? "text-indigo-300 font-medium"
                : status === "error"   ? "text-red-400"
                : "text-slate-600"
              }`}>
                {step.label}
              </span>

              {/* Connector line */}
              {i < STEPS.length - 1 && (
                <div className={`ml-auto w-px h-3 ${status === "done" ? "bg-emerald-500/30" : "bg-[#2a2d3a]"}`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
