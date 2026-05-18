import { useRef, useEffect, useState } from "react";
import type { TaskEvent, LogGroup } from "../types";

interface ExecutionLogProps {
  events: TaskEvent[];
  onHumanReview: (event: TaskEvent) => void;
  isRunning: boolean;
}

function groupEvents(events: TaskEvent[]): LogGroup[] {
  const groups: LogGroup[] = [];
  let current: LogGroup | null = null;
  for (const ev of events) {
    if (ev.type === "node_start") {
      current = { node: ev.node || "unknown", events: [ev] };
      groups.push(current);
    } else if (current) {
      current.events.push(ev);
    } else {
      current = { node: "init", events: [ev] };
      groups.push(current);
    }
  }
  return groups;
}

function nodeLabel(node: string): string {
  const map: Record<string, string> = {
    worker: "Worker",
    context_node: "Context",
    planning_node: "Planning",
    execution_node: "Execution",
    validation_node: "Validation",
    human_review_node: "Human Review",
    init: "Init",
  };
  return map[node] ?? node.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function nodeColor(node: string): string {
  const map: Record<string, string> = {
    worker: "text-slate-400 border-slate-600",
    context_node: "text-cyan-400 border-cyan-500/40",
    planning_node: "text-violet-400 border-violet-500/40",
    execution_node: "text-indigo-400 border-indigo-500/40",
    validation_node: "text-emerald-400 border-emerald-500/40",
    human_review_node: "text-amber-400 border-amber-500/40",
    init: "text-slate-400 border-slate-600",
  };
  return map[node] ?? "text-slate-400 border-slate-600";
}

function groupDone(group: LogGroup): boolean {
  return group.events.some(
    (e) => e.type === "task_complete" || e.type === "task_error" ||
      (e.type === "node_start" && group.events.length > 1),
  );
}

function EventRow({ event }: { event: TaskEvent }) {
  const [expanded, setExpanded] = useState(false);

  if (event.type === "node_start") return null; // rendered as group header

  return (
    <div className="animate-slide-in flex items-start gap-2 py-1 px-2 rounded-md text-xs hover:bg-white/5 transition-colors">
      {/* icon */}
      <span className="mt-0.5 shrink-0 w-3.5 text-center">
        {event.type === "tool_call" && <span className="text-slate-500">⚙</span>}
        {event.type === "tool_result" && (
          event.success
            ? <span className="text-emerald-400">✓</span>
            : <span className="text-red-400">✗</span>
        )}
        {event.type === "validation_result" && (
          event.passed
            ? <span className="text-emerald-400">✓</span>
            : <span className="text-red-400">✗</span>
        )}
        {event.type === "human_review_required" && <span className="text-amber-400">⚠</span>}
        {event.type === "task_complete" && <span className="text-emerald-400">■</span>}
        {event.type === "task_error" && <span className="text-red-400">■</span>}
      </span>

      {/* content */}
      <div className="flex-1 min-w-0">
        {event.type === "tool_call" && (
          <span className="text-slate-400">
            <span className="text-slate-500">tool </span>
            <code className="text-indigo-300 bg-indigo-500/10 px-1 rounded">{event.tool}</code>
            {event.params && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="ml-2 text-slate-600 hover:text-slate-400 transition-colors"
              >
                {expanded ? "▲ hide" : "▼ params"}
              </button>
            )}
            {expanded && event.params && (
              <pre className="mt-1 text-xs bg-[#0f1117] border border-[#2a2d3a] p-2 rounded overflow-x-auto text-slate-300">
                {JSON.stringify(event.params, null, 2)}
              </pre>
            )}
          </span>
        )}
        {event.type === "tool_result" && (
          <span className={event.success ? "text-emerald-300" : "text-red-300"}>
            {event.summary || event.tool || "result"}
          </span>
        )}
        {event.type === "validation_result" && (
          <span className={event.passed ? "text-emerald-300" : "text-red-300"}>
            {event.passed ? "Validation passed" : "Validation failed"}
            {event.layer && <span className="text-slate-500 ml-1">({event.layer})</span>}
            {event.errors && event.errors.length > 0 && (
              <ul className="mt-1 text-red-400 list-disc list-inside space-y-0.5">
                {event.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            )}
          </span>
        )}
        {event.type === "human_review_required" && (
          <span className="text-amber-300 font-medium">
            Human review required
            {event.review_type && <span className="ml-1 text-amber-500 font-normal">({event.review_type})</span>}
          </span>
        )}
        {event.type === "task_complete" && (
          <span className="text-emerald-300 font-medium">
            Task completed{event.duration != null ? ` — ${event.duration.toFixed(1)}s` : ""}
            {event.token_usage ? <span className="text-slate-500 ml-2">{event.token_usage} tokens</span> : null}
          </span>
        )}
        {event.type === "task_error" && (
          <span className="text-red-300">Error: {event.error}</span>
        )}
      </div>

      <span className="text-slate-600 shrink-0 tabular-nums">
        {new Date(event.timestamp).toLocaleTimeString()}
      </span>
    </div>
  );
}

function NodeCard({ group, isLast, isRunning }: { group: LogGroup; isLast: boolean; isRunning: boolean }) {
  const [collapsed, setCollapsed] = useState(false);
  const hasError = group.events.some((e) => e.type === "task_error" || (e.type === "tool_result" && e.success === false));
  const isDone = !isLast || !isRunning;
  const colorClass = nodeColor(group.node);

  return (
    <div className={`border rounded-lg overflow-hidden animate-fade-in ${
      hasError ? "border-red-500/30 bg-red-500/5"
      : isDone  ? "border-[#2a2d3a] bg-[#1a1d27]"
      : "border-indigo-500/30 bg-indigo-500/5"
    }`}>
      {/* Card header */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-white/5 transition-colors text-left"
      >
        <span className={`text-xs font-semibold ${colorClass.split(" ")[0]}`}>
          {nodeLabel(group.node)}
        </span>
        <span className="text-slate-600 text-xs">{group.events.filter((e) => e.type !== "node_start").length} events</span>
        <span className="ml-auto">
          {!isDone && isRunning ? (
            <span className="flex items-center gap-1 text-xs text-indigo-400">
              <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              running
            </span>
          ) : hasError ? (
            <span className="text-xs text-red-400">failed</span>
          ) : (
            <span className="text-xs text-emerald-400">done</span>
          )}
        </span>
        <span className="text-slate-600 text-xs">{collapsed ? "▶" : "▼"}</span>
      </button>

      {/* Card body */}
      {!collapsed && (
        <div className="px-2 pb-2 space-y-0.5 border-t border-[#2a2d3a]">
          {group.events.map((ev, i) => <EventRow key={i} event={ev} />)}
        </div>
      )}
    </div>
  );
}

export default function ExecutionLog({ events, onHumanReview, isRunning }: ExecutionLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const notifiedCountRef = useRef(0);
  const groups = groupEvents(events);

  useEffect(() => {
    for (let i = notifiedCountRef.current; i < events.length; i++) {
      if (events[i].type === "human_review_required") onHumanReview(events[i]);
    }
    notifiedCountRef.current = events.length;
  }, [events, onHumanReview]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div
        data-testid="execution-log"
        className="border border-[#2a2d3a] rounded-xl p-12 text-center"
      >
        <div className="text-3xl mb-3 opacity-20">⚡</div>
        <p className="text-slate-500 text-sm">No execution events yet.</p>
        <p className="text-slate-600 text-xs mt-1">Submit a task to begin.</p>
      </div>
    );
  }

  return (
    <div
      data-testid="execution-log"
      className="border border-[#2a2d3a] rounded-xl bg-[#1a1d27] overflow-y-auto"
      style={{ maxHeight: "calc(100vh - 220px)", minHeight: "300px" }}
    >
      <div className="p-3 space-y-2">
        {groups.map((group, gi) => (
          <NodeCard
            key={gi}
            group={group}
            isLast={gi === groups.length - 1}
            isRunning={isRunning}
          />
        ))}
        {isRunning && (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-indigo-400 animate-pulse">
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Agent is running…
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
