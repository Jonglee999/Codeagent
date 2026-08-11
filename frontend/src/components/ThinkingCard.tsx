import { useMemo, useState } from "react";

import type { TaskEvent, TaskReport, TaskState } from "../types";
import ExecutionLog from "./ExecutionLog";
import ProgressStepper from "./ProgressStepper";
import ReportPanel from "./ReportPanel";

function currentActivity(events: TaskEvent[]) {
  const event = [...events].reverse().find((item) => !["ping", "node_complete"].includes(item.type));
  if (!event) return "正在启动";
  if (event.type === "tool_call") return `正在使用 ${event.tool || "工具"}`;
  if (event.type === "tool_result") return event.success === false ? `${event.tool || "工具"}执行失败` : `${event.tool || "工具"}已完成`;
  if (event.type === "validation_result") return event.passed ? "验证通过" : "正在处理验证失败";
  if (event.type === "human_review_required") return "等待你的确认";
  if (event.type === "run_profile_selected") return event.data?.workflow === "planned" ? "正在理解复杂任务" : "准备直接执行";
  if (event.type === "node_start") {
    if (event.node?.startsWith("planning")) return "正在制定方案";
    if (event.node?.startsWith("validation")) return "正在验证结果";
    if (event.node?.startsWith("execution")) return "正在执行修改";
    if (event.node?.startsWith("context")) return "正在读取必要上下文";
  }
  return event.summary || "正在执行";
}

export default function ThinkingCard({
  events,
  taskState,
  report = null,
  completeEvent = null,
  onHumanReview,
  expanded: controlledExpanded,
  onExpandedChange,
  onRecover,
  onOpenFile,
}: {
  events: TaskEvent[];
  taskState: TaskState | "idle";
  report?: TaskReport | null;
  completeEvent?: TaskEvent | null;
  onHumanReview: (event: TaskEvent) => void;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onRecover?: () => void;
  onOpenFile?: (path: string) => void;
}) {
  const running = taskState === "pending" || taskState === "running" || taskState === "waiting_review";
  const [internalExpanded, setInternalExpanded] = useState(false);
  const expanded = controlledExpanded ?? internalExpanded;
  const toggleExpanded = () => {
    const next = !expanded;
    if (onExpandedChange) onExpandedChange(next);
    else setInternalExpanded(next);
  };

  const failed = taskState === "failed";
  const activity = useMemo(() => currentActivity(events), [events]);
  const label = running ? activity : taskState === "cancelled" ? "执行已停止" : failed ? "执行未完成" : "执行已完成";
  const eventCount = events.filter((event) => ![
    "ping", "assistant_message", "node_complete", "capabilities_resolved",
  ].includes(event.type)).length;

  return (
    <section className={`thinking-card ${expanded ? "expanded" : "collapsed"}`}>
      <button type="button" className="thinking-toggle" aria-expanded={expanded} onClick={toggleExpanded}>
        <span className="thinking-bulb" aria-hidden="true">✦</span>
        <strong>{label}</strong>
        <small>{eventCount} 项活动 · {expanded ? "收起" : "查看过程"}</small>
        <span className="thinking-chevron" aria-hidden="true">⌄</span>
      </button>
      {expanded && (
        <div className="thinking-content">
          <ProgressStepper events={events} taskState={taskState} />
          <div className="execution-surface">
            <ExecutionLog events={events} onHumanReview={onHumanReview} isRunning={running} />
          </div>
          {(report || taskState === "failed" || taskState === "completed" || taskState === "cancelled") && (
            <ReportPanel report={report} completeEvent={completeEvent} onRecover={onRecover} onOpenFile={onOpenFile} />
          )}
        </div>
      )}
    </section>
  );
}
