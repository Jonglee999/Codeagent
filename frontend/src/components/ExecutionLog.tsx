import { useEffect, useMemo, useRef, useState } from "react";
import type { TaskEvent } from "../types";

const LABELS: Record<string, string> = {
  worker: "Agent 已启动",
  context: "读取必要上下文",
  context_node: "读取必要上下文",
  planning: "制定执行方案",
  planning_node: "制定执行方案",
  execution: "执行任务",
  execution_node: "执行任务",
  validation: "验证结果",
  validation_node: "验证结果",
  reflection: "分析失败证据",
};

function titleFor(event: TaskEvent) {
  if (event.type === "run_profile_selected") return event.data?.workflow === "planned" ? "采用计划执行" : "采用直接执行";
  if (event.type === "tools_selected") return "已选择本轮工具";
  if (event.type === "context_selected") return "已选择本轮上下文";
  if (event.type === "mcp_discovery_deferred") return "已延后无关 MCP 服务";
  if (event.type === "steering_queued") return "追加指令已排队";
  if (event.type === "steering_applied") return "Agent 已应用追加指令";
  if (event.type === "recovery_started") return "已从上次运行继续";
  if (event.type === "node_start") return LABELS[event.node || ""] || (event.node || "Agent 步骤").replaceAll("_", " ");
  if (event.type === "node_complete") return event.success === false ? "步骤失败" : "步骤完成";
  if (event.type === "tool_call") return `正在使用 ${event.tool || "工具"}`;
  if (event.type === "tool_result") return event.success === false ? `${event.tool || "工具"}执行失败` : `${event.tool || "工具"}已完成`;
  if (event.type === "validation_result") return `${event.layer || "验证"}${event.passed ? "通过" : "失败"}`;
  if (event.type === "human_review_required") return "需要你的确认";
  if (event.type === "task_complete") return "任务已完成";
  if (event.type === "task_error") return "任务未完成";
  if (event.type === "task_cancelled") return "任务已取消";
  if (event.type === "deviation_detected") return "已阻止偏离目标的调用";
  if (event.type === "memory_recalled") return "使用了相关项目经验";
  if (event.type === "memory_extracted") return "保存了可复用经验";
  if (event.type === "strategy_recalled") return "使用了历史策略";
  if (event.type === "skill_resolved") return "已匹配项目能力";
  if (event.type === "transcript_saved") return "执行记录已保存";
  if (event.type === "capability_degraded") return "部分能力已降级";
  if (event.type === "capabilities_resolved") return "工具能力已就绪";
  if (event.type === "circuit_state_changed") return event.data?.capability === "model" ? "模型连接状态变化" : "MCP 连接状态变化";
  if (event.type === "permission_denied") return "工具权限被拒绝";
  if (event.type === "mcp_server_status") return "MCP 服务状态";
  if (event.type === "mcp_tool_discovered") return "发现 MCP 工具";
  if (event.type === "model_retry_scheduled") return "模型调用即将重试";
  if (event.type === "model_call_failed") return "模型调用失败";
  if (event.type === "fallback_activated") return "已切换备用模型";
  if (event.type === "infrastructure_recovered") return "基础设施连接已恢复";
  return event.type.replaceAll("_", " ");
}

function compactEvents(events: TaskEvent[]) {
  const compacted: TaskEvent[] = [];
  for (const event of events) {
    if (["ping", "assistant_message"].includes(event.type)) continue;
    if (event.type === "node_complete" && event.success !== false) continue;
    if (event.type === "tool_result") {
      const callIndex = compacted.findLastIndex((candidate) => candidate.type === "tool_call" && candidate.tool === event.tool);
      if (callIndex >= 0) {
        const call = compacted[callIndex];
        compacted.splice(callIndex, 1, { ...call, ...event, params: call.params, type: "tool_result" });
        continue;
      }
    }
    compacted.push(event);
  }
  return compacted;
}

function EventItem({ event, onReview }: { event: TaskEvent; onReview: (event: TaskEvent) => void }) {
  const [expanded, setExpanded] = useState(false);
  const failed = event.type === "task_error" || event.success === false || event.passed === false;
  const success = event.type === "task_complete" || event.success === true || event.passed === true;
  const payload = event.params || event.data || event.details;
  return (
    <article className={`event-item ${failed ? "event-failed" : ""} ${success ? "event-success" : ""}`}>
      <div className="event-rail"><span /></div>
      <div className="event-body">
        <div className="event-topline">
          <strong>{titleFor(event)}</strong>
          <time>{event.timestamp ? new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : ""}</time>
        </div>
        {(event.summary || event.error) && <p>{event.error || event.summary}</p>}
        {event.errors?.length ? <ul className="event-errors">{event.errors.map((error) => <li key={error}>{error}</li>)}</ul> : null}
        {payload && <button className="details-button" onClick={() => setExpanded((value) => !value)}>{expanded ? "收起详情" : "查看详情"}</button>}
        {expanded && payload && <pre>{JSON.stringify(payload, null, 2)}</pre>}
        {event.type === "human_review_required" && <button className="review-button" onClick={() => onReview(event)}>打开审核</button>}
      </div>
    </article>
  );
}

export default function ExecutionLog({ events, onHumanReview, isRunning }: { events: TaskEvent[]; onHumanReview: (event: TaskEvent) => void; isRunning: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);
  const visible = useMemo(() => compactEvents(events), [events]);
  useEffect(() => {
    if (typeof endRef.current?.scrollIntoView === "function") endRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [visible.length]);

  if (!visible.length) return (
    <div className="empty-timeline">
      <div className="empty-orbit"><span /></div>
      <h3>{isRunning ? "正在连接执行流" : "等待任务"}</h3>
      <p>{isRunning ? "实际使用的工具和验证结果会显示在这里。" : "描述目标后，Agent 会选择合适的执行方式。"}</p>
    </div>
  );
  return <div className="event-list">{visible.map((event, index) => <EventItem key={event.event_id || `${event.type}-${event.timestamp}-${index}`} event={event} onReview={onHumanReview} />)}<div ref={endRef} /></div>;
}
