import type { TaskEvent } from "../types";

type Phase = { key: string; label: string };

function normalizeNode(node = "") {
  return node.replace("_node", "").split("/")[0];
}

export default function ProgressStepper({ events, taskState }: { events: TaskEvent[]; taskState: string }) {
  const nodes = new Set(events.filter((event) => event.type === "node_start").map((event) => normalizeNode(event.node)));
  const toolEvents = events.filter((event) => event.type === "tool_call" || event.type === "tool_result");
  const tools = new Set(toolEvents.map((event) => event.tool));
  const terminal = taskState === "completed" || taskState === "failed" || taskState === "cancelled";

  const phases: Phase[] = [{ key: "prepare", label: "准备" }];
  if (nodes.has("planning")) phases.push({ key: "planning", label: "方案" });
  if (nodes.has("execution") || toolEvents.length) phases.push({ key: "execution", label: "执行" });
  if (nodes.has("validation") || tools.has("run_terminal")) phases.push({ key: "validation", label: "验证" });
  if (terminal) phases.push({ key: "complete", label: taskState === "completed" ? "完成" : "结束" });

  return (
    <div className="progress-strip" aria-label="执行进度">
      {phases.map((phase, index) => {
        const current = index === phases.length - 1;
        const failed = current && taskState === "failed";
        return (
          <div className={`progress-step ${current ? "current" : "done"} ${failed ? "failed" : ""}`} key={phase.key}>
            <span>{current && !terminal ? index + 1 : "✓"}</span>
            <strong>{phase.label}</strong>
            {index < phases.length - 1 && <i />}
          </div>
        );
      })}
    </div>
  );
}
