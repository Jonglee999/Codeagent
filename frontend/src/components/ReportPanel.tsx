import type { TaskEvent, TaskReport } from "../types";

function describeStep(step: unknown) {
  if (typeof step === "string") return step;
  if (step && typeof step === "object") {
    const value = step as Record<string, unknown>;
    return String(value.description || value.action || JSON.stringify(step));
  }
  return String(step);
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

export default function ReportPanel({
  report,
  completeEvent,
  onRecover,
  onOpenFile,
}: {
  report: TaskReport | null;
  completeEvent: TaskEvent | null;
  onRecover?: () => void;
  onOpenFile?: (path: string) => void;
}) {
  const failed = completeEvent?.type === "task_error" || !!report?.error;
  const duration = report?.duration ?? completeEvent?.duration;
  const tokens = report?.token_usage ?? completeEvent?.token_usage;
  const redisRuntime = (report?.infrastructure_runtime?.redis || {}) as Record<string, unknown>;
  const toolRuntime = (report?.infrastructure_runtime?.tools || {}) as Record<string, unknown>;
  const workingSet = (toolRuntime.working_set || {}) as Record<string, unknown>;
  const selectedTools = stringList(report?.tool_manifest?.selected_names);
  const deferredTools = stringList(report?.tool_manifest?.deferred_names);
  const contextSources = Array.isArray(report?.context_manifest?.sources)
    ? report.context_manifest.sources as Record<string, unknown>[]
    : [];
  const hasDiagnostics = !!(
    selectedTools.length || contextSources.length || report?.resolved_skills?.length
    || report?.mcp_servers?.length || report?.steering_instructions?.length || report?.reflection
    || (report?.model_runtime && Object.keys(report.model_runtime).length)
    || (report?.infrastructure_runtime && Object.keys(report.infrastructure_runtime).length)
  );
  const cancelled = report?.status === "cancelled" || completeEvent?.type === "task_cancelled";

  return (
    <section className={`panel report-panel ${cancelled ? "cancelled" : failed ? "failed" : "passed"}`}>
      <div className="report-heading"><div className="report-icon">{cancelled ? "■" : failed ? "!" : "✓"}</div><div><p className="eyebrow">执行报告{report?.run_profile?.workflow ? ` · ${report.run_profile.workflow === "planned" ? "计划执行" : "直接执行"}` : ""}</p><h2>{cancelled ? "执行已停止" : failed ? "变更需要处理" : "实现已完成"}</h2></div></div>
      {(completeEvent?.error || report?.error) && <div className="alert error"><span>{completeEvent?.error || report?.error}</span></div>}
      <div className="report-stats">
        <div><strong>{duration != null ? `${duration.toFixed(1)}s` : "—"}</strong><span>耗时</span></div>
        <div><strong>{tokens != null ? tokens.toLocaleString() : "—"}</strong><span>Tokens</span></div>
        <div><strong>{report?.changes?.length || 0}</strong><span>变更</span></div>
        <div><strong>{report?.validation_results?.length || 0}</strong><span>检查</span></div>
      </div>

      {report?.changes?.length ? <div className="report-section"><h3>变更文件</h3><div className="change-list">{report.changes.map((change, index) => { const path = String(change.file_path || change.path || `变更 ${index + 1}`); return onOpenFile ? <button type="button" className="change-file-link" key={index} onClick={() => onOpenFile(path)} title={`查看 ${path}`}>{path}</button> : <code key={index}>{path}</code>; })}</div></div> : null}
      {report?.validation_results?.length ? <div className="report-section"><h3>验证证据</h3><div className="validation-grid">{report.validation_results.map((result, index) => <div className={result.passed ? "valid" : "invalid"} key={index}><span>{result.passed ? "✓" : "×"}</span>{String(result.layer || result.type || `检查 ${index + 1}`)}</div>)}</div></div> : null}
      {report?.plan?.length ? <details className="report-section"><summary>执行方案（{report.plan.length} 步）</summary><ol>{report.plan.map((step, index) => <li key={index}><span>{index + 1}</span>{describeStep(step)}</li>)}</ol></details> : null}
      {report?.artifacts?.length ? <div className="report-section artifact-list"><h3>可审查产物</h3>{report.artifacts.map((artifact) => <a key={artifact.artifact_id} href={artifact.download_url} download><code>{artifact.kind}</code><span>{artifact.size != null ? `${artifact.size.toLocaleString()} B` : "下载"}</span></a>)}</div> : null}
      {report?.recovered_from_task_id ? <div className="report-section"><h3>恢复来源</h3><div className="change-list"><code>{report.recovered_from_task_id}</code></div></div> : null}

      {hasDiagnostics ? <details className="report-section run-diagnostics">
        <summary>运行范围与诊断</summary>
        {selectedTools.length ? <div><h3>本轮工具</h3><div className="change-list">{selectedTools.map((name) => <code key={name}>{name}</code>)}</div>{deferredTools.length ? <><p>以下 {deferredTools.length} 个无关或高权限工具未暴露给模型：</p><div className="change-list deferred-tools">{deferredTools.map((name) => <code key={name}>{name}</code>)}</div></> : null}</div> : null}
        {contextSources.length ? <div><h3>上下文来源</h3><div className="change-list">{contextSources.map((source, index) => <code key={index}>{String(source.kind || `来源 ${index + 1}`)}{source.tokens != null ? ` · ${Number(source.tokens)} tokens` : ""}{source.included === false ? " · 已延后" : ""}</code>)}</div></div> : null}
        {report?.resolved_skills?.length ? <div><h3>Skills</h3><div className="change-list">{report.resolved_skills.map((skill, index) => <code key={index}>{String(skill.skill_id || `Skill ${index + 1}`)}</code>)}</div></div> : null}
        {report?.mcp_servers?.length ? <div><h3>MCP</h3><div className="change-list">{report.mcp_servers.map((server, index) => <code key={index}>{String(server.name || `Server ${index + 1}`)} · {server.deferred ? "已延后" : server.available ? "可用" : "已降级"}</code>)}</div></div> : null}
        {report?.steering_instructions?.length ? <div><h3>运行中追加指令</h3><div className="change-list">{report.steering_instructions.map((instruction, index) => <code key={index}>{instruction}</code>)}</div></div> : null}
        {report?.model_runtime && Object.keys(report.model_runtime).length ? <div><h3>模型可靠性</h3><div className="change-list"><code>{String(report.model_runtime.active_model || "unknown")} · 重试 {Number(report.model_runtime.retry_count || 0)} 次 · {report.model_runtime.fallback_activated ? "已切换备用模型" : "主模型"}</code></div></div> : null}
        {report?.infrastructure_runtime && Object.keys(report.infrastructure_runtime).length ? <div><h3>基础设施可靠性</h3><div className="change-list"><code>Redis 重试 {Number(redisRuntime.retry_count || 0)} 次 · 恢复 {Number(redisRuntime.recovery_count || 0)} 次 · 丢弃事件 {Number(redisRuntime.dropped_event_count || 0)} 个</code><code>工具调用 {Number(toolRuntime.call_count || 0)} 次 · 失败 {Number(toolRuntime.failure_count || 0)} 次 · 超时 {Number(toolRuntime.timeout_count || 0)} 次</code>{Number(workingSet.read_requests || 0) > 0 ? <code>Working set 读取 {Number(workingSet.read_requests || 0)} 次 · 命中 {Number(workingSet.cache_hits || 0)} 次 · 精确失效 {Number(workingSet.invalidation_count || 0)} 项</code> : null}</div></div> : null}
        {report?.reflection ? <div><h3>失败分析</h3><div className="change-list"><code>{String(report.reflection.next_action || "finish")} · {String(report.reflection.reason || "validation evidence reviewed")}</code></div></div> : null}
      </details> : null}
      {report?.warnings?.length ? <div className="alert error"><span>{report.warnings.join("；")}</span></div> : null}
      {(cancelled || failed) && onRecover ? <button type="button" className="recover-button" onClick={onRecover}>从当前工作区继续</button> : null}
    </section>
  );
}
