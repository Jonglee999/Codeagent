import { useState } from "react";

import * as api from "../api/client";
import type { TaskPreview } from "../types";

export default function PreviewPanel({
  taskId,
  preview,
  onRefresh,
  onClose,
}: {
  taskId: string;
  preview: TaskPreview;
  onRefresh: () => void;
  onClose: () => void;
}) {
  const [entrypoint, setEntrypoint] = useState(preview.entrypoint || "");
  const selectedEntrypoint = preview.entries.includes(entrypoint)
    ? entrypoint
    : preview.entrypoint || "";

  const previewUrl = selectedEntrypoint
    ? api.taskPreviewUrl(taskId, selectedEntrypoint, preview.revision)
    : "";

  return (
    <aside className="preview-panel" aria-label="HTML 实时预览">
      <header>
        <div>
          <span>LIVE PREVIEW</span>
          <h2>页面预览</h2>
        </div>
        <div className="preview-actions">
          <button type="button" onClick={onRefresh} aria-label="刷新页面预览">↻</button>
          <button type="button" onClick={onClose} aria-label="关闭页面预览">×</button>
        </div>
      </header>
      <div className="preview-location">
        <i aria-hidden="true" />
        {preview.entries.length > 1 ? (
          <select
            aria-label="选择 HTML 入口"
            value={selectedEntrypoint}
            onChange={(event) => setEntrypoint(event.target.value)}
          >
            {preview.entries.map((entry) => <option key={entry} value={entry}>{entry}</option>)}
          </select>
        ) : <code>{selectedEntrypoint}</code>}
      </div>
      <div className="preview-canvas">
        {previewUrl ? (
          <iframe
            key={previewUrl}
            title={`实时预览：${selectedEntrypoint}`}
            src={previewUrl}
            sandbox="allow-scripts allow-forms allow-modals"
            referrerPolicy="no-referrer"
          />
        ) : <p>等待 Agent 生成 HTML 文件…</p>}
      </div>
      <footer>工作区文件变化后自动刷新 · 页面运行在隔离沙箱中</footer>
    </aside>
  );
}
