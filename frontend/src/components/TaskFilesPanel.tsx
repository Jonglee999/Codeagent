import { useCallback, useEffect, useMemo, useState } from "react";

import * as api from "../api/client";
import type { TaskFile } from "../types";

const MAX_INLINE_BYTES = 512 * 1024;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function TaskFilesPanel({
  taskId,
  initialPath,
  onClose,
}: {
  taskId: string;
  initialPath?: string | null;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<TaskFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [selectedPath, setSelectedPath] = useState<string | null>(initialPath || null);
  const [content, setContent] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  const loadFiles = useCallback(async () => {
    try {
      setFiles(await api.listTaskFiles(taskId));
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "无法读取任务文件");
    }
  }, [taskId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadFiles(), 0);
    return () => window.clearTimeout(timer);
  }, [loadFiles]);

  const openFile = useCallback(async (path: string) => {
    const file = files?.find((item) => item.path === path);
    setSelectedPath(path);
    setContent(null);
    setContentError(null);
    if (file?.binary) return; // 二进制文件不内联读取，仅提供下载/打开
    setContentLoading(true);
    try {
      const response = await fetch(api.taskWorkspaceFileUrl(taskId, path));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setContent(await response.text());
    } catch (loadError) {
      setContentError(loadError instanceof Error ? loadError.message : "无法读取文件内容");
    } finally {
      setContentLoading(false);
    }
  }, [files, taskId]);

  // 从变更文件链接打开时，文件列表加载后自动读取内容。路径可能是绝对路径，
  // 此时按 basename 回退到工作区相对路径。
  useEffect(() => {
    if (!files || !initialPath || selectedPath !== initialPath) return;
    let path: string | undefined;
    if (files.some((file) => file.path === initialPath)) {
      path = initialPath;
    } else {
      const base = initialPath.split("/").pop()?.split("\\").pop();
      path = base ? files.find((file) => file.name === base)?.path : undefined;
    }
    if (!path) return;
    const timer = window.setTimeout(() => void openFile(path), 0);
    return () => window.clearTimeout(timer);
  }, [files, initialPath, openFile, selectedPath]);

  const filteredFiles = useMemo(() => {
    if (!files) return [];
    const needle = filter.trim().toLowerCase();
    if (!needle) return files;
    return files.filter((file) => file.path.toLowerCase().includes(needle));
  }, [files, filter]);

  const grouped = useMemo(() => {
    const map = new Map<string, TaskFile[]>();
    for (const file of filteredFiles) {
      const slash = file.path.lastIndexOf("/");
      const directory = slash >= 0 ? file.path.slice(0, slash) : "";
      if (!map.has(directory)) map.set(directory, []);
      map.get(directory)!.push(file);
    }
    return [...map.entries()].map(([directory, entries]) => ({
      directory,
      entries: [...entries].sort((a, b) => a.name.localeCompare(b.name)),
    }));
  }, [filteredFiles]);

  const selectedFile = useMemo(
    () => files?.find((file) => file.path === selectedPath) || null,
    [files, selectedPath],
  );
  const fileUrl = selectedPath ? api.taskWorkspaceFileUrl(taskId, selectedPath) : "";
  const tooLarge = !!selectedFile && !selectedFile.binary && selectedFile.size > MAX_INLINE_BYTES;

  return (
    <aside className="task-files-panel" aria-label="任务工作区文件">
      <header>
        <div>
          <span>WORKSPACE FILES</span>
          <h2>任务文件</h2>
        </div>
        <div className="preview-actions">
          <button type="button" onClick={() => void loadFiles()} aria-label="刷新文件列表">↻</button>
          <button type="button" onClick={onClose} aria-label="关闭文件面板">×</button>
        </div>
      </header>
      <div className="task-files-search">
        <input
          type="search"
          placeholder="搜索文件名（如 random_generator）"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          aria-label="搜索任务文件"
        />
      </div>
      <div className="task-files-body">
        <div className="task-files-list" role="listbox" aria-label="文件列表">
          {!files && !error && <p className="task-files-empty">正在加载…</p>}
          {error && !files && <p className="task-files-empty error">{error}</p>}
          {files && files.length === 0 && <p className="task-files-empty">工作区中没有文件。</p>}
          {files && grouped.length === 0 && <p className="task-files-empty">没有匹配「{filter}」的文件。</p>}
          {grouped.map(({ directory, entries }) => (
            <div key={directory || "__root__"}>
              <div className="tf-dir">{directory || "根目录"}</div>
              {entries.map((file) => (
                <button
                  key={file.path}
                  type="button"
                  className={file.path === selectedPath ? "active" : ""}
                  onClick={() => void openFile(file.path)}
                  aria-selected={file.path === selectedPath}
                >
                  <span>{file.name}</span>
                  {file.binary && <span className="tf-binary">二进制</span>}
                </button>
              ))}
            </div>
          ))}
        </div>
        <div className="task-files-content">
          {contentLoading && <p className="task-files-empty">正在读取…</p>}
          {!contentLoading && contentError && <p className="task-files-empty error">{contentError}</p>}
          {!contentLoading && !contentError && selectedFile?.binary && (
            <p className="task-files-empty">二进制文件，请下载后查看。</p>
          )}
          {!contentLoading && !contentError && tooLarge && (
            <p className="task-files-empty">文件过大（{formatSize(selectedFile.size)}），请在浏览器中打开或下载。</p>
          )}
          {!contentLoading && !contentError && content && !tooLarge && !selectedFile?.binary && (
            <pre>{content}</pre>
          )}
          {!contentLoading && !contentError && !selectedPath && (
            <p className="task-files-empty">从左侧选择一个文件查看内容。</p>
          )}
          {selectedFile && (
            <div className="task-files-filebar">
              <code>{selectedPath}</code>
              <span>{formatSize(selectedFile.size)}</span>
              <a href={fileUrl} download={selectedFile.name}>下载</a>
              {!selectedFile.binary && <a href={fileUrl} target="_blank" rel="noreferrer">在新窗口打开</a>}
            </div>
          )}
        </div>
      </div>
      <footer>显示 Agent 在工作区中生成与修改的文件 · 隐藏 .git/.venv/node_modules 等目录</footer>
    </aside>
  );
}
