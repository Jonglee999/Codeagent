import { useRef } from "react";

import * as api from "../api/client";
import type { Project, ProjectFile } from "../types";

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export default function ProjectPanel({
  project,
  files,
  busy,
  onUpload,
  onDeleteFile,
  onDeleteProject,
  onClose,
}: {
  project: Project;
  files: ProjectFile[];
  busy: boolean;
  onUpload: (files: File[]) => void;
  onDeleteFile: (file: ProjectFile) => void;
  onDeleteProject: () => void;
  onClose: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <aside className="project-panel" aria-label={`项目：${project.name}`}>
      <header>
        <div><span>PROJECT</span><h2>{project.name}</h2></div>
        <button type="button" onClick={onClose} aria-label="关闭项目面板">×</button>
      </header>
      <section className="project-guide">
        <strong>项目工作区</strong>
        <p>项目中的对话共享文件。删除单个会话不会删除项目；删除项目会同时删除其托管文件。</p>
      </section>
      <section className="project-files">
        <div className="project-section-head">
          <strong>文件</strong>
          <button type="button" onClick={() => inputRef.current?.click()} disabled={busy}>＋ 添加</button>
        </div>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          multiple
          onChange={(event) => {
            const selected = Array.from(event.target.files || []);
            if (selected.length) onUpload(selected);
            event.target.value = "";
          }}
        />
        {files.length ? (
          <div className="project-file-list">
            {files.map((file) => (
              <article key={file.path}>
                <div className="file-glyph">{file.name.split(".").pop()?.slice(0, 3).toUpperCase() || "FILE"}</div>
                <div><strong title={file.path}>{file.name}</strong><small>{formatBytes(file.size)}</small></div>
                <a href={api.projectFileDownloadUrl(project.project_id, file.path)} download={file.name} aria-label={`下载文件：${file.name}`}>↓</a>
                <button type="button" onClick={() => onDeleteFile(file)} aria-label={`删除文件：${file.name}`}>×</button>
              </article>
            ))}
          </div>
        ) : (
          <div className="project-empty-files">
            <span aria-hidden="true">▱</span>
            <strong>还没有文件</strong>
            <p>上传资料，或让 Agent 在项目中生成文件。</p>
          </div>
        )}
      </section>
      <button type="button" className="delete-project-button" onClick={onDeleteProject}>删除项目及托管文件</button>
    </aside>
  );
}
