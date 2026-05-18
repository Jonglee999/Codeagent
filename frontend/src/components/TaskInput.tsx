import { useState } from "react";
import type { TaskCreateRequest } from "../types";

interface TaskInputProps {
  onSubmit: (req: TaskCreateRequest) => void;
  onCancel: () => void;
  loading: boolean;
  running: boolean;
}

export default function TaskInput({ onSubmit, onCancel, loading, running }: TaskInputProps) {
  const [query, setQuery] = useState("");
  const [projectRoot, setProjectRoot] = useState("");
  const [autoMode, setAutoMode] = useState(false);

  const canSubmit = !loading && !running && query.trim() && projectRoot.trim();

  const handleSubmit = () => {
    if (!canSubmit) return;
    onSubmit({ query: query.trim(), project_root: projectRoot.trim(), auto_mode: autoMode });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); handleSubmit(); }
  };

  return (
    <div className="space-y-3" data-testid="task-input">
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">New Task</h3>

      {/* Query */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-1">Task Description</label>
        <textarea
          data-testid="task-query"
          className="w-full bg-[#0f1117] border border-[#2a2d3a] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 resize-none font-mono transition-colors"
          rows={4}
          placeholder="Describe what you want the agent to do…&#10;(Ctrl+Enter to submit)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading || running}
        />
        <div className="text-right text-xs text-slate-600 mt-0.5">{query.length} chars</div>
      </div>

      {/* Project Root */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-1">Project Root Path</label>
        <input
          data-testid="project-root"
          type="text"
          className="w-full bg-[#0f1117] border border-[#2a2d3a] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 font-mono transition-colors"
          placeholder="/path/to/your/project"
          value={projectRoot}
          onChange={(e) => setProjectRoot(e.target.value)}
          disabled={loading || running}
        />
      </div>

      {/* Auto Mode + Submit/Cancel */}
      <div className="flex items-center justify-between pt-1">
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <div
            onClick={() => !loading && !running && setAutoMode((v) => !v)}
            className={`relative w-8 h-4 rounded-full transition-colors cursor-pointer ${autoMode ? "bg-indigo-500" : "bg-[#2a2d3a]"}`}
          >
            <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${autoMode ? "translate-x-4" : "translate-x-0.5"}`} />
          </div>
          <span className="text-xs text-slate-400">Auto mode</span>
        </label>

        {running ? (
          <button
            onClick={onCancel}
            className="px-4 py-1.5 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-xs font-medium hover:bg-red-500/20 transition-colors"
          >
            Cancel
          </button>
        ) : (
          <button
            data-testid="submit-btn"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="px-4 py-1.5 bg-indigo-500 text-white rounded-lg text-xs font-medium hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <span className="flex items-center gap-1.5">
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Submitting…
              </span>
            ) : "Submit Task"}
          </button>
        )}
      </div>
    </div>
  );
}
