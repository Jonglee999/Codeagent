import { useState } from "react";
import type { TaskEvent } from "../types";

interface HumanReviewModalProps {
  event: TaskEvent | null;
  onDecision: (decision: "approve" | "reject" | "modify", feedback?: string) => void;
  onClose: () => void;
}

export default function HumanReviewModal({ event, onDecision, onClose }: HumanReviewModalProps) {
  const [modifyText, setModifyText] = useState("");
  const [showModify, setShowModify] = useState(false);

  if (!event) return null;

  const details = event.details || {};
  const plan = details.plan as unknown[] | undefined;

  const handleModifySubmit = () => {
    const text = modifyText.trim();
    if (!text) return;
    onDecision("modify", text);
    setModifyText("");
    setShowModify(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-[#1a1d27] border border-amber-500/30 rounded-xl shadow-2xl w-full max-w-lg max-h-[80vh] flex flex-col animate-fade-in">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#2a2d3a] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 text-sm">
              ⚠
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-100">Human Review Required</h2>
              {event.review_type && (
                <p className="text-xs text-slate-500 mt-0.5">
                  <code className="bg-[#0f1117] px-1.5 py-0.5 rounded text-amber-400">{event.review_type}</code>
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-300 transition-colors text-lg leading-none"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 overflow-y-auto flex-1 space-y-4">
          {plan && plan.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Execution Plan</h3>
              <ol className="space-y-1.5">
                {plan.map((item: unknown, i: number) => (
                  <li key={i} className="flex items-start gap-2.5 text-sm">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-xs flex items-center justify-center font-mono">
                      {i + 1}
                    </span>
                    <span className="text-slate-300 pt-0.5">
                      {typeof item === "string" ? item : JSON.stringify(item)}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {!plan && Object.keys(details).length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Details</h3>
              <pre className="text-xs bg-[#0f1117] border border-[#2a2d3a] p-3 rounded-lg overflow-x-auto text-slate-300">
                {JSON.stringify(details, null, 2)}
              </pre>
            </div>
          )}

          {showModify && (
            <div className="animate-slide-in">
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Modification Instructions
              </label>
              <textarea
                autoFocus
                className="w-full bg-[#0f1117] border border-[#2a2d3a] rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 resize-none transition-colors"
                rows={3}
                placeholder="Describe your requested changes…"
                value={modifyText}
                onChange={(e) => setModifyText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.ctrlKey && e.key === "Enter") handleModifySubmit();
                  if (e.key === "Escape") setShowModify(false);
                }}
              />
              <p className="text-xs text-slate-600 mt-1">Ctrl+Enter to submit · Esc to cancel</p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="px-5 py-4 border-t border-[#2a2d3a]">
          {!showModify ? (
            <div className="flex gap-2">
              <button
                data-testid="review-approve"
                onClick={() => onDecision("approve")}
                className="flex-1 py-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-lg text-sm font-medium hover:bg-emerald-500/20 transition-colors"
              >
                Approve
              </button>
              <button
                data-testid="review-reject"
                onClick={() => onDecision("reject")}
                className="flex-1 py-2 bg-red-500/10 text-red-400 border border-red-500/30 rounded-lg text-sm font-medium hover:bg-red-500/20 transition-colors"
              >
                Reject
              </button>
              <button
                data-testid="review-modify"
                onClick={() => setShowModify(true)}
                className="flex-1 py-2 bg-[#0f1117] text-slate-400 border border-[#2a2d3a] rounded-lg text-sm font-medium hover:border-indigo-500/40 hover:text-indigo-400 transition-colors"
              >
                Modify
              </button>
            </div>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => setShowModify(false)}
                className="flex-1 py-2 bg-[#0f1117] text-slate-400 border border-[#2a2d3a] rounded-lg text-sm font-medium hover:bg-white/5 transition-colors"
              >
                Back
              </button>
              <button
                data-testid="review-modify-submit"
                onClick={handleModifySubmit}
                disabled={!modifyText.trim()}
                className="flex-1 py-2 bg-indigo-500 text-white rounded-lg text-sm font-medium hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Submit Changes
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
