import { useEffect, useState } from "react";
import type { TaskEvent } from "../types";

export default function HumanReviewModal({ event, onDecision, onClose }: { event: TaskEvent; onDecision: (decision: "approve" | "reject" | "modify", feedback?: string) => void; onClose: () => void }) {
  const [editing, setEditing] = useState(false);
  const [feedback, setFeedback] = useState("");
  useEffect(() => {
    const listener = (keyboardEvent: KeyboardEvent) => { if (keyboardEvent.key === "Escape") onClose(); };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [onClose]);
  return (
    <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="review-title">
      <button className="modal-backdrop" onClick={onClose} aria-label="Close review" />
      <section className="review-modal">
        <header><div className="review-symbol">?</div><div><p className="eyebrow">Human checkpoint</p><h2 id="review-title">Review the proposed direction</h2></div><button className="modal-close" onClick={onClose}>×</button></header>
        <div className="review-content">
          <p>{event.summary || "CodeAgent needs confirmation before continuing with a potentially sensitive step."}</p>
          {event.details && <pre>{JSON.stringify(event.details, null, 2)}</pre>}
          {editing && <div className="field-group"><label htmlFor="review-feedback">Required changes</label><textarea id="review-feedback" autoFocus rows={4} value={feedback} onChange={(input) => setFeedback(input.target.value)} placeholder="Explain what should change before the agent continues." /></div>}
        </div>
        <footer>
          {editing ? <><button className="secondary-button" onClick={() => setEditing(false)}>Back</button><button className="primary-button" disabled={!feedback.trim()} onClick={() => onDecision("modify", feedback.trim())}>Send feedback</button></> : <><button className="danger-button" onClick={() => onDecision("reject")}>Reject</button><button className="secondary-button" onClick={() => setEditing(true)}>Request changes</button><button className="primary-button" onClick={() => onDecision("approve")}>Approve and continue</button></>}
        </footer>
      </section>
    </div>
  );
}
