import { useCallback, useEffect, useRef, useState } from "react";
import { importConversations, listConversations } from "../api/client";
import type { HistoryEntry } from "../types";

const STORAGE_KEY = "codeagent_task_history";

function loadLegacyHistory(): HistoryEntry[] {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as Partial<HistoryEntry>[];
    return stored.map((entry) => ({
      conversation_id: entry.conversation_id || entry.task_id || crypto.randomUUID(),
      task_id: entry.task_id || "",
      query: entry.query || "",
      timestamp: entry.timestamp || new Date().toISOString(),
      state: entry.state || "completed",
      workspace_root: entry.workspace_root,
      project_id: entry.project_id,
      messages: entry.messages || [{
        id: crypto.randomUUID(), role: "user", content: entry.query || "",
        timestamp: entry.timestamp || new Date().toISOString(), task_id: entry.task_id,
      }],
    }));
  } catch {
    return [];
  }
}

export function useHistory() {
  const [initialHistory] = useState<HistoryEntry[]>(loadLegacyHistory);
  const legacyHistory = useRef<HistoryEntry[]>(initialHistory);
  const [history, setHistory] = useState<HistoryEntry[]>(initialHistory);

  useEffect(() => {
    let cancelled = false;
    const hydrate = async () => {
      try {
        if (legacyHistory.current?.length) {
          await importConversations(legacyHistory.current);
        }
        const remote = await listConversations();
        if (!cancelled) {
          setHistory(remote);
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch {
        // Keep the legacy snapshot visible when the API is temporarily unavailable.
      }
    };
    void hydrate();
    return () => { cancelled = true; };
  }, []);

  const addEntry = useCallback((entry: HistoryEntry) => {
    setHistory((prev) => [
      entry,
      ...prev.filter((item) => item.conversation_id !== entry.conversation_id),
    ]);
  }, []);

  const removeEntry = useCallback((conversationId: string) => {
    setHistory((prev) => prev.filter(
      (entry) => entry.conversation_id !== conversationId,
    ));
  }, []);

  const updateState = useCallback((taskId: string, state: string) => {
    setHistory((prev) => {
      const current = prev.find((entry) => entry.task_id === taskId);
      if (!current || current.state === state) return prev;
      return prev.map((entry) => (
        entry.task_id === taskId ? { ...entry, state } : entry
      ));
    });
  }, []);

  return { history, addEntry, updateState, removeEntry };
}
