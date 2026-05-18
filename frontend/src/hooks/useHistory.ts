import { useState } from "react";
import type { HistoryEntry } from "../types";

const STORAGE_KEY = "codeagent_task_history";
const MAX_ENTRIES = 10;

export function useHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  });

  const addEntry = (entry: HistoryEntry) => {
    setHistory((prev) => {
      const updated = [entry, ...prev.filter((h) => h.task_id !== entry.task_id)].slice(
        0,
        MAX_ENTRIES,
      );
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  };

  const updateState = (taskId: string, state: string) => {
    setHistory((prev) => {
      const updated = prev.map((h) =>
        h.task_id === taskId ? { ...h, state } : h,
      );
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  };

  return { history, addEntry, updateState };
}
