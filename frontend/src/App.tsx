import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api/client";
import CreateProjectModal from "./components/CreateProjectModal";
import HumanReviewModal from "./components/HumanReviewModal";
import PreviewPanel from "./components/PreviewPanel";
import ProjectPanel from "./components/ProjectPanel";
import TaskFilesPanel from "./components/TaskFilesPanel";
import ThinkingCard from "./components/ThinkingCard";
import { useHistory } from "./hooks/useHistory";
import { useTask } from "./hooks/useTask";
import { useWebSocket } from "./hooks/useWebSocket";
import type { BenchmarkTask, ChatMessage, HistoryEntry, Project, ProjectFile, SystemCapabilities, TaskEvent, TaskPreview, TaskState } from "./types";

const MarkdownMessage = lazy(() => import("./components/MarkdownMessage"));

type IconName =
  | "add" | "arrow" | "book" | "check" | "chevron" | "code" | "history"
  | "menu" | "plus" | "refresh" | "settings" | "spark" | "stop" | "tasks" | "trash" | "x";

const ICON_PATHS: Record<IconName, string> = {
  add: "M12 5v14M5 12h14",
  arrow: "m5 12 7-7 7 7M12 5v14",
  book: "M4 5.5A2.5 2.5 0 0 1 6.5 3H11v16H6.5A2.5 2.5 0 0 0 4 21V5.5Zm16 0A2.5 2.5 0 0 0 17.5 3H13v16h4.5A2.5 2.5 0 0 1 20 21V5.5Z",
  check: "m5 12 4 4L19 6",
  chevron: "m8 10 4 4 4-4",
  code: "m8 9-4 3 4 3m8-6 4 3-4 3m-5-9-2 18",
  history: "M3 12a9 9 0 1 0 3-6.7L3 8m0-5v5h5m4-1v5l3 2",
  menu: "M4 7h16M4 12h16M4 17h16",
  plus: "M12 5v14M5 12h14",
  refresh: "M20 6v5h-5M4 18v-5h5M18.5 9A7 7 0 0 0 6 6.5M5.5 15A7 7 0 0 0 18 17.5",
  settings: "M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7ZM19 13.5l1.5 1.2-2 3.4-1.8-.7a8 8 0 0 1-2.2 1.3l-.3 1.9h-4l-.3-1.9a8 8 0 0 1-2.2-1.3l-1.8.7-2-3.4L5 13.5a8 8 0 0 1 0-2.6L3.5 9.7l2-3.4 1.8.7a8 8 0 0 1 2.2-1.3l.3-1.9h4l.3 1.9A8 8 0 0 1 16.3 7l1.8-.7 2 3.4-1.5 1.2a8 8 0 0 1 .4 2.6Z",
  spark: "m12 3 1.4 4.6L18 9l-4.6 1.4L12 15l-1.4-4.6L6 9l4.6-1.4L12 3Zm6 11 .8 2.2L21 17l-2.2.8L18 20l-.8-2.2L15 17l2.2-.8L18 14Z",
  stop: "M8 8h8v8H8z",
  tasks: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01",
  trash: "M4 7h16M9 7V4h6v3m-9 0 1 14h10l1-14M10 11v6m4-6v6",
  x: "M6 6l12 12M18 6 6 18",
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true"><path d={ICON_PATHS[name]} /></svg>;
}

const STATE_LABELS: Record<string, string> = {
  idle: "就绪", pending: "排队中", running: "执行中", waiting_review: "等待确认",
  completed: "已完成", failed: "失败", cancelled: "已取消",
};

function StatusDot({ state }: { state: TaskState | "idle" }) {
  return <span className={`run-status state-${state}`}><i />{STATE_LABELS[state]}</span>;
}

function Composer({
  value, onChange, onSubmit, onSteer, onCancel, running, loading, canSubmit, selectedTask,
  onClearTask, runMode, onRunMode, onOpenCatalog, onOpenEnvironment,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onSteer: () => void;
  onCancel: () => void;
  running: boolean;
  loading: boolean;
  canSubmit: boolean;
  selectedTask: BenchmarkTask | null;
  onClearTask: () => void;
  runMode: "quick" | "standard" | "review";
  onRunMode: (mode: "quick" | "standard" | "review") => void;
  onOpenCatalog: () => void;
  onOpenEnvironment: () => void;
}) {
  const [plusOpen, setPlusOpen] = useState(false);
  const [modeOpen, setModeOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 58), 180)}px`;
  }, [value]);

  return (
    <div className="composer-wrap">
      {selectedTask && (
        <div className="selected-context">
          <Icon name="tasks" size={14} /><span>{selectedTask.instance_id}</span>
          <button onClick={onClearTask} aria-label="清除 SWE 任务"><Icon name="x" size={13} /></button>
        </div>
      )}
      <div className="chat-composer">
        <textarea
          ref={textareaRef}
          aria-label="向 AGENT4CODE 发送任务"
          value={value}
          maxLength={10000}
          rows={2}
          placeholder={running ? "追加运行指令，Enter 发送；停止按钮会中断任务" : "描述你想实现或修复的内容，Enter 发送，Shift + Enter 换行"}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (running) {
                if (value.trim()) onSteer();
              } else if (canSubmit) onSubmit();
            }
          }}
        />
        <div className="composer-toolbar">
          <div className="toolbar-left">
            <div className="menu-anchor">
              <button className="round-tool" onClick={() => setPlusOpen((open) => !open)} aria-expanded={plusOpen} aria-label="添加上下文">
                <Icon name="plus" />
              </button>
              {plusOpen && (
                <div className="composer-menu">
                  <button onClick={() => { setPlusOpen(false); onOpenCatalog(); }}><Icon name="tasks" />选择 SWE Smoke 任务</button>
                  <button onClick={() => { setPlusOpen(false); onOpenEnvironment(); }}><Icon name="settings" />检查运行环境</button>
                </div>
              )}
            </div>
            <span className="workspace-note"><Icon name="check" size={13} />自动创建安全工作区</span>
          </div>
          <div className="toolbar-right">
            <div className="menu-anchor">
              <button className="mode-button" onClick={() => setModeOpen((open) => !open)} aria-expanded={modeOpen}>
                {runMode === "quick" ? "快速" : runMode === "standard" ? "标准" : "需确认"}<Icon name="chevron" size={14} />
              </button>
              {modeOpen && (
                <div className="composer-menu mode-menu">
                  <button onClick={() => { onRunMode("quick"); setModeOpen(false); }}><strong>快速</strong><small>较少重试，自动执行</small></button>
                  <button onClick={() => { onRunMode("standard"); setModeOpen(false); }}><strong>标准</strong><small>完整规划和验证</small></button>
                  <button onClick={() => { onRunMode("review"); setModeOpen(false); }}><strong>需确认</strong><small>敏感步骤等待审核</small></button>
                </div>
              )}
            </div>
            {running && <button className="send-button cancel" onClick={onCancel} aria-label="停止任务"><Icon name="stop" size={17} /></button>}
            <button
              className="send-button"
              onClick={running ? onSteer : onSubmit}
              disabled={running ? !value.trim() : (!canSubmit || loading)}
              aria-label={running ? "追加运行指令" : "发送任务"}
            >
              <Icon name="arrow" size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [catalog, setCatalog] = useState<BenchmarkTask[]>([]);
  const [selectedInstance, setSelectedInstance] = useState("");
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);
  const [environmentError, setEnvironmentError] = useState<string | null>(null);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [environmentOpen, setEnvironmentOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [reviewEvent, setReviewEvent] = useState<TaskEvent | null>(null);
  const [runMode, setRunMode] = useState<"quick" | "standard" | "review">("standard");
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>([]);
  const [projectPanelOpen, setProjectPanelOpen] = useState(false);
  const [projectModalOpen, setProjectModalOpen] = useState(false);
  const [projectBusy, setProjectBusy] = useState(false);
  const [taskPreview, setTaskPreview] = useState<TaskPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  const [filesInitialPath, setFilesInitialPath] = useState<string | null>(null);
  const [expandedThinkingTasks, setExpandedThinkingTasks] = useState<Set<string>>(() => new Set());
  const previousEvents = useRef<{ taskId: string | null; count: number }>({ taskId: null, count: 0 });
  const deliveredReports = useRef(new Set<string>());
  const autoOpenedPreviewTasks = useRef(new Set<string>());
  const requestedPreviewTask = useRef<string | null>(null);
  const previewAutoOpenSuppressed = useRef(false);

  const tasks = useTask();
  const stream = useWebSocket(tasks.taskId);
  const { history, addEntry, updateState, removeEntry } = useHistory();
  const activeProject = useMemo(
    () => projects.find((project) => project.project_id === activeProjectId) || null,
    [activeProjectId, projects],
  );

  const refreshEnvironment = useCallback(async () => {
    try {
      const [catalogResult, capabilityResult] = await Promise.all([
        api.getBenchmarkCatalog(), api.getSystemCapabilities(tasks.workspaceRoot || undefined, query || undefined),
      ]);
      setCatalog(catalogResult.tasks);
      setCapabilities(capabilityResult);
      setEnvironmentError(null);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "无法连接 API");
    }
  }, [query, tasks.workspaceRoot]);

  useEffect(() => {
    void Promise.all([api.getBenchmarkCatalog(), api.getSystemCapabilities(), api.listProjects()])
      .then(([catalogResult, capabilityResult, projectResult]) => {
        setCatalog(catalogResult.tasks);
        setCapabilities(capabilityResult);
        setProjects(projectResult);
        setEnvironmentError(null);
      })
      .catch((error: unknown) => {
        setEnvironmentError(error instanceof Error ? error.message : "无法连接 API");
      });
  }, []);

  const refreshProjectFiles = useCallback(async (projectId: string) => {
    try {
      setProjectFiles(await api.listProjectFiles(projectId));
      setEnvironmentError(null);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "无法读取项目文件");
    }
  }, []);

  const refreshTaskPreview = useCallback(async (taskId: string, autoOpen = false) => {
    requestedPreviewTask.current = taskId;
    try {
      const preview = await api.getTaskPreview(taskId);
      if (requestedPreviewTask.current !== taskId) return;
      setTaskPreview(preview.available ? preview : null);
      if (
        preview.available
        && autoOpen
        && !previewAutoOpenSuppressed.current
        && !autoOpenedPreviewTasks.current.has(taskId)
      ) {
        autoOpenedPreviewTasks.current.add(taskId);
        setProjectPanelOpen(false);
        setPreviewOpen(true);
      }
    } catch {
      if (requestedPreviewTask.current === taskId) setTaskPreview(null);
    }
  }, []);

  useEffect(() => {
    if (previousEvents.current.taskId !== tasks.taskId) {
      previousEvents.current = { taskId: tasks.taskId, count: 0 };
    }
    for (const event of stream.events.slice(previousEvents.current.count)) {
      if (event.type === "human_review_required") setReviewEvent(event);
      if (event.type === "task_complete" && tasks.taskId) updateState(tasks.taskId, "completed");
      if (event.type === "task_error" && tasks.taskId) updateState(tasks.taskId, "failed");
      if (event.type === "task_cancelled" && tasks.taskId) updateState(tasks.taskId, "cancelled");
    }
    previousEvents.current.count = stream.events.length;
  }, [stream.events, tasks.taskId, updateState]);

  useEffect(() => {
    if (tasks.status && tasks.taskId) updateState(tasks.taskId, tasks.status.state);
  }, [tasks.status, tasks.taskId, updateState]);

  const selectedTask = useMemo(
    () => catalog.find((task) => task.instance_id === selectedInstance) || null,
    [catalog, selectedInstance],
  );
  const state = tasks.status?.state || (tasks.taskId ? "pending" : "idle");
  const running = state === "pending" || state === "running" || state === "waiting_review";
  const currentReport = tasks.finalReport?.task_id === tasks.taskId ? tasks.finalReport : null;
  const canSubmit = !!query.trim() && !!capabilities?.llm_configured && !preparing;
  const hasConversation = !!messages.length || !!tasks.taskId;
  const hasCurrentAssistantMessage = !!tasks.taskId && messages.some(
    (message) => message.role === "assistant" && message.task_id === tasks.taskId,
  );
  const thinkingExpanded = !!tasks.taskId && expandedThinkingTasks.has(tasks.taskId);
  const completeEvent = useMemo(
    () => [...stream.events].reverse().find(
      (event) => event.type === "task_complete" || event.type === "task_error" || event.type === "task_cancelled",
    ) || null,
    [stream.events],
  );
  const previewWriteSignal = useMemo(
    () => stream.events
      .filter((event) => event.tool === "write_file")
      .map((event) => event.event_id || `${event.seq || 0}:${event.timestamp}`)
      .join("|"),
    [stream.events],
  );
  const assistantEvent = useMemo(
    () => [...stream.events].reverse().find((event) => event.type === "assistant_message" && event.content) || null,
    [stream.events],
  );

  // Deliver the agent's answer as soon as it is available. Priority: the live
  // `assistant_message` stream event (published at the end of inline runs) →
  // the persisted report's assistant_response → a synthesized terminal reply
  // (failure/cancellation), so the front end never leaves the user with an
  // empty answer even when the report lags behind the terminal status.
  useEffect(() => {
    if (!tasks.taskId || !conversationId) return;
    const taskId = tasks.taskId;
    if (deliveredReports.current.has(taskId)) return;
    let content = assistantEvent?.content || "";
    if (!content && currentReport?.assistant_response) content = currentReport.assistant_response;
    if (!content && completeEvent?.type === "task_error") {
      content = completeEvent.error ? `任务未能完成：${completeEvent.error}` : "任务执行未通过，请查看执行报告。";
    }
    if (!content && completeEvent?.type === "task_cancelled") content = "任务已取消。";
    if (!content) return;
    deliveredReports.current.add(taskId);
    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(), role: "assistant", content,
      timestamp: new Date().toISOString(), task_id: taskId,
    };
    setMessages((current) => {
      if (current.some((item) => item.role === "assistant" && item.task_id === taskId)) return current;
      const updated = [...current, assistantMessage];
      const title = updated.find((item) => item.role === "user")?.content || "新会话";
      addEntry({
        conversation_id: conversationId as string, task_id: taskId, query: title,
        timestamp: new Date().toISOString(), state: tasks.status?.state || "completed",
        workspace_root: tasks.workspaceRoot || undefined, project_id: activeProjectId || undefined, messages: updated,
      });
      if (activeProjectId) void refreshProjectFiles(activeProjectId);
      return updated;
    });
  }, [activeProjectId, addEntry, assistantEvent, completeEvent, conversationId, currentReport, refreshProjectFiles, tasks.status?.state, tasks.taskId, tasks.workspaceRoot]);

  // Completed / failed / cancelled tasks expand their execution card by default
  // so the report and generated files are immediately visible without a click.
  useEffect(() => {
    if (!tasks.taskId) return;
    if (state === "completed" || state === "failed" || state === "cancelled") {
      const taskId = tasks.taskId;
      const timer = window.setTimeout(() => {
        setExpandedThinkingTasks((current) => {
          if (current.has(taskId)) return current;
          const next = new Set(current);
          next.add(taskId);
          return next;
        });
      }, 0);
      return () => window.clearTimeout(timer);
    }
  }, [state, tasks.taskId]);

  useEffect(() => {
    if (!tasks.taskId) {
      requestedPreviewTask.current = null;
      return;
    }
    const taskId = tasks.taskId;
    const timer = window.setTimeout(
      () => void refreshTaskPreview(taskId, true),
      previewWriteSignal ? 180 : 0,
    );
    return () => window.clearTimeout(timer);
  }, [previewWriteSignal, refreshTaskPreview, currentReport?.task_id, tasks.taskId]);
  const setThinkingExpanded = (expanded: boolean) => {
    if (!tasks.taskId) return;
    setExpandedThinkingTasks((current) => {
      const next = new Set(current);
      if (expanded) next.add(tasks.taskId as string);
      else next.delete(tasks.taskId as string);
      return next;
    });
  };

  const openPreview = () => {
    previewAutoOpenSuppressed.current = false;
    setProjectPanelOpen(false);
    setFilesOpen(false);
    setPreviewOpen(true);
  };

  const closePreview = () => {
    previewAutoOpenSuppressed.current = true;
    setPreviewOpen(false);
  };

  const openTaskFiles = (initialPath?: string | null) => {
    setFilesInitialPath(initialPath || null);
    setPreviewOpen(false);
    setProjectPanelOpen(false);
    setFilesOpen(true);
  };

  const closeTaskFiles = () => {
    setFilesOpen(false);
    setFilesInitialPath(null);
  };

  const newChat = useCallback(() => {
    tasks.resetTask();
    setQuery("");
    setConversationId(null);
    setMessages([]);
    setSelectedInstance("");
    setReviewEvent(null);
    requestedPreviewTask.current = null;
    previewAutoOpenSuppressed.current = false;
    setTaskPreview(null);
    setPreviewOpen(false);
    setFilesOpen(false);
    setFilesInitialPath(null);
    setSidebarOpen(false);
  }, [tasks]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        newChat();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [newChat]);

  const selectBenchmark = (task: BenchmarkTask) => {
    setSelectedInstance(task.instance_id);
    setQuery(task.problem_statement);
    setCatalogOpen(false);
  };

  const submit = async () => {
    const prompt = query.trim();
    if (!prompt || !capabilities?.llm_configured || running) return;
    setReviewEvent(null);
    setEnvironmentError(null);
    requestedPreviewTask.current = null;
    setTaskPreview(null);
    setPreviewOpen(false);
    let projectRoot: string | undefined = activeProject?.workspace_root;
    if (selectedTask) {
      projectRoot = selectedTask.workspace_path;
      if (!selectedTask.workspace_ready) {
        setPreparing(true);
        try {
          const prepared = await api.prepareBenchmarkWorkspace(selectedTask.instance_id);
          projectRoot = prepared.project_root;
          setCatalog((current) => current.map((item) =>
            item.instance_id === selectedTask.instance_id ? { ...item, workspace_ready: true } : item,
          ));
        } catch (error) {
          setEnvironmentError(error instanceof Error ? error.message : "SWE 工作区准备失败");
          setPreparing(false);
          return;
        }
        setPreparing(false);
      }
    }
    const id = conversationId || crypto.randomUUID();
    const options = {
      projectRoot,
      autoMode: runMode !== "review",
      maxRetries: runMode === "quick" ? 2 : 3,
      conversationHistory: messages.map(({ role, content }) => ({ role, content })),
      directExecution: messages.length > 0,
      conversationId: id,
      ...(selectedTask ? { benchmarkInstanceId: selectedTask.instance_id } : {}),
    };
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(), role: "user", content: prompt,
      timestamp: new Date().toISOString(),
    };
    const pendingMessages = [...messages, userMessage];
    setConversationId(id);
    setMessages(pendingMessages);
    setQuery("");
    const result = await tasks.createTask(prompt, options);
    if (result) {
      const updatedMessages = pendingMessages.map((message) =>
        message.id === userMessage.id ? { ...message, task_id: result.task_id } : message,
      );
      setMessages(updatedMessages);
      setQuery("");
      addEntry({
        conversation_id: id, task_id: result.task_id,
        query: pendingMessages.find((item) => item.role === "user")?.content || prompt,
        timestamp: new Date().toISOString(), state: "pending",
        workspace_root: result.project_root, project_id: activeProjectId || undefined, messages: updatedMessages,
      });
    }
  };

  const steer = async () => {
    const instruction = query.trim();
    if (!instruction || !running || !tasks.taskId) return;
    const queued = await tasks.steerTask(instruction);
    if (!queued) return;
    setMessages((current) => [...current, {
      id: crypto.randomUUID(), role: "user", content: instruction,
      timestamp: new Date().toISOString(), task_id: tasks.taskId as string,
    }]);
    setQuery("");
  };

  const recover = async () => {
    if (!tasks.taskId || !["failed", "cancelled"].includes(state)) return;
    const result = await tasks.recoverTask(undefined, {
      autoMode: runMode !== "review",
      maxRetries: runMode === "quick" ? 2 : 3,
    });
    if (!result) return;
    setExpandedThinkingTasks((current) => new Set(current).add(result.task_id));
    if (conversationId) {
      addEntry({
        conversation_id: conversationId,
        task_id: result.task_id,
        query: messages.find((item) => item.role === "user")?.content || "恢复未完成任务",
        timestamp: new Date().toISOString(),
        state: "pending",
        workspace_root: result.project_root,
        project_id: activeProjectId || undefined,
        messages,
      });
    }
  };

  const decide = async (decision: "approve" | "reject" | "modify", feedback?: string) => {
    setReviewEvent(null);
    await tasks.handleDecision(decision, feedback ? { feedback } : undefined);
  };

  const chooseHistory = (entry: HistoryEntry) => {
    setActiveProjectId(entry.project_id || null);
    setProjectPanelOpen(!!entry.project_id);
    if (entry.project_id) void refreshProjectFiles(entry.project_id);
    else setProjectFiles([]);
    setConversationId(entry.conversation_id);
    setMessages(entry.messages);
    setQuery("");
    setReviewEvent(null);
    requestedPreviewTask.current = null;
    previewAutoOpenSuppressed.current = false;
    setTaskPreview(null);
    setPreviewOpen(false);
    tasks.selectTask(entry.task_id, entry.workspace_root);
    setSidebarOpen(false);
  };

  const deleteHistory = async (entry: HistoryEntry) => {
    try {
      if (entry.conversation_id === conversationId) {
        if (running) await tasks.cancelTask();
        newChat();
      }
      await api.deleteConversation(entry.conversation_id, entry.workspace_root);
      removeEntry(entry.conversation_id);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "删除会话失败");
    }
  };

  const selectProject = (project: Project) => {
    if (running) return;
    tasks.resetTask();
    setActiveProjectId(project.project_id);
    setProjectPanelOpen(true);
    void refreshProjectFiles(project.project_id);
    setConversationId(null);
    setMessages([]);
    setQuery("");
    setSelectedInstance("");
    setReviewEvent(null);
    requestedPreviewTask.current = null;
    previewAutoOpenSuppressed.current = false;
    setTaskPreview(null);
    setPreviewOpen(false);
    setSidebarOpen(false);
  };

  const createProject = async (name: string) => {
    setProjectBusy(true);
    try {
      const project = await api.createProject(name);
      setProjects((current) => [project, ...current]);
      setProjectModalOpen(false);
      selectProject(project);
      setEnvironmentError(null);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "创建项目失败");
    } finally {
      setProjectBusy(false);
    }
  };

  const uploadProjectFiles = async (files: File[]) => {
    if (!activeProjectId) return;
    setProjectBusy(true);
    try {
      await api.uploadProjectFiles(activeProjectId, files);
      await refreshProjectFiles(activeProjectId);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "上传文件失败");
    } finally {
      setProjectBusy(false);
    }
  };

  const deleteProjectFile = async (file: ProjectFile) => {
    if (!activeProjectId) return;
    try {
      await api.deleteProjectFile(activeProjectId, file.path);
      await refreshProjectFiles(activeProjectId);
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "删除文件失败");
    }
  };

  const deleteActiveProject = async () => {
    if (!activeProject || !window.confirm(`确定删除项目“${activeProject.name}”及其全部托管文件吗？`)) return;
    setProjectBusy(true);
    try {
      if (running) await tasks.cancelTask();
      await api.deleteProject(activeProject.project_id);
      history.filter((entry) => entry.project_id === activeProject.project_id).forEach((entry) => removeEntry(entry.conversation_id));
      setProjects((current) => current.filter((project) => project.project_id !== activeProject.project_id));
      setActiveProjectId(null);
      setProjectPanelOpen(false);
      setProjectFiles([]);
      newChat();
    } catch (error) {
      setEnvironmentError(error instanceof Error ? error.message : "删除项目失败");
    } finally {
      setProjectBusy(false);
    }
  };

  return (
    <div className={`chat-app ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      {sidebarOpen && <button className="mobile-scrim" onClick={() => setSidebarOpen(false)} aria-label="关闭侧栏" />}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-head">
          <div className="brand-lockup">
            <button className="mini-logo" onClick={newChat} aria-label="AGENT4CODE 首页">A</button>
            <div><strong>AGENT4CODE</strong><span>Developer workspace</span></div>
          </div>
          <button className="icon-button" onClick={() => { setSidebarOpen(false); setSidebarCollapsed(true); }} aria-label="收起侧栏"><Icon name="menu" /></button>
        </div>
        <button className="new-chat" onClick={newChat}><Icon name="add" /><span>新建会话</span><kbd>Ctrl K</kbd></button>
        <nav className="primary-nav" aria-label="主要功能">
          <button onClick={newChat}><Icon name="code" /><span>Agent 工作台</span></button>
          <button onClick={() => setCatalogOpen(true)}><Icon name="tasks" /><span>SWE 任务集</span><em>{catalog.length}</em></button>
          <button onClick={() => setEnvironmentOpen(true)}><Icon name="settings" /><span>运行环境</span></button>
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer"><Icon name="book" /><span>API 文档</span></a>
        </nav>
        <div className="sidebar-section-title project-section-title">
          <span>项目</span>
          <button type="button" onClick={() => setProjectModalOpen(true)} aria-label="新建项目"><Icon name="plus" size={15} /></button>
        </div>
        <div className="project-list">
          <button type="button" className="new-project-row" onClick={() => setProjectModalOpen(true)}><Icon name="plus" size={15} />新建项目</button>
          {projects.map((project) => (
            <button
              type="button"
              key={project.project_id}
              className={activeProjectId === project.project_id ? "active" : ""}
              onClick={() => selectProject(project)}
            >
              <Icon name="book" size={15} /><span>{project.name}</span>
            </button>
          ))}
        </div>
        <div className="sidebar-section-title"><span>最近会话</span><Icon name="history" size={15} /></div>
        <div className="conversation-list">
          {history.length ? history.map((entry) => (
            <div key={entry.conversation_id} className={`conversation-row ${conversationId === entry.conversation_id ? "active" : ""}`}>
              <button className="conversation-select" onClick={() => chooseHistory(entry)}>
                <span>{entry.query}</span><small>{STATE_LABELS[entry.state] || entry.state}</small>
              </button>
              <button className="conversation-delete" onClick={() => void deleteHistory(entry)} aria-label={`删除会话：${entry.query}`}>
                <Icon name="trash" size={15} />
              </button>
            </div>
          )) : <p>发送第一条任务后，会话会显示在这里。</p>}
        </div>
        <div className="sidebar-foot">
          <div className="avatar">A4</div><div><strong>AGENT4CODE</strong><small>{capabilities?.model || "等待配置模型"}</small></div>
          <button className="icon-button" onClick={() => setEnvironmentOpen(true)} aria-label="打开设置"><Icon name="settings" size={17} /></button>
        </div>
      </aside>

      <main className={`chat-main ${hasConversation ? "conversation-active" : ""} ${activeProject && projectPanelOpen ? "project-open" : ""} ${previewOpen && taskPreview ? "preview-open" : ""}`}>
        <header className="chat-topbar">
          <button className="mobile-menu" onClick={() => { setSidebarOpen(true); setSidebarCollapsed(false); }} aria-label="打开侧栏"><Icon name="menu" /></button>
          <button
            type="button"
            className={`topbar-title ${activeProject ? "project-title" : ""}`}
            onClick={() => activeProject && setProjectPanelOpen(true)}
          >
            {activeProject ? `▱ ${activeProject.name}` : hasConversation ? "AGENT4CODE 会话" : "智能开发工作台"}
          </button>
          {taskPreview && tasks.taskId && (
            <button className="preview-pill" onClick={openPreview}>
              <span />页面预览
            </button>
          )}
          {tasks.taskId && !filesOpen && (
            <button className="preview-pill" onClick={() => openTaskFiles()}>
              <span />任务文件
            </button>
          )}
          <button className="runtime-pill" onClick={() => setEnvironmentOpen(true)}>
            <span className={capabilities?.llm_configured ? "ready" : "blocked"} />
            {capabilities?.llm_configured ? "环境就绪" : "需要配置模型"}
          </button>
        </header>

        {!hasConversation ? (
          <section className="welcome-view">
            <div className="welcome-content">
              <div className="welcome-kicker"><span />AI DEVELOPMENT WORKSPACE</div>
              <h1>AGENT<span>4</span>CODE</h1>
              <p>说出需求，剩下的交给 Agent。</p>
              <Composer
                value={query} onChange={setQuery} onSubmit={() => void submit()} onSteer={() => void steer()} onCancel={() => void tasks.cancelTask()}
                running={running} loading={tasks.loading || preparing} canSubmit={canSubmit} selectedTask={selectedTask}
                onClearTask={() => setSelectedInstance("")} runMode={runMode} onRunMode={setRunMode}
                onOpenCatalog={() => setCatalogOpen(true)} onOpenEnvironment={() => setEnvironmentOpen(true)}
              />
              <div className="quick-actions">
                <button onClick={() => setQuery("定位并修复当前代码中的问题，补充回归测试并运行相关验证。")}>修复问题</button>
                <button onClick={() => setQuery("重构当前实现，保持现有行为并改善可读性与测试覆盖。")}>重构代码</button>
                <button onClick={() => setQuery("为当前项目补充关键路径的自动化测试，并修复发现的问题。")}>编写测试</button>
                <button onClick={() => setCatalogOpen(true)}>SWE Smoke</button>
              </div>
            </div>
            {!capabilities?.llm_configured && (
              <button className="setup-hint" onClick={() => setEnvironmentOpen(true)}>
                <Icon name="spark" /><span><strong>开始前需要配置模型</strong>在 .env 中填写 LLM_API_KEY 后重启 Harness</span>
              </button>
            )}
          </section>
        ) : (
          <section className="conversation-view">
            <div className="message-stream">
              {messages.map((message) => (
                <article key={message.id} className={`message ${message.role === "user" ? "user-message" : "assistant-chat-message"}`}>
                  <div className={`message-avatar ${message.role === "assistant" ? "agent-avatar" : ""}`}>{message.role === "user" ? "你" : "A"}</div>
                  <div className="message-bubble">
                    {message.role === "assistant" && message.task_id === tasks.taskId && (
                      <ThinkingCard
                        events={stream.events}
                        taskState={state}
                        report={currentReport}
                        completeEvent={completeEvent}
                        onHumanReview={setReviewEvent}
                        onRecover={() => void recover()}
                        onOpenFile={openTaskFiles}
                        expanded={thinkingExpanded}
                        onExpandedChange={setThinkingExpanded}
                      />
                    )}
                    <div className="assistant-answer">
                      <Suspense fallback={<span className="markdown-fallback">{message.content}</span>}><MarkdownMessage content={message.content} /></Suspense>
                    </div>
                  </div>
                </article>
              ))}
              {tasks.taskId && !hasCurrentAssistantMessage && <article className="message agent-message">
                <div className="message-avatar agent-avatar">A</div>
                <div className="agent-response">
                  <div className="agent-response-head"><div><strong>AGENT4CODE</strong><StatusDot state={state} />{running && <span className={`stream-state ${stream.connected ? "connected" : "reconnecting"}`}>{stream.connected ? "实时" : stream.phase === "reconnecting" ? `重连 ${stream.reconnectAttempt || 1}` : "连接中"}</span>}</div>{tasks.taskId && <code>{tasks.taskId.slice(0, 8)}</code>}</div>
                  <ThinkingCard
                    events={stream.events}
                    taskState={state}
                    report={currentReport}
                    completeEvent={completeEvent}
                    onHumanReview={setReviewEvent}
                    onRecover={() => void recover()}
                    onOpenFile={openTaskFiles}
                    expanded={thinkingExpanded}
                    onExpandedChange={setThinkingExpanded}
                  />
                  {tasks.error && <div className="inline-alert"><strong>任务无法继续</strong><span>{tasks.error}</span></div>}
                  {stream.error && running && <div className="inline-alert warning"><strong>执行流重连中</strong><span>{stream.error}</span></div>}
                  {environmentError && <div className="inline-alert warning"><strong>环境需要处理</strong><span>{environmentError}</span></div>}
                </div>
              </article>}
            </div>
            <div className="conversation-composer">
              <Composer
                value={query} onChange={setQuery} onSubmit={() => void submit()} onSteer={() => void steer()} onCancel={() => void tasks.cancelTask()}
                running={running} loading={tasks.loading || preparing} canSubmit={canSubmit} selectedTask={selectedTask}
                onClearTask={() => setSelectedInstance("")} runMode={runMode} onRunMode={setRunMode}
                onOpenCatalog={() => setCatalogOpen(true)} onOpenEnvironment={() => setEnvironmentOpen(true)}
              />
              <small>AGENT4CODE 可能会犯错，请检查生成的变更和验证结果。</small>
            </div>
          </section>
        )}
      </main>

      {taskPreview && previewOpen && tasks.taskId && (
        <PreviewPanel
          taskId={tasks.taskId}
          preview={taskPreview}
          onRefresh={() => void refreshTaskPreview(tasks.taskId as string)}
          onClose={closePreview}
        />
      )}

      {filesOpen && tasks.taskId && (
        <TaskFilesPanel
          taskId={tasks.taskId}
          initialPath={filesInitialPath}
          onClose={closeTaskFiles}
        />
      )}

      {activeProject && projectPanelOpen && !previewOpen && (
        <ProjectPanel
          project={activeProject}
          files={projectFiles}
          busy={projectBusy}
          onUpload={(files) => void uploadProjectFiles(files)}
          onDeleteFile={(file) => void deleteProjectFile(file)}
          onDeleteProject={() => void deleteActiveProject()}
          onClose={() => setProjectPanelOpen(false)}
        />
      )}

      {projectModalOpen && (
        <CreateProjectModal busy={projectBusy} onCreate={(name) => void createProject(name)} onClose={() => setProjectModalOpen(false)} />
      )}

      {catalogOpen && (
        <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="catalog-title">
          <button className="overlay-backdrop" onClick={() => setCatalogOpen(false)} aria-label="关闭任务集" />
          <section className="sheet catalog-sheet">
            <header><div><span className="sheet-kicker">SWE-bench Lite</span><h2 id="catalog-title">选择 Smoke 任务</h2><p>选择后只需发送，系统会自动准备固定版本的工作区。</p></div><button className="icon-button" onClick={() => setCatalogOpen(false)}><Icon name="x" /></button></header>
            <div className="catalog-list">
              {catalog.map((task) => (
                <button key={task.instance_id} className={selectedInstance === task.instance_id ? "selected" : ""} onClick={() => selectBenchmark(task)}>
                  <div><strong>{task.instance_id}</strong><span>{task.repo}</span></div><small className={task.workspace_ready ? "ready" : ""}>{task.workspace_ready ? "已准备" : "发送时准备"}</small>
                </button>
              ))}
            </div>
          </section>
        </div>
      )}

      {environmentOpen && (
        <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="environment-title">
          <button className="overlay-backdrop" onClick={() => setEnvironmentOpen(false)} aria-label="关闭环境面板" />
          <section className="sheet environment-sheet">
            <header><div><span className="sheet-kicker">System</span><h2 id="environment-title">运行环境</h2><p>任务发送前的能力与连接状态。</p></div><button className="icon-button" onClick={() => setEnvironmentOpen(false)}><Icon name="x" /></button></header>
            <div className="capability-list">
              {[
                ["LLM", !!capabilities?.llm_configured, capabilities?.llm_configured ? capabilities.model : "在 .env 中配置 LLM_API_KEY"],
                ["模型韧性", true, capabilities?.model_resilience?.fallback_configured ? "重试、熔断与备用模型已配置" : "重试与熔断已启用 · 未配置备用模型"],
                ["Agent 工具", !!capabilities?.tools.length, `${capabilities?.tools.length || 0} 个工具可用`],
                ["Skills", !!capabilities?.skills.configured, capabilities?.skills.detail || `${capabilities?.skills.count || 0} 个产品 Skill`],
                ["MCP", !!capabilities?.mcp.available, capabilities?.mcp.detail || "未配置"],
              ].map(([label, enabled, detail]) => (
                <div className="capability-item" key={String(label)}><i className={enabled ? "on" : ""} /><div><strong>{String(label)}</strong><span>{String(detail)}</span></div></div>
              ))}
            </div>
            {environmentError && <div className="inline-alert warning"><span>{environmentError}</span></div>}
            <button className="refresh-button" onClick={() => void refreshEnvironment()}><Icon name="refresh" />刷新状态</button>
          </section>
        </div>
      )}

      {reviewEvent && <HumanReviewModal event={reviewEvent} onDecision={decide} onClose={() => setReviewEvent(null)} />}
    </div>
  );
}
