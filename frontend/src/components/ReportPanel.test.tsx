import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { TaskReport } from "../types";
import ReportPanel from "./ReportPanel";

describe("ReportPanel resilience evidence", () => {
  it("shows real model and infrastructure recovery counters", () => {
    const report: TaskReport = {
      task_id: "task-1",
      plan: [],
      changes: [],
      validation_results: [],
      duration: 1.2,
      token_usage: 42,
      assistant_response: "done",
      response_mode: "execute",
      memory_hits: [],
      resolved_skills: [],
      warnings: [],
      mcp_servers: [],
      model_runtime: {
        active_model: "primary/model",
        retry_count: 2,
        fallback_activated: true,
      },
      infrastructure_runtime: {
        redis: { retry_count: 1, recovery_count: 1, dropped_event_count: 0 },
        tools: {
          call_count: 3, failure_count: 1, timeout_count: 1,
          working_set: { read_requests: 4, cache_hits: 2, invalidation_count: 1 },
        },
      },
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.getByText(/primary\/model · 重试 2 次 · 已切换备用模型/)).toBeInTheDocument();
    expect(screen.getByText(/Redis 重试 1 次 · 恢复 1 次 · 丢弃事件 0 个/)).toBeInTheDocument();
    expect(screen.getByText(/工具调用 3 次 · 失败 1 次 · 超时 1 次/)).toBeInTheDocument();
    expect(screen.getByText(/Working set 读取 4 次 · 命中 2 次 · 精确失效 1 项/)).toBeInTheDocument();
  });

  it("renders downloadable real run artifacts", () => {
    const report = {
      task_id: "task-1",
      plan: [], changes: [], validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "done", response_mode: "execute" as const,
      memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
      model_runtime: {}, infrastructure_runtime: {},
      artifacts: [{
        artifact_id: "artifact-1",
        kind: "git_diff",
        size: 42,
        download_url: "/api/v1/tasks/task-1/artifacts/artifact-1",
      }],
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.getByRole("link", { name: /git_diff/ })).toHaveAttribute(
      "href", "/api/v1/tasks/task-1/artifacts/artifact-1",
    );
  });

  it("keeps capability and context manifests available in collapsed diagnostics", () => {
    const report: TaskReport = {
      task_id: "task-1",
      plan: [], changes: [], validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "done", response_mode: "execute",
      memory_hits: [], resolved_skills: [], warnings: [],
      mcp_servers: [{ name: "github", available: false, deferred: true }],
      model_runtime: {}, infrastructure_runtime: {},
      tool_manifest: {
        selected_names: ["read_file", "apply_patch"],
        deferred_names: ["delete_file"],
      },
      context_manifest: {
        strategy: "just_in_time_tools",
        sources: [{ kind: "repository_index", included: false, reason: "deferred" }],
      },
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.getByText("read_file")).toBeInTheDocument();
    expect(screen.getByText("apply_patch")).toBeInTheDocument();
    expect(screen.getByText("delete_file")).toBeInTheDocument();
    expect(screen.getByText(/repository_index/)).toBeInTheDocument();
    expect(screen.getByText("github · 已延后")).toBeInTheDocument();
  });

  it("does not expose internal memory hits in the user report", () => {
    const report: TaskReport = {
      task_id: "task-memory", plan: [], changes: [], validation_results: [],
      duration: 1, token_usage: 1, assistant_response: "done", response_mode: "execute",
      memory_hits: [{ name: "private-preference" }], resolved_skills: [], warnings: [],
      mcp_servers: [], model_runtime: {}, infrastructure_runtime: {},
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.queryByText("相关记忆")).not.toBeInTheDocument();
    expect(screen.queryByText("private-preference")).not.toBeInTheDocument();
  });

  it("renders cancelled outcome and applied steering", () => {
    const report: TaskReport = {
      task_id: "task-1", status: "cancelled", plan: [], changes: [],
      validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "Task cancelled by the user.", response_mode: "execute",
      memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
      model_runtime: {}, infrastructure_runtime: {},
      steering_instructions: ["Do not change the public API"],
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.getByText("执行已停止")).toBeInTheDocument();
    expect(screen.getByText("Do not change the public API")).toBeInTheDocument();
  });

  it("offers recovery for a cancelled run", async () => {
    const onRecover = vi.fn();
    const report: TaskReport = {
      task_id: "task-1", status: "cancelled", plan: [], changes: [],
      validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "cancelled", response_mode: "execute",
      memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
      model_runtime: {}, infrastructure_runtime: {},
    };
    render(<ReportPanel report={report} completeEvent={null} onRecover={onRecover} />);

    await userEvent.click(screen.getByRole("button", { name: "从当前工作区继续" }));

    expect(onRecover).toHaveBeenCalledOnce();
  });

  it("shows durable recovery lineage", () => {
    const report: TaskReport = {
      task_id: "task-2", status: "completed", recovered_from_task_id: "task-1",
      plan: [], changes: [], validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "done", response_mode: "execute",
      memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
      model_runtime: {}, infrastructure_runtime: {},
    };

    render(<ReportPanel report={report} completeEvent={null} />);

    expect(screen.getByText("恢复来源")).toBeInTheDocument();
    expect(screen.getByText("task-1")).toBeInTheDocument();
  });

  it("opens the changed file for review when clicked", async () => {
    const onOpenFile = vi.fn();
    const report: TaskReport = {
      task_id: "task-1",
      plan: [],
      changes: [{ file_path: "src/generator.py" }, { path: "notes.md" }],
      validation_results: [], duration: 1, token_usage: 1,
      assistant_response: "done", response_mode: "execute",
      memory_hits: [], resolved_skills: [], warnings: [], mcp_servers: [],
      model_runtime: {}, infrastructure_runtime: {},
    };

    render(<ReportPanel report={report} completeEvent={null} onOpenFile={onOpenFile} />);

    await userEvent.click(screen.getByRole("button", { name: /src\/generator\.py/ }));

    expect(onOpenFile).toHaveBeenCalledWith("src/generator.py");
  });
});
