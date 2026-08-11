import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ThinkingCard from "./ThinkingCard";

describe("ThinkingCard", () => {
  it("shows real events and can collapse and expand", async () => {
    const user = userEvent.setup();
    render(<ThinkingCard events={[{
      type: "tool_call",
      timestamp: new Date().toISOString(),
      tool: "read_file",
      summary: "读取配置",
    }]} taskState="running" onHumanReview={vi.fn()} />);

    const toggle = screen.getByRole("button", { name: /正在使用 read_file/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText(/1 项活动/)).toBeInTheDocument();
    expect(screen.queryByText("读取配置")).not.toBeInTheDocument();

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("读取配置")).toBeInTheDocument();
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });

  it("keeps the final execution report inside the collapsed thinking card", async () => {
    const user = userEvent.setup();
    const { container } = render(<ThinkingCard
      events={[]}
      taskState="completed"
      report={{
        task_id: "task-1",
        plan: [],
        changes: [],
        validation_results: [],
        duration: 1,
        token_usage: 10,
        assistant_response: "done",
        response_mode: "execute",
        memory_hits: [],
        resolved_skills: [],
        warnings: [],
        mcp_servers: [],
        model_runtime: {},
        infrastructure_runtime: {},
      }}
      onHumanReview={vi.fn()}
    />);

    const toggle = container.querySelector<HTMLButtonElement>(".thinking-toggle")!;
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(container.querySelector(".report-panel")).not.toBeInTheDocument();

    await user.click(toggle);

    const report = container.querySelector(".report-panel");
    expect(report).toBeInTheDocument();
    expect(report?.closest(".thinking-card")).toBe(container.querySelector(".thinking-card"));
  });

  it("does not count transport and lifecycle bookkeeping as user-visible activity", () => {
    const timestamp = new Date().toISOString();
    render(<ThinkingCard events={[
      { type: "node_start", node: "execution", timestamp },
      { type: "node_complete", node: "execution", success: true, timestamp },
      { type: "capabilities_resolved", timestamp },
      { type: "assistant_message", content: "done", timestamp },
      { type: "task_complete", status: "success", timestamp },
    ]} taskState="completed" onHumanReview={vi.fn()} />);

    expect(screen.getByText(/2 项活动/)).toBeInTheDocument();
  });
});
