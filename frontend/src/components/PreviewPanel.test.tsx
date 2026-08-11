import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import PreviewPanel from "./PreviewPanel";

describe("PreviewPanel", () => {
  it("renders a sandboxed real workspace preview and supports entry switching", async () => {
    const user = userEvent.setup();
    render(<PreviewPanel
      taskId="task-1"
      preview={{
        available: true,
        entrypoint: "index.html",
        entries: ["index.html", "docs/demo.html"],
        revision: "42",
      }}
      onRefresh={vi.fn()}
      onClose={vi.fn()}
    />);

    const frame = screen.getByTitle("实时预览：index.html");
    expect(frame).toHaveAttribute(
      "src",
      "/api/v1/tasks/task-1/preview/index.html?revision=42",
    );
    expect(frame).toHaveAttribute("sandbox", "allow-scripts allow-forms allow-modals");

    await user.selectOptions(screen.getByRole("combobox", { name: "选择 HTML 入口" }), "docs/demo.html");
    expect(screen.getByTitle("实时预览：docs/demo.html")).toHaveAttribute(
      "src",
      "/api/v1/tasks/task-1/preview/docs/demo.html?revision=42",
    );
  });
});
