import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as api from "../api/client";
import TaskFilesPanel from "./TaskFilesPanel";

vi.mock("../api/client", () => ({
  listTaskFiles: vi.fn(),
  taskWorkspaceFileUrl: vi.fn(),
}));

const TEXT_FILE = { path: "src/generator.py", name: "generator.py", size: 120, modified: 1, binary: false };
const BINARY_FILE = { path: "logo.png", name: "logo.png", size: 900, modified: 1, binary: true };

describe("TaskFilesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.taskWorkspaceFileUrl).mockReturnValue("http://test/files/src/generator.py");
  });

  it("lists workspace files and marks binary entries", async () => {
    vi.mocked(api.listTaskFiles).mockResolvedValue([TEXT_FILE, BINARY_FILE]);

    render(<TaskFilesPanel taskId="task-1" onClose={vi.fn()} />);

    expect(await screen.findByText("generator.py")).toBeInTheDocument();
    expect(screen.getByText("logo.png")).toBeInTheDocument();
    expect(screen.getByText("二进制")).toBeInTheDocument();
  });

  it("loads and displays text file content on click", async () => {
    vi.mocked(api.listTaskFiles).mockResolvedValue([TEXT_FILE]);
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: async () => "def generate():\n    return 42",
    }) as unknown as typeof fetch;

    render(<TaskFilesPanel taskId="task-1" onClose={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: /generator\.py/ }));

    expect(await screen.findByText(/def generate/)).toBeInTheDocument();
    expect(api.taskWorkspaceFileUrl).toHaveBeenCalledWith("task-1", "src/generator.py");
  });

  it("does not fetch content for binary files", async () => {
    vi.mocked(api.listTaskFiles).mockResolvedValue([BINARY_FILE]);
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<TaskFilesPanel taskId="task-1" onClose={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: /logo\.png/ }));

    expect(await screen.findByText("二进制文件，请下载后查看。")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("auto-opens a changed file given as a workspace-relative path", async () => {
    vi.mocked(api.listTaskFiles).mockResolvedValue([TEXT_FILE]);
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: async () => "def generate():\n    return 42",
    }) as unknown as typeof fetch;

    render(<TaskFilesPanel taskId="task-1" initialPath="src/generator.py" onClose={vi.fn()} />);

    expect(await screen.findByText(/def generate/)).toBeInTheDocument();
    expect(api.taskWorkspaceFileUrl).toHaveBeenCalledWith("task-1", "src/generator.py");
  });

  it("falls back to a basename match when the changed-file path is absolute", async () => {
    vi.mocked(api.listTaskFiles).mockResolvedValue([TEXT_FILE]);
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: async () => "def generate():\n    return 42",
    }) as unknown as typeof fetch;

    render(
      <TaskFilesPanel
        taskId="task-1"
        initialPath="D:/sessions/chat-1/src/generator.py"
        onClose={vi.fn()}
      />,
    );

    expect(await screen.findByText(/def generate/)).toBeInTheDocument();
    expect(api.taskWorkspaceFileUrl).toHaveBeenCalledWith("task-1", "src/generator.py");
  });
});
