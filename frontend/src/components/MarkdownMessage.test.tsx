import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import MarkdownMessage from "./MarkdownMessage";

describe("MarkdownMessage", () => {
  beforeEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("renders GFM structure, safe links, and highlighted fenced code", () => {
    render(<MarkdownMessage content={`# 修复结果

- [x] 已完成
- [ ] 待确认

| 文件 | 状态 |
| --- | --- |
| app.py | 通过 |

[文档](https://example.com)

\`\`\`python
def answer():
    return 42
\`\`\``} />);

    expect(screen.getByRole("heading", { name: "修复结果" })).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "文档" })).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByText("def", { selector: ".hljs-keyword" })).toBeInTheDocument();
  });

  it("copies the literal code block without markup", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<MarkdownMessage content={"```ts\nconst value = 42;\n```"} />);

    await user.click(screen.getByRole("button", { name: "复制代码" }));

    expect(writeText).toHaveBeenCalledWith("const value = 42;");
    expect(screen.getByText("已复制")).toBeInTheDocument();
  });

  it("does not execute or render raw HTML", () => {
    const { container } = render(
      <MarkdownMessage content={'<img src=x onerror="alert(1)"><script>alert(1)</script>safe'} />,
    );

    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByTestId("markdown-message")).toHaveTextContent("safe");
  });

  it("renders an unfinished streaming fence without crashing", () => {
    render(<MarkdownMessage content={"正在生成：\n\n```python\nprint('partial')"} />);
    expect(screen.getByText("print")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "复制代码" })).toBeInTheDocument();
  });
});
