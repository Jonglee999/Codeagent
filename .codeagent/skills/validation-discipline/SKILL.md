---
name: validation-discipline
description: Defines completion evidence and bounded repair behavior.
version: "1.0"
triggers: [fix, test, validate, implement, create, modify, 修复, 测试, 验证, 实现, 创建, 修改]
languages: [python, typescript, javascript]
allowed_tools: [read_file, list_files, write_file, apply_patch, delete_file, search_code, get_diagnostics, run_terminal, git, "mcp:*"]
---

# Validation and reflection

- Never report completion solely because a file was written or the model produced prose.
- Treat syntax, targeted tests, and the task's fail-to-pass tests as required evidence.
- On failure, summarize the hypothesis, evidence, and next change before retrying.
- Do not repeat an unchanged command after an unchanged failure.
- Stop after the configured repair budget and return the exact blocker plus current diff.
- Preserve pre-existing failures separately from failures introduced by the patch.
