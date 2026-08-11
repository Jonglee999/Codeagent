---
name: swe-debugging
description: Evidence-first workflow for issue-driven repository fixes.
version: "1.0"
triggers: [fix, bug, error, failing, debug, 修复, 错误, 失败, 问题]
languages: [python, typescript, javascript]
allowed_tools: [read_file, list_files, write_file, apply_patch, delete_file, search_code, get_diagnostics, run_terminal, git, "mcp:*"]
---

# SWE debugging discipline

1. Read the issue, repository instructions, and the smallest relevant source/test files.
2. Reproduce the failing behavior or run the named target test before editing when feasible.
3. Search for the defining symbol and nearby tests; do not guess file locations.
4. Make the smallest coherent production change and add a regression test when appropriate.
5. Run the most targeted test first, inspect the complete failure, then broaden validation.
6. Inspect `git diff` before finishing. Remove debug output and unrelated formatting churn.
7. A command that ran successfully but returned a non-zero exit code is failed validation.
