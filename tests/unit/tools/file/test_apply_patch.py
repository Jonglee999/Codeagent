from __future__ import annotations

import pytest

from codeagent.tools.file.apply_patch import ApplyPatchTool


@pytest.mark.asyncio
async def test_applies_exact_replacement_and_returns_diff(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        file_path="app.py", old_text="value = 1", new_text="value = 2"
    )

    assert result.success
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert result.data["replacements"] == 1
    assert "-value = 1" in result.data["diff"]
    assert result.data["backup_path"]


@pytest.mark.asyncio
async def test_conflict_leaves_file_unchanged(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text("current\n", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        file_path="app.py", old_text="stale", new_text="updated"
    )

    assert not result.success
    assert result.error_code == "PATCH_CONFLICT"
    assert result.retryable is True
    assert "Re-read" in result.suggested_recovery
    assert result.data["actual_matches"] == 0
    assert target.read_text(encoding="utf-8") == "current\n"


@pytest.mark.asyncio
async def test_can_require_multiple_matches(tmp_path) -> None:
    target = tmp_path / "values.txt"
    target.write_text("x x x", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        file_path="values.txt",
        old_text="x",
        new_text="y",
        expected_replacements=3,
    )

    assert result.success
    assert target.read_text(encoding="utf-8") == "y y y"


@pytest.mark.asyncio
async def test_applies_codex_wrapped_unified_patch(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text("before\nvalue = 1\nafter\n", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        file_path="app.py",
        patch="""*** Begin Patch
*** Update File: app.py
@@
 before
-value = 1
+value = 2
 after
*** End Patch
""",
    )

    assert result.success
    assert target.read_text(encoding="utf-8") == "before\nvalue = 2\nafter\n"
    assert result.data["replacements"] == 1


@pytest.mark.asyncio
async def test_infers_safe_target_from_codex_patch(tmp_path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        patch="""*** Begin Patch
*** Update File: src/app.py
@@
-value = 1
+value = 2
*** End Patch
""",
    )

    assert result.success
    assert target.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_missing_path_rejects_multiple_or_traversal_targets(tmp_path) -> None:
    tool = ApplyPatchTool(tmp_path)

    multiple = await tool.execute(
        patch="""*** Begin Patch
*** Update File: one.py
@@
-one
+ONE
*** Update File: two.py
@@
-two
+TWO
*** End Patch
""",
    )
    traversal = await tool.execute(
        patch="""*** Begin Patch
*** Update File: ../outside.py
@@
-one
+ONE
*** End Patch
""",
    )

    assert not multiple.success
    assert "exactly one" in multiple.error_message
    assert not traversal.success
    assert "unsafe" in traversal.error_message


@pytest.mark.asyncio
async def test_unified_patch_rejects_a_different_target_file(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = await ApplyPatchTool(tmp_path).execute(
        file_path="app.py",
        patch="""*** Begin Patch
*** Update File: other.py
@@
-value = 1
+value = 2
*** End Patch
""",
    )

    assert not result.success
    assert result.error_code == "PATCH_FAILED"
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_conflicting_unified_patch_returns_closest_exact_context(tmp_path) -> None:
    target = tmp_path / "fields.py"
    target.write_text(
        "    default_error_messages = {\n"
        "        'invalid': _(\"value must be in \"\n"
        "                     \"[DD] [HH:[MM:]]ss format.\")\n"
        "    }\n",
        encoding="utf-8",
    )

    result = await ApplyPatchTool(tmp_path).execute(
        patch="""*** Begin Patch
*** Update File: fields.py
@@
-        'invalid': _(\"value must be in [DD] [HH:[MM:]]ss format.\")
+        'invalid': _(\"value must be in [DD] [[HH:]MM:]ss format.\")
*** End Patch
""",
    )

    assert not result.success
    assert result.error_code == "PATCH_CONFLICT"
    assert "closest_context" in result.data
    assert "[DD] [HH:[MM:]]ss" in result.data["closest_context"]
    assert target.read_text(encoding="utf-8").endswith("    }\n")
