from __future__ import annotations

from pathlib import Path

from codeagent.extensions import resolve_project_instructions


def _skill(root: Path, name: str, body: str) -> None:
    path = root / ".codeagent" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_managed_workspace_inherits_matching_product_skill(tmp_path: Path) -> None:
    product = tmp_path / "product"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _skill(product, "debug", "---\nname: debug\ntriggers: [fix]\nallowed_tools: [read_file]\n---\nRead evidence first.")

    resolution = resolve_project_instructions(workspace, "fix the bug", product_root=product)

    assert [skill.skill_id for skill in resolution.skills] == ["debug"]
    assert resolution.skills[0].source == "product"
    assert resolution.allowed_tools == ["read_file"]
    assert "Read evidence first" in resolution.instructions


def test_unmatched_skill_is_not_injected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _skill(workspace, "python", "---\nname: python\ntriggers: [pytest]\n---\nUse pytest.")

    resolution = resolve_project_instructions(workspace, "update the README", product_root=tmp_path)

    assert resolution.skills == []
    assert "Use pytest" not in resolution.instructions


def test_language_and_framework_metadata_activate_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _skill(
        workspace,
        "fastapi",
        "---\nname: fastapi\nlanguages: [python]\nframeworks: [FastAPI]\n---\nUse API conventions.",
    )

    by_language = resolve_project_instructions(
        workspace, "review this Python module", product_root=tmp_path,
    )
    by_framework = resolve_project_instructions(
        workspace, "fix the FastAPI route", product_root=tmp_path,
    )

    assert [skill.skill_id for skill in by_language.skills] == ["fastapi"]
    assert [skill.skill_id for skill in by_framework.skills] == ["fastapi"]


def test_ascii_trigger_does_not_match_inside_identifier(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _skill(workspace, "go", "---\nname: go\ntriggers: [go]\n---\nUse Go tooling.")

    resolution = resolve_project_instructions(
        workspace, "update django settings", product_root=tmp_path,
    )

    assert resolution.skills == []


def test_malformed_skill_is_reported_as_warning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _skill(workspace, "broken", "---\ntriggers: [unterminated\n---\nBroken")

    resolution = resolve_project_instructions(
        workspace, "anything", product_root=tmp_path,
    )

    assert resolution.skills == []
    assert any("Could not load skill" in warning for warning in resolution.warnings)


def test_workspace_skill_overrides_product_skill(tmp_path: Path) -> None:
    product = tmp_path / "product"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = "---\nname: shared\ntriggers: [fix]\n---\n{}"
    _skill(product, "shared", manifest.format("product rule"))
    _skill(workspace, "shared", manifest.format("workspace rule"))

    resolution = resolve_project_instructions(workspace, "fix it", product_root=product)

    assert len(resolution.skills) == 1
    assert resolution.skills[0].source == "workspace"
    assert "workspace rule" in resolution.instructions
    assert "product rule" not in resolution.instructions


def test_disabled_skills_keep_agents_instructions_and_report_no_match(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Keep changes focused.", encoding="utf-8")
    _skill(workspace, "testing", "---\nname: testing\nalways: true\n---\nUse pytest.")

    resolution = resolve_project_instructions(
        workspace,
        "run tests",
        product_root=tmp_path,
        skills_enabled=False,
    )

    assert resolution.skills == []
    assert resolution.allowed_tools == []
    assert resolution.repository_instructions is True
    assert "Keep changes focused." in resolution.instructions
    assert "Use pytest." not in resolution.instructions
