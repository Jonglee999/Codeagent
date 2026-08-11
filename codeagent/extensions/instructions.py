"""Resolve bounded, scoped Skill and repository instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

import frontmatter
from yaml import YAMLError

MAX_INSTRUCTION_CHARS = 20_000
MAX_SKILL_CHARS = 8_000


def _as_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


@dataclass(frozen=True)
class ProjectSkill:
    skill_id: str
    path: Path
    content: str
    description: str = ""
    source: str = "workspace"
    version: str = "1"
    triggers: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    always: bool = False

    def matches(self, query: str) -> bool:
        if self.always:
            return True
        return (
            any(_term_matches(query, trigger, allow_inflections=True) for trigger in self.triggers)
            or any(_term_matches(query, language) for language in self.languages)
            or any(_term_matches(query, framework) for framework in self.frameworks)
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "source": self.source,
            "version": self.version,
            "triggers": list(self.triggers),
            "languages": list(self.languages),
            "frameworks": list(self.frameworks),
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass
class InstructionResolution:
    instructions: str = ""
    skills: list[ProjectSkill] = field(default_factory=list)
    repository_instructions: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def allowed_tools(self) -> list[str]:
        return sorted({tool for skill in self.skills for tool in skill.allowed_tools})


def _term_matches(query: str, term: str, *, allow_inflections: bool = False) -> bool:
    """Match metadata terms without triggering inside unrelated identifiers."""
    normalized = term.strip()
    if not normalized:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in normalized):
        return normalized.lower() in query.lower()
    suffix = (
        r"(?:s|es|ed|ing)?"
        if allow_inflections and len(normalized) >= 3 and normalized.isalpha()
        else ""
    )
    pattern = rf"(?<![\w]){re.escape(normalized)}{suffix}(?![\w])"
    return bool(re.search(pattern, query, re.IGNORECASE))


def _read_skill(
    path: Path,
    source: str,
    warnings: list[str] | None = None,
) -> ProjectSkill | None:
    try:
        raw = path.read_text(encoding="utf-8")[:MAX_SKILL_CHARS]
        post = frontmatter.loads(raw)
    except (OSError, UnicodeError, TypeError, ValueError, YAMLError) as exc:
        if warnings is not None:
            warnings.append(f"Could not load skill {path}: {type(exc).__name__}: {exc}")
        return None
    metadata = post.metadata
    return ProjectSkill(
        skill_id=str(metadata.get("name") or path.parent.name),
        path=path,
        content=post.content.strip(),
        description=str(metadata.get("description", "")),
        source=source,
        version=str(metadata.get("version", "1")),
        triggers=_as_tuple(metadata.get("triggers")),
        languages=_as_tuple(metadata.get("languages")),
        frameworks=_as_tuple(metadata.get("frameworks")),
        allowed_tools=_as_tuple(metadata.get("allowed_tools")),
        always=bool(metadata.get("always", False)),
    )


def discover_skills(
    project_root: str | Path,
    *,
    source: str = "workspace",
    warnings: list[str] | None = None,
) -> list[ProjectSkill]:
    root = Path(project_root).resolve()
    skill_root = root if root.name == "skills" else root / ".codeagent" / "skills"
    if not skill_root.is_dir():
        return []
    skills: list[ProjectSkill] = []
    for path in sorted(skill_root.glob("*/SKILL.md")):
        skill = _read_skill(path, source, warnings)
        if skill is not None:
            skills.append(skill)
    return skills


def resolve_project_instructions(
    project_root: str | Path,
    query: str,
    *,
    product_root: str | Path | None = None,
    user_skill_root: str | Path | None = None,
    skills_enabled: bool = True,
) -> InstructionResolution:
    """Resolve Product -> User -> AGENTS.md -> Workspace with later overrides."""
    workspace = Path(project_root).resolve()
    product = Path(product_root or Path.cwd()).resolve()
    user_root = Path(user_skill_root).expanduser().resolve() if user_skill_root else Path.home() / ".codeagent" / "skills"

    ordered: list[ProjectSkill] = []
    warnings: list[str] = []
    if skills_enabled:
        ordered.extend(discover_skills(product, source="product", warnings=warnings))
        ordered.extend(discover_skills(user_root, source="user", warnings=warnings))
        ordered.extend(discover_skills(workspace, source="workspace", warnings=warnings))

    # Higher-precedence sources replace an earlier skill with the same stable id.
    by_id: dict[str, ProjectSkill] = {}
    for skill in ordered:
        by_id[skill.skill_id] = skill
    activated = [skill for skill in by_id.values() if skill.matches(query)]

    sections: list[str] = []
    repository_instructions = False

    # Product/user skills precede repository instructions; workspace skills follow them.
    for skill in activated:
        if skill.source in {"product", "user"}:
            sections.append(f"## {skill.source.title()} skill: {skill.skill_id}\n{skill.content}")

    agents_file = workspace / "AGENTS.md"
    if agents_file.is_file():
        try:
            sections.append(f"## Repository instructions (AGENTS.md)\n{agents_file.read_text(encoding='utf-8')}")
            repository_instructions = True
        except (OSError, UnicodeError) as exc:
            warnings.append(f"Could not read AGENTS.md: {exc}")

    for skill in activated:
        if skill.source == "workspace":
            sections.append(f"## Workspace skill: {skill.skill_id}\n{skill.content}")

    return InstructionResolution(
        instructions=("\n\n".join(sections))[:MAX_INSTRUCTION_CHARS],
        skills=activated,
        repository_instructions=repository_instructions,
        warnings=warnings,
    )


def load_project_instructions(project_root: str | Path) -> str:
    """Backward-compatible loader that includes all local workspace skills."""
    root = Path(project_root).resolve()
    sections: list[str] = []
    agents_file = root / "AGENTS.md"
    if agents_file.is_file():
        try:
            sections.append(f"## Repository instructions (AGENTS.md)\n{agents_file.read_text(encoding='utf-8')}")
        except (OSError, UnicodeError):
            pass
    for skill in discover_skills(root):
        sections.append(f"## Project skill: {skill.skill_id}\n{skill.content}")
    return ("\n\n".join(sections))[:MAX_INSTRUCTION_CHARS]
