"""Project instruction and extension discovery."""

from .instructions import (
    InstructionResolution,
    ProjectSkill,
    discover_skills,
    load_project_instructions,
    resolve_project_instructions,
)

__all__ = [
    "InstructionResolution",
    "ProjectSkill",
    "discover_skills",
    "load_project_instructions",
    "resolve_project_instructions",
]
