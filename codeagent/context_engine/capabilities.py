"""Secret-free context capability discovery for API and Harness doctor."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

from codeagent import config
from codeagent.context_engine.budget import select_context_budget


_TREE_SITTER_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
}


def context_capability_report(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root).resolve()
    tree_sitter_languages = [
        language
        for language, module in _TREE_SITTER_MODULES.items()
        if importlib.util.find_spec("tree_sitter") is not None
        and importlib.util.find_spec(module) is not None
    ]
    lance_available = importlib.util.find_spec("lancedb") is not None
    semantic_mode = config.get_context_semantic_mode()
    index_parent = root / ".codeagent"
    parent = index_parent if index_parent.exists() else root
    budget = select_context_budget(
        "",
        target_tokens=config.get_context_budget(),
        min_tokens=config.get_context_min_input_tokens(),
        max_tokens=config.get_context_max_input_tokens(),
        context_window=config.get_model_context_window(),
        max_output_tokens=config.get_max_completion_tokens_per_call(),
        safety_margin_tokens=config.get_context_safety_margin_tokens(),
        tool_overhead_tokens=config.get_context_tool_overhead_tokens(),
    )
    return {
        "context_mode": config.get_context_mode(),
        "ast_mode": config.get_context_ast_mode(),
        "semantic_mode": semantic_mode,
        "index_background": config.get_context_index_background(),
        "rg_available": shutil.which("rg") is not None,
        "tree_sitter_languages": tree_sitter_languages,
        "lancedb_available": lance_available,
        "lancedb_writable": parent.is_dir() and _directory_writable(parent),
        "semantic_index_status": "disabled" if semantic_mode == "off" else "deferred",
        "embedding_model_status": "deferred",
        "embedding_model": config.get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "model_context_window": config.get_model_context_window(),
        "effective_context_budget": config.get_context_budget(),
        "adaptive_context": {
            "profiles": [16000, 32000, 48000, 80000],
            "minimum": config.get_context_min_input_tokens(),
            "maximum": config.get_context_max_input_tokens(),
            "model_available_tokens": budget.model_available_tokens,
        },
        "a2a_status": "disabled",
    }


def _directory_writable(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and os.access(path, os.W_OK)
    except OSError:
        return False


def main() -> None:
    print(json.dumps(context_capability_report(), ensure_ascii=False))


if __name__ == "__main__":
    main()
