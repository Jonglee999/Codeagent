"""Tree-sitter based import extraction with explicit evidence metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DependencyRecord:
    module: str
    kind: str
    source: str = "ast"


_PY_FROM = re.compile(r"^from\s+([^\s]+)\s+import\s+(.+)$", re.S)
_PY_IMPORT = re.compile(r"^import\s+(.+)$", re.S)
_JS_STRING = re.compile(r"['\"]([^'\"]+)['\"]")


class TreeSitterDependencyAdapter:
    """Extract only imports represented by syntax nodes, never comments/strings."""

    _PY_NODES = {"import_statement", "import_from_statement"}
    _JS_NODES = {"import_statement", "export_statement", "call_expression"}

    def extract(self, code: str, language: str) -> list[DependencyRecord]:
        from codeagent.context_engine.symbol_table import _init_parser

        parser = _init_parser(language)
        raw = code.encode("utf-8")
        tree = parser.parse(raw)
        records: list[DependencyRecord] = []
        self._walk(tree.root_node, raw, language, records)
        return self._deduplicate(records)

    def _walk(
        self,
        node: Any,
        raw: bytes,
        language: str,
        records: list[DependencyRecord],
    ) -> None:
        if language == "python" and node.type in self._PY_NODES:
            records.extend(self._python_node(node, raw))
            return
        if language != "python" and node.type in self._JS_NODES:
            record = self._javascript_node(node, raw)
            if record is not None:
                records.append(record)
            # A call expression is self-contained. Import/export nodes may have
            # nested strings but no nested import statements worth traversing.
            return
        for child in node.children:
            self._walk(child, raw, language, records)

    @staticmethod
    def _text(node: Any, raw: bytes) -> str:
        return raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _python_node(self, node: Any, raw: bytes) -> list[DependencyRecord]:
        text = self._text(node, raw).strip()
        from_match = _PY_FROM.match(text)
        if from_match:
            module = from_match.group(1).strip()
            imported = from_match.group(2)
            if module == ".":
                return [
                    DependencyRecord(f".{name}", "import")
                    for name in self._python_names(imported)
                ]
            return [DependencyRecord(module, "import")]
        import_match = _PY_IMPORT.match(text)
        if not import_match:
            return []
        return [
            DependencyRecord(name.split(".")[0], "import")
            for name in self._python_names(import_match.group(1))
        ]

    @staticmethod
    def _python_names(value: str) -> list[str]:
        value = value.replace("(", "").replace(")", "").replace("\\\n", " ")
        names = []
        for part in value.split(","):
            name = part.strip().split(" as ", 1)[0].strip()
            if name and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
                names.append(name)
        return names

    def _javascript_node(self, node: Any, raw: bytes) -> DependencyRecord | None:
        text = self._text(node, raw).strip()
        if node.type == "call_expression" and not re.match(r"^(?:require|import)\s*\(", text):
            return None
        match = _JS_STRING.search(text)
        if match is None:
            return None
        kind = "export" if node.type == "export_statement" else "dynamic_import" if node.type == "call_expression" else "import"
        return DependencyRecord(match.group(1), kind)

    @staticmethod
    def _deduplicate(records: list[DependencyRecord]) -> list[DependencyRecord]:
        seen: set[tuple[str, str]] = set()
        result = []
        for record in records:
            key = (record.module, record.kind)
            if key not in seen:
                seen.add(key)
                result.append(record)
        return result
