"""Task-local file observations with precise invalidation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeagent.gateway.tool_gateway import ToolResult


@dataclass
class _ReadObservation:
    result: ToolResult
    file_path: str
    fingerprint: tuple[int, int]


class TaskWorkingSet:
    """Reuse stable file reads inside one ToolGateway/task lifetime."""

    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root).resolve()
        self._reads: dict[tuple[Any, ...], _ReadObservation] = {}
        self.read_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.invalidation_count = 0
        self.broad_invalidation_count = 0
        self.recent_invalidations: list[dict[str, Any]] = []

    def _resolved_file(self, raw_path: Any) -> tuple[Path, str] | None:
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            return None
        try:
            candidate = Path(raw_path)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (self._root / candidate).resolve()
            )
            relative = resolved.relative_to(self._root).as_posix()
            return resolved, relative
        except (OSError, RuntimeError, ValueError):
            return None

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        return stat.st_mtime_ns, stat.st_size

    def _read_key(self, params: dict[str, Any]) -> tuple[Any, ...] | None:
        resolved = self._resolved_file(params.get("file_path"))
        if resolved is None:
            return None
        _path, relative = resolved
        return (
            relative,
            params.get("start_line"),
            params.get("end_line"),
            params.get("encoding", "utf-8"),
        )

    def get_read(self, params: dict[str, Any]) -> ToolResult | None:
        self.read_requests += 1
        key = self._read_key(params)
        if key is None:
            self.cache_misses += 1
            return None
        observation = self._reads.get(key)
        if observation is None:
            self.cache_misses += 1
            return None
        resolved = self._resolved_file(observation.file_path)
        fingerprint = self._fingerprint(resolved[0]) if resolved else None
        if fingerprint != observation.fingerprint:
            self.invalidate_file(observation.file_path, reason="external_change")
            self.cache_misses += 1
            return None

        self.cache_hits += 1
        result = deepcopy(observation.result)
        if isinstance(result.data, dict):
            result.data["working_set"] = {"cache_hit": True}
        return result

    def remember_read(self, params: dict[str, Any], result: ToolResult) -> None:
        if not result.success:
            return
        key = self._read_key(params)
        if key is None:
            return
        resolved = self._resolved_file(params.get("file_path"))
        fingerprint = self._fingerprint(resolved[0]) if resolved else None
        if resolved is None or fingerprint is None:
            return
        stored = deepcopy(result)
        if isinstance(stored.data, dict):
            stored.data.pop("working_set", None)
        self._reads[key] = _ReadObservation(
            result=stored,
            file_path=resolved[1],
            fingerprint=fingerprint,
        )

    def invalidate_file(self, raw_path: Any, *, reason: str) -> int:
        resolved = self._resolved_file(raw_path)
        if resolved is None:
            return 0
        relative = resolved[1]
        stale = [key for key, item in self._reads.items() if item.file_path == relative]
        for key in stale:
            self._reads.pop(key, None)
        self.invalidation_count += len(stale)
        self._record_invalidation(relative, reason, len(stale))
        return len(stale)

    def invalidate_all(self, *, reason: str) -> int:
        count = len(self._reads)
        self._reads.clear()
        self.invalidation_count += count
        self.broad_invalidation_count += 1
        self._record_invalidation("*", reason, count)
        return count

    def _record_invalidation(self, path: str, reason: str, entries: int) -> None:
        self.recent_invalidations.append(
            {"file_path": path, "reason": reason, "entries": entries}
        )
        if len(self.recent_invalidations) > 20:
            del self.recent_invalidations[:-20]

    def snapshot(self) -> dict[str, Any]:
        hit_rate = self.cache_hits / self.read_requests if self.read_requests else 0.0
        return {
            "read_requests": self.read_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 4),
            "cache_entries": len(self._reads),
            "tracked_files": len({item.file_path for item in self._reads.values()}),
            "invalidation_count": self.invalidation_count,
            "broad_invalidation_count": self.broad_invalidation_count,
            "recent_invalidations": list(self.recent_invalidations),
        }
