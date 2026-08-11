"""Curated benchmark catalog with allow-listed workspace preparation."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    gold_patch_changed_lines: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkTask":
        return cls(
            instance_id=str(value["instance_id"]),
            repo=str(value["repo"]),
            base_commit=str(value["base_commit"]),
            version=str(value.get("version", "")),
            problem_statement=str(value["problem_statement"]),
            fail_to_pass=[str(item) for item in value.get("fail_to_pass", [])],
            pass_to_pass=[str(item) for item in value.get("pass_to_pass", [])],
            gold_patch_changed_lines=(
                int(value["gold_patch_changed_lines"])
                if value.get("gold_patch_changed_lines") is not None
                else None
            ),
        )


class BenchmarkCatalog:
    def __init__(
        self,
        repository_root: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.repository_root = repository_root or Path(__file__).resolve().parents[2]
        configured_manifest = os.environ.get("CODEAGENT_BENCHMARK_MANIFEST")
        selected_manifest = manifest_path or (
            Path(configured_manifest) if configured_manifest else None
        )
        if selected_manifest is None:
            selected_manifest = Path("evals") / "swe_smoke" / "tasks.json"
        if not selected_manifest.is_absolute():
            selected_manifest = self.repository_root / selected_manifest
        self.manifest_path = selected_manifest.resolve()
        self.workspace_root = (
            self.repository_root / ".codeagent" / "workspaces" / "benchmarks"
        )
        self.repository_cache_root = (
            self.repository_root / ".codeagent" / "benchmarks" / "repositories"
        )

    def list_tasks(self) -> list[BenchmarkTask]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [BenchmarkTask.from_dict(item) for item in payload["tasks"]]

    def get_task(self, instance_id: str) -> BenchmarkTask:
        for task in self.list_tasks():
            if task.instance_id == instance_id:
                return task
        raise KeyError(instance_id)

    def workspace_for(self, instance_id: str, run_id: str | None = None) -> Path:
        task = self.get_task(instance_id)
        destination = (
            self.workspace_root / task.instance_id / (run_id or "latest")
        ).resolve()
        destination.relative_to(self.workspace_root.resolve())
        return destination

    def repository_cache_for(self, instance_id: str) -> Path:
        task = self.get_task(instance_id)
        destination = (
            self.repository_cache_root / task.repo.replace("/", "__")
        ).resolve()
        destination.relative_to(self.repository_cache_root.resolve())
        return destination

    @staticmethod
    def _clone_urls(repo: str) -> tuple[str, ...]:
        github = f"https://github.com/{repo}.git"
        return (
            github,
            f"https://gitclone.com/github.com/{repo}.git",
            f"https://ghfast.top/{github}",
        )

    async def _prepare_repository_cache(self, task: BenchmarkTask) -> Path:
        destination = self.repository_cache_for(task.instance_id)
        self.repository_cache_root.mkdir(parents=True, exist_ok=True)
        if (destination / ".git").is_dir():
            return destination

        clone_urls = self._clone_urls(task.repo)
        last_error: Exception | None = None
        for attempt, clone_url in enumerate(clone_urls):
            staging = self.repository_cache_root / (
                f".p-cache-{uuid.uuid4().hex[:12]}"
            )
            try:
                await self._run(
                    self.repository_cache_root,
                    "git", "clone", "--filter=blob:none", "--no-checkout",
                    clone_url, str(staging),
                )
                staging.replace(destination)
                return destination
            except asyncio.CancelledError:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            except Exception as exc:
                last_error = exc
                shutil.rmtree(staging, ignore_errors=True)
                if attempt + 1 < len(clone_urls):
                    await asyncio.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    async def prepare_workspace(
        self,
        instance_id: str,
        *,
        run_id: str | None = None,
    ) -> Path:
        task = self.get_task(instance_id)
        cache = await self._prepare_repository_cache(task)
        token = run_id or uuid.uuid4().hex
        destination = self.workspace_for(instance_id, token)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Benchmark run workspace already exists: {destination}")
        local_seed = self._workspace_seed(task.instance_id)
        repository_seed = (
            local_seed
            if local_seed is not None
            else cache if self._repository_contains_commit(cache, task.base_commit)
            else None
        )
        clone_urls = self._clone_urls(task.repo)
        last_error: Exception | None = None
        for attempt in range(3):
            # Keep the staging component short: deep upstream paths can otherwise
            # exceed Git-for-Windows' effective path limit before publication.
            staging = destination.parent / f".p-run-{uuid.uuid4().hex[:12]}"
            try:
                # Try the zero-network seed once. A partial clone may know the exact
                # commit while still missing lazily fetched blobs; subsequent attempts
                # must fall back to the authoritative remote instead of repeating the
                # same incomplete local fetch.
                attempt_seed = repository_seed if attempt == 0 else None
                if attempt_seed is not None:
                    staging.mkdir(parents=True)
                    await self._git(staging, "init")
                    await self._git(
                        staging,
                        "fetch",
                        "--depth=1",
                        str(attempt_seed),
                        task.base_commit,
                    )
                else:
                    remote_attempt = (
                        max(0, attempt - 1) if repository_seed is not None else attempt
                    )
                    clone_url = clone_urls[min(remote_attempt, len(clone_urls) - 1)]
                    # A cached partial clone cannot safely be used as an alternate
                    # for an exact historical checkout: Git may later try to lazily
                    # fetch missing blobs by object id, which several mirrors reject
                    # as an unadvertised object. Remote fallbacks intentionally make
                    # a self-contained clone so checkout never depends on the cache's
                    # promisor remote.
                    await self._run(
                        destination.parent,
                        "git", "clone", "--no-checkout", clone_url, str(staging),
                    )
                await self._git(staging, "checkout", "--detach", task.base_commit)
                checkout_status = await self._git(
                    staging, "status", "--porcelain", "--untracked-files=all"
                )
                dirty_entries = [
                    line for line in checkout_status.splitlines() if line.strip()
                ]
                if dirty_entries and all(
                    line.startswith(" D ") for line in dirty_entries
                ):
                    # Some Windows filesystems occasionally omit tracked files during
                    # the first checkout. Re-materialize the brand-new staging index;
                    # no user-authored files can exist here yet.
                    await self._git(staging, "checkout-index", "--all", "--force")
                    checkout_status = await self._git(
                        staging, "status", "--porcelain", "--untracked-files=all"
                    )
                if checkout_status.strip():
                    raise RuntimeError(
                        "Fresh benchmark checkout is not clean; retrying preparation"
                    )
                marker = {
                    "schema_version": 1,
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                (staging / ".git" / "codeagent-benchmark.json").write_text(
                    json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                staging.replace(destination)
                return destination
            except asyncio.CancelledError:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            except Exception as exc:
                last_error = exc
                shutil.rmtree(staging, ignore_errors=True)
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _repository_contains_commit(repository: Path, commit: str) -> bool:
        """Return whether an exact commit is fully available without lazy fetch."""
        if not (repository / ".git").is_dir():
            return False
        result = subprocess.run(
            ["git", "rev-list", "--objects", "--missing=print", commit],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        return result.returncode == 0 and not any(
            line.startswith("?") for line in result.stdout.splitlines()
        )

    def _clean_workspace_seed(self, instance_id: str) -> Path | None:
        """Find a clean prior checkout usable as an offline, exact-commit source."""
        task_root = self.workspace_root / instance_id
        if not task_root.is_dir():
            return None
        candidates = sorted(
            (item for item in task_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            valid, _reason = self.validate_workspace(
                instance_id, candidate, require_clean=True
            )
            if valid:
                return candidate
        return None

    def _workspace_seed(self, instance_id: str) -> Path | None:
        """Find a prior exact checkout usable only as a Git object source.

        Prefer a clean workspace. If none exists, a correctly marked checkout at
        the exact base commit is still safe as a fetch source: the new staging
        workspace checks out the immutable commit and never copies its working tree.
        This also tolerates task-local untracked `.codeagent` artifacts.
        """

        clean = self._clean_workspace_seed(instance_id)
        if clean is not None:
            return clean
        task_root = self.workspace_root / instance_id
        if not task_root.is_dir():
            return None
        task = self.get_task(instance_id)
        candidates = sorted(
            (item for item in task_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            valid, _reason = self.validate_workspace(
                instance_id, candidate, require_clean=False
            )
            if not valid:
                continue
            tree = subprocess.run(
                ["git", "cat-file", "-e", f"{task.base_commit}^{{tree}}"],
                cwd=candidate,
                capture_output=True,
                text=True,
                check=False,
            )
            if tree.returncode == 0:
                return candidate
        return None

    def validate_workspace(
        self,
        instance_id: str,
        workspace: str | Path,
        *,
        require_clean: bool,
    ) -> tuple[bool, str]:
        """Validate ownership, identity, base commit, and optional cleanliness."""
        task = self.get_task(instance_id)
        candidate = Path(workspace).resolve()
        try:
            candidate.relative_to(self.workspace_root.resolve())
        except ValueError:
            return False, "workspace is outside the benchmark runtime root"
        marker_path = candidate / ".git" / "codeagent-benchmark.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, "benchmark workspace marker is missing or invalid"
        expected = {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "base_commit": task.base_commit,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            return False, "benchmark workspace marker does not match the catalog task"
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=candidate, capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if head.returncode != 0 or head.stdout.strip() != task.base_commit:
            return False, "benchmark workspace HEAD does not match the base commit"
        if require_clean:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=candidate, capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False,
            )
            if status.returncode != 0 or status.stdout.strip():
                return False, "benchmark workspace is not clean"
        return True, "ready"

    async def _git(self, cwd: Path, *args: str) -> str:
        return await self._run(cwd, "git", *args)

    async def _run(
        self,
        cwd: Path,
        *args: str,
        timeout_seconds: float = 600,
    ) -> str:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except TimeoutError as exc:
            await self._terminate_process_tree(process)
            raise RuntimeError(
                f"Command timed out after {timeout_seconds:g}s: {args[0]}"
            ) from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Command failed ({process.returncode}): {detail[-4000:]}")
        return stdout.decode("utf-8", errors="replace")

    @staticmethod
    async def _terminate_process_tree(
        process: asyncio.subprocess.Process,
    ) -> None:
        """Stop a timed-out command and its descendants before staging cleanup."""
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
                await process.wait()
