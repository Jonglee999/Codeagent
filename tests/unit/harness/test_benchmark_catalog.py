from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from codeagent.benchmarks import BenchmarkCatalog
from codeagent.benchmarks.catalog import BenchmarkTask
from codeagent.extensions import discover_skills, load_project_instructions


def test_catalog_resolves_allowlisted_workspaces_inside_runtime_root() -> None:
    catalog = BenchmarkCatalog()
    tasks = catalog.list_tasks()
    assert len(tasks) == 30
    for task in tasks:
        workspace = catalog.workspace_for(task.instance_id)
        workspace.relative_to(catalog.workspace_root.resolve())
        assert workspace.parent.name == task.instance_id
        assert workspace.name == "latest"


def test_catalog_loads_gold_free_verified_manifest() -> None:
    catalog = BenchmarkCatalog(
        manifest_path=Path("evals/swe_verified_50/tasks.json")
    )

    tasks = catalog.list_tasks()

    assert len(tasks) == 50
    assert all(task.gold_patch_changed_lines is None for task in tasks)


def test_partial_cache_is_not_treated_as_offline_commit_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "partial"
    (repository / ".git").mkdir(parents=True)
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="deadbeef path.py\n?missingblob\n",
            stderr="",
        )

    monkeypatch.setattr("codeagent.benchmarks.catalog.subprocess.run", fake_run)

    assert not BenchmarkCatalog._repository_contains_commit(repository, "deadbeef")
    assert seen["env"]["GIT_NO_LAZY_FETCH"] == "1"  # type: ignore[index]


def test_catalog_manifest_can_be_selected_by_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "CODEAGENT_BENCHMARK_MANIFEST",
        "evals/swe_verified_50/tasks.json",
    )

    catalog = BenchmarkCatalog()

    assert len(catalog.list_tasks()) == 50


def test_project_instructions_load_agents_and_bounded_skills(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Run targeted tests.", encoding="utf-8")
    skill_dir = tmp_path / ".codeagent" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Inspect the final diff.", encoding="utf-8")

    skills = discover_skills(tmp_path)
    instructions = load_project_instructions(tmp_path)

    assert [skill.skill_id for skill in skills] == ["review"]
    assert "Run targeted tests." in instructions
    assert "Inspect the final diff." in instructions


def test_unknown_benchmark_id_is_rejected() -> None:
    catalog = BenchmarkCatalog()
    try:
        catalog.workspace_for("../../outside")
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown benchmark IDs must not produce paths")


def test_workspace_seed_accepts_marked_exact_checkout_with_untracked_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_id = "owner__repo-1"
    workspace_root = tmp_path / "workspaces"
    candidate = workspace_root / instance_id / "prior-run"
    candidate.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=candidate, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=candidate,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=candidate,
        check=True,
        capture_output=True,
    )
    tracked = candidate / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=candidate, check=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=candidate,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=candidate,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    task = BenchmarkTask(
        instance_id=instance_id,
        repo="owner/repo",
        base_commit=commit,
        version="1",
        problem_statement="fix it",
        fail_to_pass=[],
        pass_to_pass=[],
        gold_patch_changed_lines=1,
    )
    marker = {
        "schema_version": 1,
        "instance_id": instance_id,
        "repo": task.repo,
        "base_commit": commit,
    }
    (candidate / ".git" / "codeagent-benchmark.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    artifact = candidate / ".codeagent" / "report.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")

    catalog = BenchmarkCatalog(repository_root=tmp_path)
    catalog.workspace_root = workspace_root
    monkeypatch.setattr(catalog, "get_task", lambda _: task)

    assert catalog._clean_workspace_seed(instance_id) is None
    assert catalog._workspace_seed(instance_id) == candidate


@pytest.mark.asyncio
async def test_prepare_retries_in_staging_and_publishes_only_complete_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"
    cache_attempts = 0

    async def fake_run(cwd: Path, *args: str) -> str:
        nonlocal cache_attempts
        if args[:2] == ("git", "clone"):
            staging = Path(args[-1])
            is_cache_clone = cwd == catalog.repository_cache_root
            if is_cache_clone:
                cache_attempts += 1
                if cache_attempts == 1:
                    raise RuntimeError("transient network failure")
            staging.mkdir(parents=True)
            (staging / ".git").mkdir()
        return ""

    async def no_delay(_: float) -> None:
        return None

    monkeypatch.setattr(catalog, "_run", fake_run)
    monkeypatch.setattr("codeagent.benchmarks.catalog.asyncio.sleep", no_delay)

    destination = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert cache_attempts == 2
    assert (destination / ".git").is_dir()
    assert (destination / ".git" / "codeagent-benchmark.json").is_file()
    assert not list(tmp_path.rglob(".p-*-*"))


@pytest.mark.asyncio
async def test_each_prepare_gets_a_fresh_run_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"

    async def fake_run(cwd: Path, *args: str) -> str:
        if args[:2] == ("git", "clone"):
            staging = Path(args[-1])
            staging.mkdir(parents=True)
            (staging / ".git").mkdir()
        return ""

    monkeypatch.setattr(catalog, "_run", fake_run)
    first = await catalog.prepare_workspace("pytest-dev__pytest-5227")
    (first / "dirty.py").write_text("old patch\n", encoding="utf-8")
    second = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert first != second
    assert not (second / "dirty.py").exists()


@pytest.mark.asyncio
async def test_timed_out_command_is_terminated_before_return(tmp_path: Path) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())

    with pytest.raises(RuntimeError, match="Command timed out"):
        await catalog._run(
            tmp_path,
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            timeout_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_prepare_repairs_missing_files_in_fresh_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"
    cache = catalog.repository_cache_for("pytest-dev__pytest-5227")
    (cache / ".git").mkdir(parents=True)
    clone_attempts = 0
    status_checks = 0
    checkout_index_calls = 0

    async def fake_run(cwd: Path, *args: str) -> str:
        nonlocal clone_attempts, status_checks, checkout_index_calls
        if args[:2] == ("git", "clone"):
            clone_attempts += 1
            staging = Path(args[-1])
            staging.mkdir(parents=True)
            (staging / ".git").mkdir()
        if args[:3] == ("git", "status", "--porcelain"):
            status_checks += 1
            return " D missing.py\n" if status_checks == 1 else ""
        if args[:3] == ("git", "checkout-index", "--all"):
            checkout_index_calls += 1
        return ""

    async def no_delay(_: float) -> None:
        return None

    monkeypatch.setattr(catalog, "_run", fake_run)
    monkeypatch.setattr("codeagent.benchmarks.catalog.asyncio.sleep", no_delay)

    destination = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert destination.is_dir()
    assert clone_attempts == 1
    assert status_checks == 2
    assert checkout_index_calls == 1
    assert not list(tmp_path.rglob(".p-*-*"))


@pytest.mark.asyncio
async def test_prepare_uses_clean_local_seed_without_github(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"
    cache = catalog.repository_cache_for("pytest-dev__pytest-5227")
    (cache / ".git").mkdir(parents=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    calls: list[tuple[str, ...]] = []

    async def fake_run(cwd: Path, *args: str) -> str:
        calls.append(args)
        if args[:2] == ("git", "init"):
            (cwd / ".git").mkdir()
        return ""

    monkeypatch.setattr(catalog, "_run", fake_run)
    monkeypatch.setattr(catalog, "_clean_workspace_seed", lambda _: seed)

    destination = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert destination.is_dir()
    assert any(call[:3] == ("git", "fetch", "--depth=1") for call in calls)
    assert not any(call[:2] == ("git", "clone") for call in calls)


@pytest.mark.asyncio
async def test_prepare_uses_cached_exact_commit_without_github(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"
    cache = catalog.repository_cache_for("pytest-dev__pytest-5227")
    (cache / ".git").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    async def fake_run(cwd: Path, *args: str) -> str:
        calls.append(args)
        if args[:2] == ("git", "init"):
            (cwd / ".git").mkdir()
        return ""

    monkeypatch.setattr(catalog, "_run", fake_run)
    monkeypatch.setattr(catalog, "_clean_workspace_seed", lambda _: None)
    monkeypatch.setattr(catalog, "_repository_contains_commit", lambda *_: True)

    destination = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert destination.is_dir()
    assert any(
        call[:3] == ("git", "fetch", "--depth=1") and str(cache) in call
        for call in calls
    )
    assert not any(call[:2] == ("git", "clone") for call in calls)


@pytest.mark.asyncio
async def test_remote_fallback_is_self_contained_after_partial_seed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = BenchmarkCatalog(repository_root=Path.cwd())
    catalog.workspace_root = tmp_path / "workspaces"
    catalog.repository_cache_root = tmp_path / "repositories"
    cache = catalog.repository_cache_for("pytest-dev__pytest-5227")
    (cache / ".git").mkdir(parents=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    calls: list[tuple[str, ...]] = []
    checkout_calls = 0

    async def fake_run(cwd: Path, *args: str) -> str:
        nonlocal checkout_calls
        calls.append(args)
        if args[:2] == ("git", "init"):
            (cwd / ".git").mkdir()
        if args[:2] == ("git", "checkout"):
            checkout_calls += 1
            if checkout_calls == 1:
                raise RuntimeError("Server does not allow unadvertised object")
        if args[:2] == ("git", "clone"):
            staging = Path(args[-1])
            staging.mkdir(parents=True)
            (staging / ".git").mkdir()
        return ""

    async def no_delay(_: float) -> None:
        return None

    monkeypatch.setattr(catalog, "_run", fake_run)
    monkeypatch.setattr(catalog, "_clean_workspace_seed", lambda _: seed)
    monkeypatch.setattr("codeagent.benchmarks.catalog.asyncio.sleep", no_delay)

    destination = await catalog.prepare_workspace("pytest-dev__pytest-5227")

    assert destination.is_dir()
    remote_clone = next(call for call in calls if call[:2] == ("git", "clone"))
    assert "--filter=blob:none" not in remote_clone
    assert "--reference-if-able" not in remote_clone
