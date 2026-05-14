"""DependencyGraph 单元测试。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest

from codeagent.context_engine.dependency_graph import DependencyGraph


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> str:
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
async def dep_graph() -> DependencyGraph:
    return DependencyGraph()


@pytest.fixture
def py_project(temp_dir: str) -> str:
    """创建带依赖关系的 Python 项目。"""
    # main.py — 导入多个模块
    with open(os.path.join(temp_dir, "main.py"), "w") as f:
        f.write("""
import os
import sys
from models.user import User
from services.auth import login
from utils.helper import format_response
""")

    # utils/helper.py
    utils_dir = os.path.join(temp_dir, "utils")
    os.makedirs(utils_dir)
    with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
        f.write("from .helper import format_response\n")
    with open(os.path.join(utils_dir, "helper.py"), "w") as f:
        f.write("""
def format_response(data):
    return {"data": data}
""")

    # models/user.py
    models_dir = os.path.join(temp_dir, "models")
    os.makedirs(models_dir)
    with open(os.path.join(models_dir, "__init__.py"), "w") as f:
        f.write("from .user import User\n")
    with open(os.path.join(models_dir, "user.py"), "w") as f:
        f.write("""
from dataclasses import dataclass

@dataclass
class User:
    name: str
""")

    # services/auth.py
    services_dir = os.path.join(temp_dir, "services")
    os.makedirs(services_dir)
    with open(os.path.join(services_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(services_dir, "auth.py"), "w") as f:
        f.write("""
from models.user import User

def login(user: User) -> bool:
    return True
""")

    return temp_dir


# ── Tests: Basic build and query ─────────────────────────────────────────


class TestBuildAndQuery:
    @pytest.mark.asyncio
    async def test_build_creates_nodes(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # Should have nodes for all source files
        assert dep_graph.graph.has_node("main.py")
        assert dep_graph.graph.has_node("utils/__init__.py")
        assert dep_graph.graph.has_node("utils/helper.py")
        assert dep_graph.graph.has_node("models/__init__.py")
        assert dep_graph.graph.has_node("services/__init__.py")
        assert dep_graph.graph.has_node("services/auth.py")
        assert dep_graph.graph.has_node("models/user.py")

    @pytest.mark.asyncio
    async def test_build_creates_edges(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # main.py should depend on models/user.py and services/auth.py
        deps = dep_graph.get_dependencies("main.py")
        assert "models/user.py" in deps
        assert "services/auth.py" in deps
        assert "utils/helper.py" in deps

    @pytest.mark.asyncio
    async def test_external_dependencies(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # os, sys are external nodes
        assert dep_graph.graph.has_node("<external>os")
        assert dep_graph.graph.has_node("<external>sys")

    @pytest.mark.asyncio
    async def test_get_dependents(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # models/user.py should have main.py and services/auth.py as dependents
        deps_of = dep_graph.get_dependents("models/user.py")
        assert "main.py" in deps_of
        assert "services/auth.py" in deps_of

    @pytest.mark.asyncio
    async def test_get_dependencies(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # services/auth.py should depend on models/user.py
        deps = dep_graph.get_dependencies("services/auth.py")
        assert "models/user.py" in deps

    @pytest.mark.asyncio
    async def test_no_dependencies(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # utils/helper.py should have no internal deps
        deps = dep_graph.get_dependencies("utils/helper.py")
        # All its deps should be external
        assert all(d.startswith("<external>") for d in deps) if deps else True

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, dep_graph: DependencyGraph) -> None:
        deps = dep_graph.get_dependencies("nonexistent.py")
        assert deps == []
        deps_of = dep_graph.get_dependents("nonexistent.py")
        assert deps_of == []


# ── Tests: Impact scope ──────────────────────────────────────────────────


class TestImpactScope:
    @pytest.mark.asyncio
    async def test_impact_scope_basic(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # Changing models/user.py should impact services/auth.py and main.py
        scope = dep_graph.get_impact_scope("models/user.py")
        assert "services/auth.py" in scope
        assert "main.py" in scope

    @pytest.mark.asyncio
    async def test_impact_scope_leaf(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # Changing main.py should have no impacts (nothing depends on it)
        scope = dep_graph.get_impact_scope("main.py")
        assert scope == []

    @pytest.mark.asyncio
    async def test_impact_scope_transitive(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        # Changing utils/helper.py should impact utils/__init__.py and main.py
        scope = dep_graph.get_impact_scope("utils/helper.py")
        assert "utils/__init__.py" in scope
        assert "main.py" in scope

    @pytest.mark.asyncio
    async def test_impact_scope_nonexistent(
        self, dep_graph: DependencyGraph
    ) -> None:
        scope = dep_graph.get_impact_scope("nonexistent.py")
        assert scope == []


# ── Tests: Circular dependency detection ─────────────────────────────────


class TestCircularDependencies:
    @pytest.mark.asyncio
    async def test_no_circular_deps(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        circles = dep_graph.find_circular_dependencies()
        assert circles == []

    @pytest.mark.asyncio
    async def test_detect_circular_deps(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        # Create a circular dependency: a.py depends on b.py, b.py depends on a.py
        with open(os.path.join(temp_dir, "a.py"), "w") as f:
            f.write("from b import bar\n")
        with open(os.path.join(temp_dir, "b.py"), "w") as f:
            f.write("from a import foo\n")

        await dep_graph.build(temp_dir)
        circles = dep_graph.find_circular_dependencies()
        assert len(circles) > 0
        assert any("a.py" in c and "b.py" in c for c in circles)

    @pytest.mark.asyncio
    async def test_circular_three_modules(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        # a -> b -> c -> a
        with open(os.path.join(temp_dir, "a.py"), "w") as f:
            f.write("from b import bar\n")
        with open(os.path.join(temp_dir, "b.py"), "w") as f:
            f.write("from c import baz\n")
        with open(os.path.join(temp_dir, "c.py"), "w") as f:
            f.write("from a import foo\n")

        await dep_graph.build(temp_dir)
        circles = dep_graph.find_circular_dependencies()
        assert len(circles) > 0

    @pytest.mark.asyncio
    async def test_self_import_not_circular(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        # from . import X in a package's __init__.py — should not create self-loop
        pkg_dir = os.path.join(temp_dir, "mypkg")
        os.makedirs(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("from . import module\n")
        with open(os.path.join(pkg_dir, "module.py"), "w") as f:
            f.write("x = 1\n")

        await dep_graph.build(temp_dir)
        circles = dep_graph.find_circular_dependencies()
        # No self-loops expected
        assert all(len(c) >= 2 for c in circles) if circles else True
        # Verify no self-loop in graph
        for e in dep_graph.graph.edges():
            assert e[0] != e[1]


# ── Tests: Incremental update ────────────────────────────────────────────


class TestIncrementalUpdate:
    @pytest.mark.asyncio
    async def test_update_file_new_dep(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)

        # Add a new import to main.py
        main_path = os.path.join(py_project, "main.py")
        with open(main_path, "a") as f:
            f.write("from services.admin import check\n")

        await dep_graph.update_file(main_path)
        deps = dep_graph.get_dependencies("main.py")
        assert "services/admin.py" in deps  # new node

    @pytest.mark.asyncio
    async def test_update_file_remove_dep(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)

        # Remove all imports from main.py
        main_path = os.path.join(py_project, "main.py")
        with open(main_path, "w") as f:
            f.write("x = 1\n")

        await dep_graph.update_file(main_path)
        deps = dep_graph.get_dependencies("main.py")
        # Should have no internal deps now
        internal = [d for d in deps if not d.startswith("<external>")]
        assert internal == []

    @pytest.mark.asyncio
    async def test_update_deleted_file(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        main_path = os.path.join(py_project, "main.py")
        os.remove(main_path)

        # update_file should handle deleted file gracefully
        await dep_graph.update_file(main_path)
        assert not dep_graph.graph.has_node("main.py")


# ── Tests: JS/TS dependency extraction ───────────────────────────────────


class TestJsTsDependencies:
    @pytest.mark.asyncio
    async def test_js_import_statements(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        with open(os.path.join(temp_dir, "app.js"), "w") as f:
            f.write("""
import { helper } from './utils/helper';
import User from './models/user';
const express = require('express');
""")
        await dep_graph.build(temp_dir)
        assert dep_graph.graph.has_node("app.js")

    @pytest.mark.asyncio
    async def test_ts_import_statements(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        with open(os.path.join(temp_dir, "app.ts"), "w") as f:
            f.write("""
import { User } from './models/user';
import type { ID } from './types';
""")
        await dep_graph.build(temp_dir)
        assert dep_graph.graph.has_node("app.ts")

    @pytest.mark.asyncio
    async def test_dynamic_import(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        with open(os.path.join(temp_dir, "app.js"), "w") as f:
            f.write('const mod = import("./dynamic.js");\n')
        await dep_graph.build(temp_dir)
        assert dep_graph.graph.has_node("app.js")


# ── Tests: Serialization ─────────────────────────────────────────────────


class TestSerialization:
    @pytest.mark.asyncio
    async def test_to_dict(
        self, dep_graph: DependencyGraph, py_project: str
    ) -> None:
        await dep_graph.build(py_project)
        d = dep_graph.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0

    @pytest.mark.asyncio
    async def test_to_dict_empty(self, dep_graph: DependencyGraph) -> None:
        d = dep_graph.to_dict()
        assert d == {}


# ── Tests: Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_project(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        await dep_graph.build(temp_dir)
        assert len(dep_graph.graph.nodes()) == 0

    @pytest.mark.asyncio
    async def test_no_source_files(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        with open(os.path.join(temp_dir, "readme.md"), "w") as f:
            f.write("# readme\n")
        await dep_graph.build(temp_dir)
        assert len(dep_graph.graph.nodes()) == 0

    @pytest.mark.asyncio
    async def test_build_nonexistent_dir(
        self, dep_graph: DependencyGraph
    ) -> None:
        with pytest.raises(NotADirectoryError):
            await dep_graph.build("/nonexistent/dir")

    @pytest.mark.asyncio
    async def test_package_init_imports(
        self, dep_graph: DependencyGraph, temp_dir: str
    ) -> None:
        """from . import X should resolve to the submodule, not __init__.py itself."""
        pkg = os.path.join(temp_dir, "pkg")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w") as f:
            f.write("from . import sub\n")
        with open(os.path.join(pkg, "sub.py"), "w") as f:
            f.write("x = 1\n")

        await dep_graph.build(temp_dir)
        deps = dep_graph.get_dependencies("pkg/__init__.py")
        assert "pkg/sub.py" in deps
        assert "pkg/__init__.py" not in deps  # no self-loop
