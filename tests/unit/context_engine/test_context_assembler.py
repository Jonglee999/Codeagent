"""ContextAssembler 单元测试。"""

from __future__ import annotations

import pytest

from codeagent.context_engine.context_assembler import (
    BudgetAllocation,
    BudgetReport,
    ContextAssembler,
)
from codeagent.gateway.context_gateway import CodeSnippet, ContextPackage


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def assembler() -> ContextAssembler:
    return ContextAssembler(total_budget=8000)


@pytest.fixture
def sample_package() -> ContextPackage:
    """包含基本上下文的数据包。"""
    return ContextPackage(
        file_tree={
            "name": "myproject",
            "type": "directory",
            "path": ".",
            "children": [
                {"name": "main.py", "type": "file", "path": "main.py", "size": 100, "extension": ".py"},
                {"name": "utils.py", "type": "file", "path": "utils.py", "size": 200, "extension": ".py"},
                {
                    "name": "src",
                    "type": "directory",
                    "path": "src",
                    "children": [
                        {"name": "__init__.py", "type": "file", "path": "src/__init__.py", "size": 0, "extension": ".py"},
                    ],
                },
            ],
        },
        related_code=[
            CodeSnippet(
                file_path="main.py",
                start_line=1,
                end_line=5,
                code="def hello():\n    print('hello')\n",
                score=0.95,
            ),
            CodeSnippet(
                file_path="utils.py",
                start_line=10,
                end_line=15,
                code="def util():\n    return 42\n",
                score=0.80,
            ),
        ],
        dependency_info={
            "python_packages": ["pytest", "click"],
            "local_modules": ["utils", "src"],
        },
    )


# ── Tests: Basic assembly ─────────────────────────────────────────────────


class TestBasicAssembly:
    """基本组装功能测试。"""

    def test_assemble_returns_string(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        result = assembler.assemble(sample_package)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_assemble_contains_xml_tags(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        result = assembler.assemble(sample_package)
        assert "<project_tree>" in result
        assert "</project_tree>" in result
        assert "<related_code>" in result
        assert "</related_code>" in result
        assert "<dependency_info>" in result
        assert "</dependency_info>" in result

    def test_assemble_contains_file_tree(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        result = assembler.assemble(sample_package)
        assert "myproject/" in result
        assert "main.py" in result
        assert "utils.py" in result
        assert "src/" in result

    def test_assemble_contains_related_code(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        result = assembler.assemble(sample_package)
        assert "main.py:1-5" in result
        assert "utils.py:10-15" in result
        assert "def hello()" in result
        assert "def util()" in result

    def test_assemble_contains_dependency_info(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        result = assembler.assemble(sample_package)
        assert "pytest" in result
        assert "click" in result
        assert "utils" in result


# ── Tests: Budget report ─────────────────────────────────────────────────


class TestBudgetReport:
    """预算报告测试。"""

    def test_get_budget_report_before_assemble(self, assembler: ContextAssembler) -> None:
        assert assembler.get_budget_report() is None

    def test_get_budget_report_after_assemble(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        assembler.assemble(sample_package)
        report = assembler.get_budget_report()
        assert report is not None
        assert isinstance(report, BudgetReport)
        assert report.total_budget == 8000
        assert report.total_used > 0

    def test_report_contains_allocations(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        assembler.assemble(sample_package)
        report = assembler.get_budget_report()
        assert report is not None
        sections = {a.section for a in report.allocations}
        assert "file_tree" in sections
        assert "related_code" in sections
        assert "current_file" in sections
        assert "dependency" in sections

    def test_report_budget_distribution(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        assembler.assemble(sample_package)
        report = assembler.get_budget_report()
        assert report is not None
        for alloc in report.allocations:
            assert alloc.budget > 0
            assert alloc.used >= 0

    def test_total_used_within_budget(self, assembler: ContextAssembler, sample_package: ContextPackage) -> None:
        assembler.assemble(sample_package)
        report = assembler.get_budget_report()
        assert report is not None
        assert report.total_used <= report.total_budget


# ── Tests: Empty inputs ──────────────────────────────────────────────────


class TestEmptyInputs:
    """空输入场景测试。"""

    def test_empty_package(self, assembler: ContextAssembler) -> None:
        """空数据包应返回空字符串。"""
        package = ContextPackage()
        result = assembler.assemble(package)
        assert result == ""

    def test_empty_related_code(self, assembler: ContextAssembler) -> None:
        package = ContextPackage(file_tree={"name": "root", "type": "directory", "path": ".", "children": []})
        result = assembler.assemble(package)
        assert "<project_tree>" in result
        assert "<related_code>" not in result  # 空时不输出

    def test_empty_dependency(self, assembler: ContextAssembler) -> None:
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
        )
        result = assembler.assemble(package)
        assert "<project_tree>" in result
        assert "<dependency_info>" not in result  # 空时不输出


# ── Tests: Token counting ────────────────────────────────────────────────


class TestTokenCounting:
    """Token 计数测试。"""

    def test_token_count_accuracy(self) -> None:
        """验证 tiktoken 计数的基本准确性。"""
        assembler = ContextAssembler()
        text = "Hello, world!"
        count = assembler._count_tokens(text)
        assert count > 0

    def test_large_text_token_count(self) -> None:
        assembler = ContextAssembler()
        text = "word " * 1000
        count = assembler._count_tokens(text)
        assert count > 100


# ── Tests: Budget trimming ───────────────────────────────────────────────


class TestBudgetTrimming:
    """预算裁剪逻辑测试。"""

    def test_within_budget_no_trim(self) -> None:
        """内容在预算内不应裁剪。"""
        assembler = ContextAssembler(total_budget=100000)  # 大预算
        package = ContextPackage(
            file_tree={
                "name": "big",
                "type": "directory",
                "path": ".",
                "children": [{"name": f"file_{i}.py", "type": "file", "path": f"file_{i}.py", "size": 10}
                             for i in range(100)],
            },
        )
        result = assembler.assemble(package)
        # 应有全部文件
        assert "file_99.py" in result

    def test_trim_related_code(self) -> None:
        """相关代码超预算时裁剪。"""
        assembler = ContextAssembler(total_budget=200)  # 小预算以触发裁剪
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            related_code=[
                CodeSnippet(file_path=f"f{i}.py", start_line=1, end_line=100,
                            code="x\n" * 100, score=0.5)
                for i in range(50)
            ],
        )
        result = assembler.assemble(package)
        report = assembler.get_budget_report()
        assert report is not None
        assert report.total_used <= report.total_budget

    def test_trim_file_tree(self) -> None:
        """文件树在深度裁剪时不应崩溃。"""
        assembler = ContextAssembler(total_budget=100)
        deep_tree = {
            "name": "deep",
            "type": "directory",
            "path": ".",
            "children": [
                {"name": f"file_{i}.py", "type": "file", "path": f"file_{i}.py",
                 "size": 1000, "extension": ".py"}
                for i in range(200)
            ],
        }
        package = ContextPackage(file_tree=deep_tree)
        result = assembler.assemble(package)
        assert isinstance(result, str)


# ── Tests: Current file section ──────────────────────────────────────────


class TestCurrentFile:
    """当前文件区块测试。"""

    def test_current_file_with_python_files(self, assembler: ContextAssembler) -> None:
        package = ContextPackage(
            file_tree={
                "name": "proj",
                "type": "directory",
                "path": ".",
                "children": [
                    {"name": "app.py", "type": "file", "path": "app.py",
                     "size": 100, "extension": ".py"},
                    {"name": "README.md", "type": "file", "path": "README.md",
                     "size": 50, "extension": ".md"},
                ],
            },
        )
        result = assembler.assemble(package)
        assert "app.py" in result
        assert "Key Python files" in result or "<current_file>" in result

    def test_current_file_no_python(self, assembler: ContextAssembler) -> None:
        package = ContextPackage(
            file_tree={
                "name": "proj",
                "type": "directory",
                "path": ".",
                "children": [
                    {"name": "README.md", "type": "file", "path": "README.md",
                     "size": 50, "extension": ".md"},
                ],
            },
        )
        result = assembler.assemble(package)
        # 无 .py 文件时，current_file 区块应为空
        assert "<current_file>" not in result


class TestEdgeCases:
    """ContextAssembler 边缘场景测试。"""

    def test_non_ascii_file_paths(self, assembler: ContextAssembler) -> None:
        """包含非 ASCII（中文）文件路径。"""
        package = ContextPackage(
            file_tree={
                "name": "项目",
                "type": "directory",
                "path": ".",
                "children": [
                    {"name": "中文文件.py", "type": "file",
                     "path": "中文文件.py", "size": 50, "extension": ".py"},
                    {"name": "データ.py", "type": "file",
                     "path": "データ.py", "size": 100, "extension": ".py"},
                ],
            },
        )
        result = assembler.assemble(package)
        assert "中文文件.py" in result
        assert "データ.py" in result

    def test_extremely_large_related_code_trimmed(self) -> None:
        """极小预算下超长 related_code 应被裁剪。"""
        assembler = ContextAssembler(total_budget=500)
        huge_code = "\n".join(f"# line {i}" + "x" * 50 for i in range(1000))
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            related_code=[
                CodeSnippet(
                    file_path="huge.py",
                    start_line=1,
                    end_line=1000,
                    code=huge_code,
                    score=0.5,
                ),
            ],
        )
        result = assembler.assemble(package)
        line_count = result.count("line ")
        assert line_count < 500

    def test_zero_budget_returns_minimal_output(self) -> None:
        """0 token 预算应输出极简内容（XML 标签框架）。"""
        assembler = ContextAssembler(total_budget=0)
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
        )
        result = assembler.assemble(package)
        assert "<project_tree>" in result

    def test_negative_budget_returns_minimal_output(self) -> None:
        """负 token 预算应输出极简内容。"""
        assembler = ContextAssembler(total_budget=-100)
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
        )
        result = assembler.assemble(package)
        assert "<project_tree>" in result


# ── Tests: Budget allocation dataclass ────────────────────────────────────


class TestBudgetAllocation:
    """BudgetAllocation 数据类测试。"""

    def test_default_values(self) -> None:
        alloc = BudgetAllocation(section="test", budget=1000)
        assert alloc.section == "test"
        assert alloc.budget == 1000
        assert alloc.used == 0
        assert alloc.trimmed is False

    def test_full_construction(self) -> None:
        alloc = BudgetAllocation(
            section="test", budget=1000, used=500, trimmed=True
        )
        assert alloc.used == 500
        assert alloc.trimmed is True


class TestBudgetReportDataclass:
    """BudgetReport 数据类测试。"""

    def test_default_values(self) -> None:
        report = BudgetReport(total_budget=8000, total_used=0)
        assert report.total_budget == 8000
        assert report.total_used == 0
        assert report.allocations == []

    def test_with_allocations(self) -> None:
        allocs = [BudgetAllocation(section="a", budget=100)]
        report = BudgetReport(
            total_budget=8000, total_used=50, allocations=allocs
        )
        assert report.total_used == 50
        assert len(report.allocations) == 1
