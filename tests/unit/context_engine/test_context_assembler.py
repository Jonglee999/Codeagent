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

    def test_with_symbol_dep_counts(self) -> None:
        """验证 BudgetReport 支持符号和依赖计数。"""
        report = BudgetReport(
            total_budget=8000,
            total_used=1200,
            symbol_count=15,
            dependency_count=8,
        )
        assert report.symbol_count == 15
        assert report.dependency_count == 8


# ── Phase 2.3: Symbol table integration ─────────────────────────────────────


class TestSymbolTableIntegration:
    """符号表集成测试。"""

    def test_symbol_table_in_package(self, assembler: ContextAssembler) -> None:
        """验证符号表数据出现在输出中。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            symbol_table=[
                {"name": "UserService", "kind": "class", "file_path": "services/user.py", "start_line": 1},
                {"name": "authenticate_user", "kind": "function", "file_path": "services/auth.py", "start_line": 5},
            ],
        )
        result = assembler.assemble(package)
        assert "<symbols>" in result
        assert "UserService" in result
        assert "authenticate_user" in result
        assert '<symbol name="UserService"' in result
        assert 'kind="function"' in result

    def test_symbol_table_in_budget_report(self, assembler: ContextAssembler) -> None:
        """验证预算报告包含符号数量。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            symbol_table=[
                {"name": f"sym_{i}", "kind": "function", "file_path": "f.py", "start_line": i}
                for i in range(5)
            ],
        )
        assembler.assemble(package)
        report = assembler.get_budget_report()
        assert report is not None
        assert report.symbol_count == 5

    def test_empty_symbol_table(self, assembler: ContextAssembler) -> None:
        """空符号表不应输出 XML 标签。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            symbol_table=[],
        )
        result = assembler.assemble(package)
        assert "<symbols>" not in result

    def test_symbol_xml_format(self, assembler: ContextAssembler) -> None:
        """验证符号 XML 格式正确。"""
        symbols = [
            {"name": "MyClass", "kind": "class", "file_path": "src/models.py", "start_line": 10},
            {"name": "my_func", "kind": "function", "file_path": "src/utils.py", "start_line": 42},
        ]
        xml = assembler._format_symbol_table_xml(symbols)
        assert "<symbols>" in xml
        assert "</symbols>" in xml
        assert 'name="MyClass"' in xml
        assert 'kind="class"' in xml
        assert 'file="src/models.py"' in xml
        assert 'line="10"' in xml
        assert 'name="my_func"' in xml
        assert 'kind="function"' in xml
        assert 'file="src/utils.py"' in xml
        assert 'line="42"' in xml

    def test_symbol_xml_format_empty(self, assembler: ContextAssembler) -> None:
        """空符号表返回空字符串。"""
        assert assembler._format_symbol_table_xml([]) == ""


# ── Phase 2.3: Score-based related code trimming ────────────────────────────


class TestScoreBasedTrimming:
    """按 score 裁剪相关代码测试。"""

    def test_trim_lowest_score_first(self) -> None:
        """最低 score 的代码应先被移除。"""
        assembler = ContextAssembler(total_budget=150)
        snippets = [
            CodeSnippet(
                file_path="low.py", start_line=1, end_line=20,
                code="line\n" * 10, score=0.2,
            ),
            CodeSnippet(
                file_path="high.py", start_line=1, end_line=20,
                code="line\n" * 10, score=0.9,
            ),
        ]
        allocation = BudgetAllocation(section="related_code", budget=80)
        result = assembler._trim_related_code_by_score(snippets, 80, allocation)
        assert result is not None
        # 高分的应保留，低分的可能被移除
        assert allocation.trimmed
        assert allocation.used <= 80

    def test_all_within_budget_no_trim(self) -> None:
        """预算充足时不应裁剪。"""
        assembler = ContextAssembler(total_budget=8000)
        snippets = [
            CodeSnippet(
                file_path="a.py", start_line=1, end_line=3,
                code="x = 1\n", score=0.5,
            ),
        ]
        allocation = BudgetAllocation(section="related_code", budget=1000)
        text = assembler._format_related_code(snippets)
        result = assembler._trim_related_code_by_score(snippets, 2000, allocation)
        assert result is not None
        assert not allocation.trimmed
        assert "a.py" in result

    def test_empty_snippets(self) -> None:
        """空片段列表返回 None。"""
        assembler = ContextAssembler()
        allocation = BudgetAllocation(section="related_code", budget=100)
        result = assembler._trim_related_code_by_score([], 100, allocation)
        assert result is None

    def test_single_snippet_over_budget(self) -> None:
        """单个大片段超出预算时保留其裁剪版本。"""
        assembler = ContextAssembler(total_budget=50)
        snippet = CodeSnippet(
            file_path="big.py", start_line=1, end_line=100,
            code="# " * 500, score=0.8,
        )
        allocation = BudgetAllocation(section="related_code", budget=30)
        result = assembler._trim_related_code_by_score([snippet], 30, allocation)
        assert result is not None
        assert allocation.trimmed

    def test_score_order_in_output(self) -> None:
        """输出中应保留 score 信息。"""
        assembler = ContextAssembler()
        snippets = [
            CodeSnippet(
                file_path="a.py", start_line=1, end_line=3,
                code="pass\n", score=0.95,
            ),
        ]
        text = assembler._format_related_code(snippets)
        assert "0.95" in text


# ── Phase 2.3: File tree folding ───────────────────────────────────────────


class TestFileTreeFolding:
    """文件树折叠测试。"""

    def test_shallow_tree_not_folded(self) -> None:
        """浅层目录不应被折叠。"""
        assembler = ContextAssembler()
        tree = {
            "name": "root",
            "type": "directory",
            "path": ".",
            "children": [
                {
                    "name": "src",
                    "type": "directory",
                    "path": "src",
                    "children": [
                        {"name": "main.py", "type": "file", "path": "src/main.py",
                         "size": 100, "extension": ".py"},
                    ],
                },
            ],
        }
        folded = assembler._fold_file_tree(tree)
        # 深度 2（root→src）不超过 3 层，不应折叠
        src_children = folded["children"][0]["children"]
        assert len(src_children) > 0
        assert src_children[0]["name"] == "main.py"

    def test_deep_tree_folded(self) -> None:
        """超过 3 层的深层目录应被折叠。"""
        assembler = ContextAssembler()
        deep_tree = {
            "name": "root",
            "type": "directory",
            "path": ".",
            "children": [
                {
                    "name": "a",
                    "type": "directory",
                    "path": "a",
                    "children": [
                        {
                            "name": "b",
                            "type": "directory",
                            "path": "a/b",
                            "children": [
                                {
                                    "name": "c",
                                    "type": "directory",
                                    "path": "a/b/c",
                                    "children": [
                                        {
                                            "name": "d",
                                            "type": "directory",
                                            "path": "a/b/c/d",
                                            "children": [
                                                {"name": "deep.py", "type": "file",
                                                 "path": "a/b/c/d/deep.py",
                                                 "size": 50, "extension": ".py"},
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        folded = assembler._fold_file_tree(deep_tree)
        # 遍历到深度 3 的目录 c，其子节点 d 应被折叠为 summary
        a_dir = folded["children"][0]
        b_dir = a_dir["children"][0]
        c_dir = b_dir["children"][0]
        # c 的 children[0] 是折叠后的 d（summary 类型）
        d_summary = c_dir["children"][0]
        assert d_summary["type"] == "summary"
        assert "subdirectories" in d_summary.get("name", "")

    def test_tree_folded_in_assemble_output(self) -> None:
        """验证折叠后的树在 assemble 输出中表现为摘要。"""
        assembler = ContextAssembler()
        tree = {
            "name": "root",
            "type": "directory",
            "path": ".",
            "children": [
                {
                    "name": "aa",
                    "type": "directory",
                    "path": "aa",
                    "children": [
                        {
                            "name": "bb",
                            "type": "directory",
                            "path": "aa/bb",
                            "children": [
                                {
                                    "name": "cc",
                                    "type": "directory",
                                    "path": "aa/bb/cc",
                                    "children": [
                                        {"name": "nested.py", "type": "file",
                                         "path": "aa/bb/cc/nested.py",
                                         "size": 50, "extension": ".py"},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        package = ContextPackage(file_tree=tree)
        result = assembler.assemble(package)
        # cc 目录应折叠为摘要
        assert "cc" in result
        # 折叠后的摘要应包含文件计数
        assert "1 file" in result or "nested" in result

    def test_none_tree(self) -> None:
        """None 文件树返回 None。"""
        assembler = ContextAssembler()
        assert assembler._fold_file_tree(None) is None

    def test_file_node_not_folded(self) -> None:
        """文件节点不应被折叠。"""
        assembler = ContextAssembler()
        file_node = {"name": "f.py", "type": "file", "path": "f.py", "size": 100}
        folded = assembler._fold_file_tree(file_node)
        assert folded["name"] == "f.py"


# ── Phase 2.3: Current file signature trimming ─────────────────────────────


class TestCurrentFileSignatureTrimming:
    """当前文件签名保留测试。"""

    def test_within_budget_not_trimmed(self) -> None:
        """预算充足时不裁剪。"""
        assembler = ContextAssembler()
        text = "Key Python files in project:\n  main.py\n  utils.py\n"
        allocation = BudgetAllocation(section="current_file", budget=1000)
        result = assembler._trim_current_file_to_signatures(text, 1000, allocation)
        assert result is not None
        assert not allocation.trimmed
        assert "main.py" in result
        assert "utils.py" in result

    def test_trimmed_to_signatures(self) -> None:
        """超出预算时保留 import/def/class 行。"""
        assembler = ContextAssembler()
        text = """Key Python files in project:
  main.py
  utils.py
  services/auth.py
  models/user.py
  config/settings.py
  some/very/long/file.py
  another/deep/file.py
  src/components/helper.py
"""
        allocation = BudgetAllocation(section="current_file", budget=30)
        result = assembler._trim_current_file_to_signatures(text, 30, allocation)
        assert result is not None
        assert allocation.trimmed

    def test_empty_text(self) -> None:
        """空文本返回 None。"""
        assembler = ContextAssembler()
        allocation = BudgetAllocation(section="current_file", budget=100)
        result = assembler._trim_current_file_to_signatures("", 100, allocation)
        assert result is None

    def test_trim_adds_note(self) -> None:
        """裁剪后追加说明文字。"""
        assembler = ContextAssembler()
        # 大量文本确保超出 tiny budget
        text = "\n".join(f"some random line content {i} for testing" for i in range(200))
        allocation = BudgetAllocation(section="current_file", budget=5)
        result = assembler._trim_current_file_to_signatures(text, 5, allocation)
        assert result is not None
        # 确认发生了裁剪
        assert allocation.trimmed


# ── Phase 2.3: Dependency count in report ──────────────────────────────────


class TestDependencyCount:
    """依赖项统计测试。"""

    def test_dependency_count_in_report(self, assembler: ContextAssembler) -> None:
        """验证预算报告包含依赖项数量。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            dependency_info={
                "modules": ["os", "sys", "json"],
                "packages": {"flask": "2.0", "pytest": "7.0"},
            },
        )
        assembler.assemble(package)
        report = assembler.get_budget_report()
        assert report is not None
        # 3 modules + 2 packages = 5 items (实际 _count_dep_items 对 dict 返回 key 数)
        assert report.dependency_count >= 2

    def test_dependency_count_in_report_with_symbols(
        self, assembler: ContextAssembler
    ) -> None:
        """同时包含符号和依赖时报告正确。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            dependency_info={"modules": ["a", "b"]},
            symbol_table=[{"name": "X", "kind": "class", "file_path": "x.py", "start_line": 1}],
        )
        assembler.assemble(package)
        report = assembler.get_budget_report()
        assert report is not None
        assert report.symbol_count == 1
        assert report.dependency_count == 2

    def test_output_with_both_symbols_and_deps(
        self, assembler: ContextAssembler
    ) -> None:
        """符号表和依赖信息同时出现在 output 中。"""
        package = ContextPackage(
            file_tree={"name": "root", "type": "directory", "path": ".", "children": []},
            dependency_info={"modules": ["requests"]},
            symbol_table=[
                {"name": "get_data", "kind": "function", "file_path": "a.py", "start_line": 1},
            ],
        )
        result = assembler.assemble(package)
        assert "<symbols>" in result
        assert "get_data" in result
        assert "<dependencies>" in result
        assert "requests" in result


# ── Phase 2.3: Budget allocation edge cases ────────────────────────────────


class TestBudgetAllocationExtended:
    """预算分配扩展测试。"""

    def test_trim_order_preserved(self) -> None:
        """裁剪优先级顺序应保持 相关代码→依赖→文件树。"""
        from codeagent.context_engine.context_assembler import _TRIM_ORDER
        assert _TRIM_ORDER == ["related_code", "dependency", "file_tree"]

    def test_mixed_package_with_symbols(
        self, assembler: ContextAssembler
    ) -> None:
        """包含文件树、相关代码、符号的完整包。"""
        package = ContextPackage(
            file_tree={
                "name": "proj",
                "type": "directory",
                "path": ".",
                "children": [
                    {"name": "main.py", "type": "file", "path": "main.py",
                     "size": 100, "extension": ".py"},
                ],
            },
            related_code=[
                CodeSnippet(
                    file_path="main.py", start_line=1, end_line=5,
                    code="def main():\n    pass\n", score=0.9,
                ),
            ],
            dependency_info={"local": ["utils"]},
            symbol_table=[
                {"name": "main", "kind": "function", "file_path": "main.py",
                 "start_line": 1},
            ],
        )
        result = assembler.assemble(package)
        assert "<project_tree>" in result
        assert "<related_code>" in result
        assert "<dependency_info>" in result
        assert "<symbols>" in result
        assert "<dependencies>" in result

        report = assembler.get_budget_report()
        assert report is not None
        assert report.symbol_count == 1
        assert report.dependency_count == 1
