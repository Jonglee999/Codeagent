"""ErrorAnalyzer 单元测试。

覆盖 traceback 解析、错误归属、修复建议生成、摘要提取、边界情况。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeagent.validation.error_analyzer import ErrorAnalyzer, FixSuggestion


# ── 夹具：模拟 pytest 输出 ─────────────────────────────────────


def _make_pytest_output(
    body: str,
    summary_failed: list[str] | None = None,
    stats: str = "1 failed, 2 passed in 0.12s",
) -> str:
    """构建模拟的 pytest 完整输出。"""
    lines = [
        "============================= test session starts =============================",
        "collected 3 items",
        "",
        "tests/test_sample.py F",
        "",
        "================================= FAILURES ===================================",
    ]
    lines.append(body)

    if summary_failed:
        lines.append("=========================== short test summary info ===========================")
        for sf in summary_failed:
            lines.append(f"FAILED {sf}")

    lines.append(f"============================== {stats} ==============================")
    return "\n".join(lines)


SIMPLE_ASSERTION_ERROR = _make_pytest_output(
    body="""_______________________________ test_fail _________________________________

    def test_fail():
>       assert 1 == 2
E       assert 1 == 2

tests/test_sample.py:3: AssertionError
""",
    summary_failed=["tests/test_sample.py::test_fail - AssertionError: assert 1 == 2"],
)


MULTI_LEVEL_TRACEBACK = _make_pytest_output(
    body="""_______________________________ test_outer ________________________________

tests/test_sample.py:10: in test_outer
    result = process_data()
tests/utils.py:5: in process_data
    return 1 / 0
E   ZeroDivisionError: division by zero
""",
    summary_failed=["tests/test_sample.py::test_outer - ZeroDivisionError: division by zero"],
)


TYPE_ERROR = _make_pytest_output(
    body="""_______________________________ test_type_error _____________________________

    def test_type_error():
>       add(1, "2")
E       TypeError: unsupported operand type(s) for +: 'int' and 'str'

tests/test_sample.py:8: TypeError
""",
    summary_failed=["tests/test_sample.py::test_type_error - TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
)


IMPORT_ERROR = _make_pytest_output(
    body="""_______________________________ test_import_error ___________________________

    def test_import_error():
>       from missing_module import magic
E       ModuleNotFoundError: No module named 'missing_module'

tests/test_sample.py:12: ModuleNotFoundError
""",
    summary_failed=["tests/test_sample.py::test_import_error - ModuleNotFoundError: No module named 'missing_module'"],
)


MULTIPLE_FAILURES = _make_pytest_output(
    body="""_______________________________ test_first _________________________________

    def test_first():
>       assert True == False
E       assert True == False

tests/test_a.py:3: AssertionError

_______________________________ test_second ________________________________

    def test_second():
>       raise ValueError("bad")
E       ValueError: bad

tests/test_b.py:7: ValueError
""",
    summary_failed=[
        "tests/test_a.py::test_first - AssertionError: assert True == False",
        "tests/test_b.py::test_second - ValueError: bad",
    ],
    stats="2 failed, 1 passed in 0.15s",
)


ALL_PASS_OUTPUT = """\
============================= test session starts =============================
collected 3 items

tests/test_a.py .
tests/test_b.py .
tests/test_c.py .

============================== 3 passed in 0.10s ==============================
"""


TRACEBACK_WITH_PATH_TRAVERSAL = _make_pytest_output(
    body="""_______________________________ test_deep _________________________________

app/services/user_service.py:42: in create_user
    result = db.query(User).filter_by(id=user_id)
app/models/user.py:28: in filter_by
    return self._filter(id=value)
E   AttributeError: 'NoneType' object has no attribute 'filter'

tests/test_services.py:15: AttributeError
""",
    summary_failed=["tests/test_services.py::test_deep - AttributeError: 'NoneType' object has no attribute 'filter'"],
)


# ── Traceback 解析 ──────────────────────────────────────────────


class TestTracebackParsing:
    """Traceback 解析能力测试。"""

    def test_parse_assertion_error(self) -> None:
        """解析简单的 AssertionError。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(SIMPLE_ASSERTION_ERROR)

        assert len(failures) == 1
        assert failures[0].test_name == "test_fail"
        assert failures[0].error_type == "AssertionError"
        assert "assert" in failures[0].message

    def test_parse_multi_level_traceback(self) -> None:
        """解析多层调用 traceback。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(MULTI_LEVEL_TRACEBACK)

        assert len(failures) == 1
        assert failures[0].test_name == "test_outer"
        assert failures[0].error_type == "ZeroDivisionError"
        assert "division by zero" in failures[0].message

    def test_parse_type_error(self) -> None:
        """解析 TypeError。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(TYPE_ERROR)

        assert len(failures) == 1
        assert failures[0].test_name == "test_type_error"
        assert failures[0].error_type == "TypeError"
        assert "unsupported operand" in failures[0].message

    def test_parse_import_error(self) -> None:
        """解析 ImportError/ModuleNotFoundError。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(IMPORT_ERROR)

        assert len(failures) == 1
        assert failures[0].error_type == "ModuleNotFoundError"
        assert "missing_module" in failures[0].message

    def test_parse_multiple_failures(self) -> None:
        """解析多个测试失败。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(MULTIPLE_FAILURES)

        assert len(failures) == 2
        assert failures[0].test_name == "test_first"
        assert failures[1].test_name == "test_second"

    def test_parse_empty_output(self) -> None:
        """空输出返回空列表。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback("")
        assert failures == []

    def test_parse_all_pass_output(self) -> None:
        """全部通过的输出返回空列表。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(ALL_PASS_OUTPUT)
        assert failures == []

    def test_parse_traceback_with_path_traversal(self) -> None:
        """解析跨文件的 traceback。"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(TRACEBACK_WITH_PATH_TRAVERSAL)

        assert len(failures) == 1
        assert failures[0].test_name == "test_deep"
        assert failures[0].error_type == "AttributeError"
        # traceback 中应包含跨文件路径
        assert len(failures[0].traceback_lines) >= 2


# ── 错误归属 ────────────────────────────────────────────────────


class TestErrorMapping:
    """错误归属测试（新错误 vs 已有错误）。"""

    def test_new_error_in_changed_file(self) -> None:
        """错误文件在 changed_files 中 → is_pre_existing=False。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            SIMPLE_ASSERTION_ERROR,
            changed_files=["tests/test_sample.py"],
        )

        assert len(suggestions) == 1
        assert not suggestions[0].is_pre_existing
        assert any("test_sample.py" in f for f in suggestions[0].affected_files)

    def test_pre_existing_error(self) -> None:
        """错误文件不在 changed_files 中 → is_pre_existing=True。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            SIMPLE_ASSERTION_ERROR,
            changed_files=["src/main.py"],
        )

        assert len(suggestions) == 1
        assert suggestions[0].is_pre_existing
        assert any("test_sample.py" in f for f in suggestions[0].affected_files)

    def test_mixed_errors(self) -> None:
        """混合错误：一些文件修改过，一些没有。"""
        analyzer = ErrorAnalyzer()
        # test_first 在 test_a.py（已修改），test_second 在 test_b.py（未修改）
        suggestions = analyzer.analyze(
            MULTIPLE_FAILURES,
            changed_files=["tests/test_a.py"],
        )

        assert len(suggestions) == 2
        # test_first (AssertionError) 在 test_a.py 中 → 新错误
        first = next(s for s in suggestions if "assert True" in s.error_summary)
        assert not first.is_pre_existing
        # test_second (ValueError) 在 test_b.py 中 → 已有错误
        second = next(s for s in suggestions if "bad" in s.error_summary)
        assert second.is_pre_existing

    def test_traceback_path_resolution(self) -> None:
        """traceback 中涉及 changed_files 也应标记为新错误。"""
        analyzer = ErrorAnalyzer()
        # 即使 tests/test_services.py 未修改，traceback 中 app/services/user_service.py 已修改
        suggestions = analyzer.analyze(
            TRACEBACK_WITH_PATH_TRAVERSAL,
            changed_files=["app/services/user_service.py"],
        )

        assert len(suggestions) == 1
        assert not suggestions[0].is_pre_existing
        # affected_files 应包含 app/services/user_service.py
        assert any("user_service" in f for f in suggestions[0].affected_files)

    def test_empty_changed_files(self) -> None:
        """空 changed_files → 所有错误标记为已有。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            SIMPLE_ASSERTION_ERROR, changed_files=[]
        )

        assert len(suggestions) == 1
        assert suggestions[0].is_pre_existing


# ── 修复建议 ────────────────────────────────────────────────────


class TestFixSuggestion:
    """FixSuggestion 生成测试。"""

    def test_assertion_error_suggestion(self) -> None:
        """AssertionError 生成适当修复建议。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            SIMPLE_ASSERTION_ERROR, changed_files=["tests/test_sample.py"]
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        assert "assert" in s.likely_cause.lower()
        assert "assertion" in s.suggested_fix.lower()

    def test_import_error_suggestion(self) -> None:
        """ImportError 生成适当修复建议。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            IMPORT_ERROR, changed_files=["tests/test_sample.py"]
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.error_summary and "missing_module" in s.error_summary
        assert "module" in s.suggested_fix.lower()

    def test_type_error_suggestion(self) -> None:
        """TypeError 生成适当修复建议。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            TYPE_ERROR, changed_files=["tests/test_sample.py"]
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        assert "type" in s.likely_cause.lower()
        assert "signature" in s.suggested_fix.lower()

    def test_zero_division_suggestion(self) -> None:
        """ZeroDivisionError 生成适当修复建议。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            MULTI_LEVEL_TRACEBACK,
            changed_files=["tests/utils.py"],
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        assert "division by zero" in s.error_summary.lower()
        assert "zero" in s.suggested_fix.lower()

    def test_unknown_error_type(self) -> None:
        """未知错误类型有合理的兜底建议。"""
        output = _make_pytest_output(
            body="""_______________________________ test_weird ________________________________

    def test_weird():
>       raise CustomWeirdError("something strange")
E       CustomWeirdError: something strange

tests/test_strange.py:5: CustomWeirdError
""",
            summary_failed=["tests/test_strange.py::test_weird - CustomWeirdError: something strange"],
        )
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            output, changed_files=["tests/test_strange.py"]
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        # 未知类型应返回兜底建议
        assert s.likely_cause
        assert s.suggested_fix

    def test_fix_suggestion_fields(self) -> None:
        """FixSuggestion 关键字段非空。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(
            SIMPLE_ASSERTION_ERROR, changed_files=["tests/test_sample.py"]
        )

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.error_summary
        assert s.affected_files
        assert s.likely_cause
        assert s.suggested_fix
        assert isinstance(s.is_pre_existing, bool)


# ── 摘要提取 ────────────────────────────────────────────────────


class TestSummaryExtraction:
    """测试摘要提取。"""

    def test_extract_summary_with_failures(self) -> None:
        """提取失败测试的摘要信息。"""
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary(SIMPLE_ASSERTION_ERROR)

        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["total"] == 3
        assert len(summary["failed_tests"]) == 1

    def test_extract_summary_all_pass(self) -> None:
        """全部通过的摘要。"""
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary(ALL_PASS_OUTPUT)

        assert summary["passed"] == 3
        assert summary["failed"] == 0
        assert summary["total"] == 3
        assert len(summary["failed_tests"]) == 0

    def test_extract_summary_multiple_failures(self) -> None:
        """多个失败的摘要。"""
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary(MULTIPLE_FAILURES)

        assert summary["failed"] == 2
        assert summary["passed"] == 1
        assert summary["total"] == 3
        assert len(summary["failed_tests"]) == 2

    def test_extract_summary_errors(self) -> None:
        """含 errors 的摘要。"""
        output = _make_pytest_output(
            body="""_______________________________ test_error _______________________________

    def test_error():
>       raise RuntimeError("system error")
E       RuntimeError: system error

tests/test_error.py:3: RuntimeError
""",
            summary_failed=["tests/test_error.py::test_error - RuntimeError: system error"],
            stats="1 failed, 1 passed, 1 error in 0.12s",
        )
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary(output)

        assert summary["failed"] == 1
        assert summary["passed"] == 1
        assert summary["errors"] == 1
        assert summary["total"] == 3

    def test_extract_summary_empty_output(self) -> None:
        """空输出返回零值。"""
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary("")

        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["failed_tests"] == []


# ── 边界情况 ────────────────────────────────────────────────────


class TestEdgeCases:
    """边界情况测试。"""

    def test_output_with_only_summary(self) -> None:
        """只有 FAILED 摘要行，没有完整 FAILURES 块。"""
        output = """\
============================= test session starts =============================
collected 1 item

tests/test_x.py F

=========================== short test summary info ===========================
FAILED tests/test_x.py::test_x - AssertionError: assert False
============================== 1 failed in 0.05s ==============================
"""
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(output)

        # 应通过 FAILED 摘要行解析
        assert len(failures) == 1
        assert failures[0].test_name == "test_x"
        assert failures[0].error_type == "AssertionError"

    def test_no_newline_at_end(self) -> None:
        """输出末尾没有换行。"""
        output = SIMPLE_ASSERTION_ERROR.rstrip("\n")
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(output)

        assert len(failures) >= 1

    def test_unicode_in_output(self) -> None:
        """输出含中文/Unicode 字符。"""
        output = _make_pytest_output(
            body="""_______________________________ test_unicode ______________________________

    def test_unicode():
>       assert "你好" == "世界"
E       AssertionError: assert '你好' == '世界'

tests/test_unicode.py:3: AssertionError
""",
            summary_failed=["tests/test_unicode.py::test_unicode - AssertionError: assert '你好' == '世界'"],
        )
        analyzer = ErrorAnalyzer()
        failures = analyzer._parse_traceback(output)

        assert len(failures) == 1
        assert "你好" in failures[0].message or "世界" in failures[0].message

    def test_analyze_with_empty_output(self) -> None:
        """analyze() 传入空输出返回空列表。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze("", changed_files=["test.py"])
        assert suggestions == []

    def test_analyze_with_no_changed_files(self) -> None:
        """analyze() 传入无 changed_files 仍能返回建议。"""
        analyzer = ErrorAnalyzer()
        suggestions = analyzer.analyze(SIMPLE_ASSERTION_ERROR, changed_files=[])
        assert len(suggestions) == 1
        assert suggestions[0].is_pre_existing

    def test_infer_cause_unknown_type(self) -> None:
        """未知错误类型返回通用原因。"""
        analyzer = ErrorAnalyzer()
        from codeagent.validation.error_analyzer import TestFailure

        failure = TestFailure(
            test_name="test_unknown",
            file_path="test.py",
            line=1,
            error_type="CustomInternalError",
            message="Something went wrong internally",
        )
        cause = analyzer._infer_cause(failure)
        assert "CustomInternalError" in cause

    def test_extract_summary_only_passed(self) -> None:
        """只有 passed 的摘要行。"""
        output = "============================== 5 passed in 0.10s =============================="
        analyzer = ErrorAnalyzer()
        summary = analyzer.extract_summary(output)

        assert summary["passed"] == 5
        assert summary["failed"] == 0
        assert summary["errors"] == 0
