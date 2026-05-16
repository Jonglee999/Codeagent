"""StrategyExtractor 单元测试。

覆盖：
- Strategy 数据模型创建和默认值
- should_extract 触发条件判断（5 次成功、修复后触发、未触发）
- extract 正常提炼（Mock LLM 返回固定 JSON）
- extract JSON 解析失败降级（返回空列表）
- extract 置信度过滤
- extract 最大条数限制（最多 5 条）
- 去重检测（title 完全一致、内容高度相似跳过）
- extract_from_recent 委托 load_recent
- _tokenize / _keyword_score 关键词匹配
- _clean_json 清理 LLM 输出
- 异常降级（LLM 调用失败、store 不可用）
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest

from codeagent.context_engine.evolution import (
    Strategy,
    StrategyExtractor,
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStep,
)


# ── Mock Helpers ──────────────────────────────────────────────


class MockResponse:
    """模拟 LLM 响应对象。"""

    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


class MockChoice:
    def __init__(self, content: str) -> None:
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class MockStore:
    """模拟 StrategyStore，仅用于去重检查。"""

    def __init__(self, existing: Optional[list[Strategy]] = None) -> None:
        self._strategies = existing or []

    def list_active(self) -> list[Strategy]:
        return self._strategies


async def mock_llm_factory(content: str, delay: float = 0) -> Any:
    """创建返回固定内容的 Mock LLM 客户端。"""

    async def client(model: str = "", messages: list | None = None) -> MockResponse:
        if delay:
            await __import__("asyncio").sleep(delay)
        return MockResponse(content)

    return client


def make_trajectory(
    task_id: str = "",
    user_request: str = "Fix bug",
    steps_count: int = 3,
    final_status: str = "success",
    repair_rounds: int = 0,
    validation_passed: bool = True,
) -> Trajectory:
    """创建测试用 Trajectory 对象。"""
    tid = task_id or str(uuid.uuid4())
    now = datetime.now()
    steps = [
        TrajectoryStep(
            step_id=f"s{i}",
            node_name="execution",
            step_type="tool_call",
            timestamp=now + timedelta(seconds=i),
            input_summary=f"input_{i}",
            output_summary=f"output_{i}",
            duration_ms=100,
            success=True,
        )
        for i in range(steps_count)
    ]
    return Trajectory(
        task_id=tid,
        user_request=user_request,
        started_at=now,
        completed_at=now + timedelta(seconds=10),
        steps=steps,
        final_status=final_status,  # type: ignore[arg-type]
        repair_rounds=repair_rounds,
        validation_passed=validation_passed,
    )


_VALID_STRATEGY_JSON = """{
  "strategies": [
    {
      "title": "添加 FastAPI 路由时同步更新测试",
      "condition": "当需要在 FastAPI 项目中添加新路由时",
      "action": "应同时在 tests/ 下添加对应的测试文件",
      "rationale": "未同步写测试导致 CI 失败",
      "category": "workflow",
      "confidence": 0.85
    },
    {
      "title": "修改数据库模型后执行迁移",
      "condition": "当修改 SQLAlchemy 模型定义时",
      "action": "应生成对应的迁移脚本并执行",
      "rationale": "模型修改后未迁移导致运行时错误",
      "category": "workflow",
      "confidence": 0.75
    }
  ]
}"""


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


@pytest.fixture
async def empty_extractor(mock_store: MockStore) -> StrategyExtractor:
    """创建一个可用的 StrategyExtractor（LLM 返回空策略）。"""

    async def empty_llm(model: str = "", messages: list | None = None) -> MockResponse:
        return MockResponse('{"strategies": []}')

    return StrategyExtractor(
        llm_client=empty_llm,
        recorder=None,  # type: ignore[arg-type]
        store=mock_store,
    )


@pytest.fixture
def sample_trajectories() -> list[Trajectory]:
    """创建 5 条示例轨迹。"""
    return [
        make_trajectory(
            task_id=f"task-{i}",
            user_request=f"Task {i}: fix function",
            steps_count=4,
            final_status="success",
        )
        for i in range(5)
    ]


# ── Strategy Dataclass 测试 ───────────────────────────────────


class TestStrategy:
    """Strategy dataclass 测试。"""

    def test_minimal_creation(self) -> None:
        strategy = Strategy(
            strategy_id="s-1",
            title="Test strategy",
            condition="when X",
            action="do Y",
            rationale="because Z",
            category="workflow",
            source_task_ids=["task-1"],
        )
        assert strategy.strategy_id == "s-1"
        assert strategy.title == "Test strategy"
        assert strategy.condition == "when X"
        assert strategy.action == "do Y"
        assert strategy.rationale == "because Z"
        assert strategy.category == "workflow"
        assert strategy.source_task_ids == ["task-1"]
        assert strategy.confidence == 0.5
        assert strategy.applied_count == 0
        assert strategy.success_count == 0
        assert strategy.last_applied is None

    def test_full_creation(self) -> None:
        now = datetime.now()
        strategy = Strategy(
            strategy_id="s-2",
            title="Full strategy",
            condition="condition text",
            action="action text",
            rationale="rationale text",
            category="error_avoidance",
            source_task_ids=["t1", "t2"],
            confidence=0.95,
            created_at=now,
            applied_count=10,
            success_count=8,
            last_applied=now,
        )
        assert strategy.strategy_id == "s-2"
        assert strategy.category == "error_avoidance"
        assert strategy.confidence == 0.95
        assert strategy.applied_count == 10
        assert strategy.success_count == 8
        assert strategy.last_applied == now

    def test_all_categories(self) -> None:
        for cat in ("workflow", "coding_style", "error_avoidance", "tool_usage"):
            strategy = Strategy(
                strategy_id=str(uuid.uuid4()),
                title="test",
                condition="c",
                action="a",
                rationale="r",
                category=cat,  # type: ignore[arg-type]
                source_task_ids=[],
            )
            assert strategy.category == cat

    def test_default_values(self) -> None:
        strategy = Strategy(
            strategy_id="s-3",
            title="Default values",
            condition="c",
            action="a",
            rationale="r",
            category="coding_style",
            source_task_ids=["t1"],
        )
        assert isinstance(strategy.created_at, datetime)
        assert strategy.confidence == 0.5
        assert strategy.applied_count == 0
        assert strategy.success_count == 0
        assert strategy.last_applied is None


# ── should_extract 测试 ───────────────────────────────────────


class TestShouldExtract:
    """should_extract 触发条件测试。"""

    @pytest.mark.asyncio
    async def test_should_extract_every_5_tasks(
        self, empty_extractor: StrategyExtractor,
    ) -> None:
        """每 5 次成功任务应触发。"""
        assert await empty_extractor.should_extract(task_count=5, last_had_repair=False) is True
        assert await empty_extractor.should_extract(task_count=10, last_had_repair=False) is True
        assert await empty_extractor.should_extract(task_count=15, last_had_repair=False) is True

    @pytest.mark.asyncio
    async def test_should_extract_after_repair(
        self, empty_extractor: StrategyExtractor,
    ) -> None:
        """经过修复的任务应触发。"""
        assert await empty_extractor.should_extract(task_count=1, last_had_repair=True) is True
        assert await empty_extractor.should_extract(task_count=3, last_had_repair=True) is True

    @pytest.mark.asyncio
    async def test_should_extract_not_triggered(
        self, empty_extractor: StrategyExtractor,
    ) -> None:
        """不满足条件时不应触发。"""
        assert await empty_extractor.should_extract(task_count=0, last_had_repair=False) is False
        assert await empty_extractor.should_extract(task_count=1, last_had_repair=False) is False
        assert await empty_extractor.should_extract(task_count=2, last_had_repair=False) is False
        assert await empty_extractor.should_extract(task_count=4, last_had_repair=False) is False

    @pytest.mark.asyncio
    async def test_should_extract_task_count_zero_with_repair(
        self, empty_extractor: StrategyExtractor,
    ) -> None:
        """task_count=0 但 last_had_repair=True 应触发（首次任务失败修复后）。"""
        assert await empty_extractor.should_extract(task_count=0, last_had_repair=True) is True


# ── extract 测试 ──────────────────────────────────────────────


class TestExtract:
    """extract 方法测试。"""

    @pytest.mark.asyncio
    async def test_extract_returns_strategies(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """正常提炼应返回 Strategy 对象列表。"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(_VALID_STRATEGY_JSON),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) == 2
        assert all(isinstance(s, Strategy) for s in strategies)
        assert strategies[0].title == "添加 FastAPI 路由时同步更新测试"
        assert strategies[0].category == "workflow"
        assert strategies[0].confidence == 0.85
        assert len(strategies[0].source_task_ids) == 5

    @pytest.mark.asyncio
    async def test_extract_empty_trajectories(
        self, mock_store: MockStore,
    ) -> None:
        """空轨迹列表应返回空列表。"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(_VALID_STRATEGY_JSON),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract([])
        assert strategies == []

    @pytest.mark.asyncio
    async def test_extract_json_parse_failure(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """LLM 返回非法 JSON 时应返回空列表。"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory("not valid json"),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert strategies == []

    @pytest.mark.asyncio
    async def test_extract_llm_call_failure(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """LLM 调用失败时应返回空列表。"""

        async def failing_llm(model: str = "", messages: list | None = None) -> None:
            raise RuntimeError("LLM connection failed")

        extractor = StrategyExtractor(
            llm_client=failing_llm,
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert strategies == []

    @pytest.mark.asyncio
    async def test_extract_filters_low_confidence(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """应过滤掉置信度低于 min_confidence 的策略。"""
        json_with_low = """{
          "strategies": [
            {
              "title": "Good strategy",
              "condition": "when X",
              "action": "do Y",
              "rationale": "because Z",
              "category": "workflow",
              "confidence": 0.8
            },
            {
              "title": "Low confidence strategy",
              "condition": "when A",
              "action": "do B",
              "rationale": "because C",
              "category": "coding_style",
              "confidence": 0.3
            }
          ]
        }"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(json_with_low),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
            min_confidence=0.5,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) == 1
        assert strategies[0].title == "Good strategy"

    @pytest.mark.asyncio
    async def test_extract_max_5_strategies(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """每次最多提炼 5 条策略。"""
        many_strategies = {
            "strategies": [
                {
                    "title": f"Strategy {i}",
                    "condition": f"when X{i}",
                    "action": f"do Y{i}",
                    "rationale": f"because Z{i}",
                    "category": "workflow",
                    "confidence": 0.9,
                }
                for i in range(10)
            ]
        }
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(str(many_strategies).replace("'", '"')),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) <= 5

    @pytest.mark.asyncio
    async def test_extract_invalid_category_fallback(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """无效 category 应回退为 'workflow'。"""
        json_with_invalid_cat = """{
          "strategies": [
            {
              "title": "Invalid cat",
              "condition": "when X",
              "action": "do Y",
              "rationale": "because Z",
              "category": "invalid_category",
              "confidence": 0.8
            }
          ]
        }"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(json_with_invalid_cat),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) == 1
        assert strategies[0].category == "workflow"


# ── 去重检测测试 ──────────────────────────────────────────────


class TestCheckDuplicate:
    """_check_duplicate 去重检测测试。"""

    def test_check_duplicate_identical_title(self) -> None:
        title = "添加 FastAPI 路由时同步更新测试"
        strategy = Strategy(
            strategy_id="new-1",
            title=title,
            condition="when X",
            action="do Y",
            rationale="because Z",
            category="workflow",
            source_task_ids=["t1"],
        )
        existing = [
            Strategy(
                strategy_id="old-1",
                title=title,
                condition="when A",
                action="do B",
                rationale="because C",
                category="workflow",
                source_task_ids=["t0"],
            ),
        ]
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, existing)
        assert is_dup is True
        assert dup_id == "old-1"

    def test_check_duplicate_similar_content(self) -> None:
        strategy = Strategy(
            strategy_id="new-2",
            title="Fix db migration",
            condition="当修改 SQLAlchemy 模型定义时",
            action="应生成对应的迁移脚本",
            rationale="because Z",
            category="workflow",
            source_task_ids=["t1"],
        )
        existing = [
            Strategy(
                strategy_id="old-2",
                title="DB migration best practice",
                condition="当修改 SQLAlchemy 模型定义时",
                action="应生成对应的迁移脚本并执行",
                rationale="different rationale",
                category="workflow",
                source_task_ids=["t0"],
            ),
        ]
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, existing)
        assert is_dup is True
        assert dup_id == "old-2"

    def test_check_duplicate_no_duplicate(self) -> None:
        strategy = Strategy(
            strategy_id="new-3",
            title="Frontend testing",
            condition="当修改 React 组件时",
            action="应更新对应的测试文件",
            rationale="because",
            category="workflow",
            source_task_ids=["t1"],
        )
        existing = [
            Strategy(
                strategy_id="old-3",
                title="修改数据库模型时执行迁移",
                condition="当修改 SQLAlchemy 模型时",
                action="应生成迁移脚本",
                rationale="different",
                category="coding_style",
                source_task_ids=["t0"],
            ),
        ]
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, existing)
        assert is_dup is False
        assert dup_id is None

    def test_check_duplicate_empty_existing(self) -> None:
        strategy = Strategy(
            strategy_id="new-4",
            title="Test",
            condition="when X",
            action="do Y",
            rationale="because Z",
            category="workflow",
            source_task_ids=["t1"],
        )
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, [])
        assert is_dup is False
        assert dup_id is None

    def test_check_duplicate_same_category_high_action_similarity(self) -> None:
        """同 category 且 action 高度相似应视为重复。"""
        strategy = Strategy(
            strategy_id="new-5",
            title="Testing routes",
            condition="当添加路由时",
            action="应在 tests/ 目录添加对应的测试文件覆盖正常和异常路径",
            rationale="because",
            category="workflow",
            source_task_ids=["t1"],
        )
        existing = [
            Strategy(
                strategy_id="old-5",
                title="Adding route tests",
                condition="某情況下",
                action="应在 tests/ 目录添加对应的测试文件覆盖正常和异常路径",
                rationale="different",
                category="workflow",
                source_task_ids=["t0"],
            ),
        ]
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, existing)
        assert is_dup is True
        assert dup_id == "old-5"

    def test_check_duplicate_different_category_low_similarity(self) -> None:
        """不同 category 且内容不同不应视为重复。"""
        strategy = Strategy(
            strategy_id="new-6",
            title="Error handling",
            condition="当捕获异常时使用具体类型",
            action="使用具体的异常类型而非裸 except",
            rationale="better debugging",
            category="coding_style",
            source_task_ids=["t1"],
        )
        existing = [
            Strategy(
                strategy_id="old-6",
                title="Docker optimization",
                condition="当编写 Dockerfile 时",
                action="使用多阶段构建减小镜像体积",
                rationale="smaller images",
                category="tool_usage",
                source_task_ids=["t0"],
            ),
        ]
        extractor = StrategyExtractor(
            llm_client=None,  # type: ignore[arg-type]
            recorder=None,  # type: ignore[arg-type]
            store=MockStore(),
        )
        is_dup, dup_id = extractor._check_duplicate(strategy, existing)
        assert is_dup is False
        assert dup_id is None


# ── extract_from_recent 测试 ──────────────────────────────────


class TestExtractFromRecent:
    """extract_from_recent 方法测试。"""

    @pytest.mark.asyncio
    async def test_extract_from_recent_calls_load_recent(
        self, tmp_path: Path,
    ) -> None:
        """extract_from_recent 应调用 recorder.load_recent()。"""
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajs"))

        # 添加 3 条轨迹
        for i in range(3):
            tid = f"recent-task-{i}"
            recorder.start_task(tid, f"Task {i}")
            recorder.complete_task(tid, "success")

        store = MockStore()
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory('{"strategies": []}'),
            recorder=recorder,
            store=store,
        )
        strategies = await extractor.extract_from_recent()
        # No strategies expected since LLM returns empty
        assert strategies == []

    @pytest.mark.asyncio
    async def test_extract_from_recent_with_data(
        self, tmp_path: Path,
    ) -> None:
        """从最近轨迹中提炼策略。"""
        recorder = TrajectoryRecorder(base_path=str(tmp_path / "trajs"))

        for i in range(3):
            tid = f"task-{i}"
            recorder.start_task(tid, f"Task {i}: fix function")
            step = TrajectoryStep(
                step_id=f"s{i}",
                node_name="execution",
                step_type="tool_call",
                timestamp=datetime.now(),
                input_summary="in",
                output_summary="out",
            )
            recorder.record_step(tid, step)
            recorder.complete_task(tid, "success")

        store = MockStore()
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(_VALID_STRATEGY_JSON),
            recorder=recorder,
            store=store,
        )
        strategies = await extractor.extract_from_recent()
        assert len(strategies) == 2
        assert all(s.title for s in strategies)

    @pytest.mark.asyncio
    async def test_extract_from_recent_recorder_error(
        self, mock_store: MockStore,
    ) -> None:
        """recorder.load_recent 抛出异常时应返回空列表。"""

        class BrokenRecorder:
            def load_recent(self, n: int = 20, success_only: bool = False) -> list:
                raise RuntimeError("Disk error")

        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory('{"strategies": []}'),
            recorder=BrokenRecorder(),  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract_from_recent()
        assert strategies == []


# ── 关键词匹配测试 ────────────────────────────────────────────


class TestKeywordMatching:
    """关键词匹配辅助方法测试。"""

    def test_tokenize(self) -> None:
        result = StrategyExtractor._tokenize("Hello World! 测试123")
        assert "hello" in result
        assert "world" in result
        assert "测试123" in result

    def test_tokenize_empty(self) -> None:
        result = StrategyExtractor._tokenize("")
        assert result == set()

    def test_tokenize_case_insensitive(self) -> None:
        result = StrategyExtractor._tokenize("Hello HELLO hello")
        assert len(result) == 1
        assert "hello" in result

    def test_keyword_score_full_match(self) -> None:
        query = {"fastapi", "route", "test"}
        score = StrategyExtractor._keyword_score(query, "add fastapi route and test")
        assert score == 1.0

    def test_keyword_score_partial_match(self) -> None:
        query = {"fastapi", "route", "test", "migration"}
        score = StrategyExtractor._keyword_score(query, "add fastapi route with test")
        assert score == 0.75

    def test_keyword_score_no_match(self) -> None:
        query = {"fastapi", "route"}
        score = StrategyExtractor._keyword_score(query, "javascript react component")
        assert score == 0.0

    def test_keyword_score_empty_query(self) -> None:
        score = StrategyExtractor._keyword_score(set(), "some text")
        assert score == 0.0


# ── JSON 清理测试 ────────────────────────────────────────────


class TestCleanJson:
    """_clean_json 测试。"""

    def test_clean_json_markdown_block(self) -> None:
        raw = "```json\n{\"strategies\": []}\n```"
        result = StrategyExtractor._clean_json(raw)
        assert result == '{"strategies": []}'

    def test_clean_json_direct_json(self) -> None:
        raw = '{"strategies": []}'
        result = StrategyExtractor._clean_json(raw)
        assert result == raw

    def test_clean_json_with_prefix_text(self) -> None:
        raw = 'Some prefix text\n{"strategies": []}'
        result = StrategyExtractor._clean_json(raw)
        # 当 content 中包含 ``` 时走代码块逻辑，否则直接返回
        assert result == raw

    def test_clean_json_with_triple_backtick_no_json(self) -> None:
        raw = "```\n{\"strategies\": []}\n```"
        result = StrategyExtractor._clean_json(raw)
        assert result == '{"strategies": []}'

    def test_clean_json_array_format(self) -> None:
        """LLM 可能直接返回数组格式。"""
        raw = '[{"title": "test", "condition": "c", "action": "a", "rationale": "r", "category": "workflow", "confidence": 0.8}]'
        result = StrategyExtractor._clean_json(raw)
        assert result == raw


# ── 边界情况和错误处理测试 ────────────────────────────────────


class TestEdgeCases:
    """边界情况和错误处理测试。"""

    @pytest.mark.asyncio
    async def test_extract_with_non_dict_items(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """策略列表中的非字典项应被跳过。"""
        json_with_invalid = """{
          "strategies": [
            {"title": "Valid", "condition": "c", "action": "a", "rationale": "r", "category": "workflow", "confidence": 0.8},
            "not a dict",
            null,
            42
          ]
        }"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(json_with_invalid),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) == 1

    @pytest.mark.asyncio
    async def test_extract_with_missing_fields(
        self, mock_store: MockStore, sample_trajectories: list[Trajectory],
    ) -> None:
        """缺少字段的项应使用默认值。"""
        json_missing_fields = """{
          "strategies": [
            {"title": "Minimal", "condition": "c", "action": "a"}
          ]
        }"""
        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(json_missing_fields),
            recorder=None,  # type: ignore[arg-type]
            store=mock_store,
        )
        strategies = await extractor.extract(sample_trajectories)
        assert len(strategies) == 1
        assert strategies[0].rationale == ""
        assert strategies[0].category == "workflow"
        assert strategies[0].confidence == 0.5

    @pytest.mark.asyncio
    async def test_store_unavailable(
        self, sample_trajectories: list[Trajectory],
    ) -> None:
        """store.list_active 抛出异常时不应影响 extract 主流程。"""

        class BrokenStore:
            def list_active(self) -> list:
                raise RuntimeError("Store unavailable")

        extractor = StrategyExtractor(
            llm_client=await mock_llm_factory(_VALID_STRATEGY_JSON),
            recorder=None,  # type: ignore[arg-type]
            store=BrokenStore(),  # type: ignore[arg-type]
        )
        strategies = await extractor.extract(sample_trajectories)
        # Should still return strategies despite store error
        assert len(strategies) == 2

    @pytest.mark.asyncio
    async def test_trajectories_to_summary_empty(
        self, empty_extractor: StrategyExtractor,
    ) -> None:
        """空轨迹列表的摘要应为空字符串。"""
        summary = empty_extractor._trajectories_to_summary([])
        assert summary == ""

    @pytest.mark.asyncio
    async def test_trajectories_to_summary_with_data(
        self, empty_extractor: StrategyExtractor,
        sample_trajectories: list[Trajectory],
    ) -> None:
        """轨迹摘要应包含关键信息。"""
        summary = empty_extractor._trajectories_to_summary(sample_trajectories)
        assert "Trajectory 1" in summary
        assert "Task:" in summary
        assert "Steps:" in summary
        assert "Status:" in summary
        assert "Repairs:" in summary
        assert "Validated:" in summary
        assert "Request:" in summary
