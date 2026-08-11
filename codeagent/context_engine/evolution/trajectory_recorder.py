"""TrajectoryRecorder — 执行轨迹记录器。

在任务执行过程中记录完整的决策和执行轨迹，
作为后续 StrategyExtractor 分析的原始数据。

存储格式：每个任务一个 JSONL 文件
目录结构：.codeagent/trajectories/{YYYY-MM}/{task_id}.jsonl
性能要求：单步记录 < 5ms（使用异步写入 + 仅记录摘要）
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# 缓冲区刷盘阈值（步数）
_FLUSH_INTERVAL = 10


@dataclass
class TrajectoryStep:
    """单个执行轨迹步骤。

    Attributes:
        step_id: 唯一 ID（UUID）
        node_name: LangGraph 节点名（context/planning/execution/validation）
        step_type: 步骤类型
        timestamp: 时间戳
        input_summary: 节点输入摘要（≤200 字符）
        output_summary: 节点输出摘要（≤200 字符）
        tool_calls: 工具调用列表 [{tool, params_summary, result_summary}]
        duration_ms: 执行耗时（毫秒）
        success: 是否成功
        error: 错误信息（可选）
    """

    step_id: str
    node_name: str
    step_type: Literal["llm_call", "tool_call", "human_review", "validation"]
    timestamp: datetime
    input_summary: str
    output_summary: str
    tool_calls: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class Trajectory:
    """完整任务执行轨迹。

    Attributes:
        task_id: 唯一任务 ID（UUID）
        user_request: 用户请求
        started_at: 开始时间
        completed_at: 完成时间（可选）
        steps: 执行步骤列表
        final_status: 最终状态
        repair_rounds: 经历了几轮修复
        validation_passed: 验证是否通过
    """

    task_id: str
    user_request: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_status: Optional[Literal["success", "failure", "aborted"]] = None
    repair_rounds: int = 0
    validation_passed: bool = False


class TrajectoryRecorder:
    """轨迹记录器——在任务执行过程中记录完整的决策和执行轨迹。

    存储格式：每个任务一个 JSONL 文件
    目录结构：.codeagent/trajectories/{YYYY-MM}/{task_id}.jsonl
    性能要求：单步记录 < 5ms（使用内存缓冲区 + 异步刷盘）
    """

    def __init__(self, base_path: str = ".codeagent/trajectories") -> None:
        """初始化 TrajectoryRecorder。

        Args:
            base_path: 轨迹存储根目录
        """
        self._base_path = Path(base_path)
        self._trajectories: dict[str, Trajectory] = {}
        self._buffers: dict[str, list[str]] = {}
        self._flush_tasks: set[asyncio.Task] = set()

    # ── 公开接口 ─────────────────────────────────────────────────

    def start_task(self, task_id: str, user_request: str) -> None:
        """任务开始时调用。

        创建 Trajectory 实例，写入开始时间戳。
        自动创建存储目录。

        Args:
            task_id: 任务唯一 ID
            user_request: 用户请求描述
        """
        trajectory = Trajectory(
            task_id=task_id,
            user_request=user_request,
            started_at=datetime.now(),
        )
        self._trajectories[task_id] = trajectory
        self._buffers[task_id] = []

        # 确保目录存在
        date_dir = self._date_dir()
        date_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Task %s trajectory recording started", task_id)

    def record_step(self, task_id: str, step: TrajectoryStep) -> None:
        """记录一个执行步骤。

        追加写入 JSONL 文件。
        必须 < 5ms（同步写内存缓存 + 异步刷盘）。

        Args:
            task_id: 任务唯一 ID
            step: 轨迹步骤
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            logger.warning("No active trajectory for task %s, skipping step", task_id)
            return

        trajectory.steps.append(step)

        # 序列化为 JSON 行
        step_dict = asdict(step)
        step_dict["timestamp"] = step.timestamp.isoformat()
        line = json.dumps(step_dict, ensure_ascii=False, default=str)

        self._buffers.setdefault(task_id, []).append(line)

        # 达到阈值时触发异步刷盘
        if len(self._buffers[task_id]) > _FLUSH_INTERVAL:
            self._flush_async(task_id)

    def complete_task(
        self,
        task_id: str,
        status: Literal["success", "failure", "aborted"],
        repair_rounds: int = 0,
        validation_passed: bool = False,
    ) -> None:
        """任务结束时调用。

        标记完成状态、修复轮数、验证结果。
        刷新缓存到磁盘。

        Args:
            task_id: 任务唯一 ID
            status: 最终状态
            repair_rounds: 修复轮数
            validation_passed: 验证是否通过
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            logger.warning("No active trajectory for task %s, skipping complete", task_id)
            return

        trajectory.completed_at = datetime.now()
        trajectory.final_status = status
        trajectory.repair_rounds = repair_rounds
        trajectory.validation_passed = validation_passed

        # 强制刷盘
        self._flush_sync(task_id)
        # 确保 JSONL 文件存在（即使无步骤，也用于 load_recent 发现该任务）
        jsonl_path = self._file_path(task_id)
        if not jsonl_path.exists():
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.touch()
        self._write_metadata(task_id)

        logger.debug(
            "Task %s trajectory completed: status=%s, steps=%d, repairs=%d",
            task_id, status, len(trajectory.steps), repair_rounds,
        )

    def load_recent(self, n: int = 20, success_only: bool = False) -> list[Trajectory]:
        """加载最近 N 条轨迹（按日期降序）。

        Args:
            n: 最大返回条数
            success_only: 只加载成功轨迹

        Returns:
            轨迹列表（按最后修改时间降序）
        """
        if not self._base_path.exists():
            return []

        # 收集所有 JSONL 文件
        jsonl_files: list[tuple[Path, float]] = []
        for date_dir in sorted(self._base_path.iterdir()):
            if not date_dir.is_dir():
                continue
            for f in date_dir.iterdir():
                if f.suffix == ".jsonl":
                    mtime = f.stat().st_mtime
                    jsonl_files.append((f, mtime))

        # 按修改时间降序排列
        jsonl_files.sort(key=lambda x: x[1], reverse=True)

        # 加载前 N 个
        results: list[Trajectory] = []
        for file_path, _ in jsonl_files[:n]:
            try:
                trajectory = self._load_trajectory_from_file(file_path)
                if trajectory is not None:
                    if success_only and trajectory.final_status != "success":
                        continue
                    results.append(trajectory)
            except Exception as exc:
                logger.warning("Failed to load trajectory %s: %s", file_path, exc)

        return results

    def get_trajectory(self, task_id: str) -> Optional[Trajectory]:
        """按 task_id 加载单条轨迹。

        Args:
            task_id: 任务唯一 ID

        Returns:
            轨迹对象，不存在时返回 None
        """
        # 先查内存
        cached = self._trajectories.get(task_id)
        if cached is not None:
            return cached

        # 在所有日期目录中查找
        if not self._base_path.exists():
            return None

        for date_dir in self._base_path.iterdir():
            if not date_dir.is_dir():
                continue
            file_path = date_dir / f"{task_id}.jsonl"
            if file_path.exists():
                try:
                    return self._load_trajectory_from_file(file_path)
                except Exception as exc:
                    logger.warning("Failed to load trajectory %s: %s", file_path, exc)
                    return None

        return None

    # ── 内部方法 ─────────────────────────────────────────────────

    def _date_dir(self) -> Path:
        """获取当前日期的子目录路径。"""
        now = datetime.now()
        return self._base_path / now.strftime("%Y-%m")

    def _file_path(self, task_id: str) -> Path:
        """获取任务对应的文件路径。"""
        return self._date_dir() / f"{task_id}.jsonl"

    def _flush_async(self, task_id: str) -> None:
        """异步刷盘（不阻塞主流程）。

        如果没有运行中的事件循环，降级为同步刷盘。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，降级为同步刷盘
            self._flush_sync(task_id)
            return

        task = loop.create_task(self._do_flush(task_id))
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_tasks.discard)

    def _flush_sync(self, task_id: str) -> None:
        """同步刷盘。"""
        lines = self._buffers.pop(task_id, [])
        if not lines:
            return

        file_path = self._file_path(task_id)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except OSError as exc:
            logger.error("Failed to flush trajectory %s: %s", task_id, exc)

    async def _do_flush(self, task_id: str) -> None:
        """异步刷盘实现。"""
        self._flush_sync(task_id)

    def _write_metadata(self, task_id: str) -> None:
        """写入轨迹元数据到单独的 JSON 文件。

        用于持久化 final_status、repair_rounds 等信息，
        使 load_recent 能从文件加载过滤条件。

        Args:
            task_id: 任务唯一 ID
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            return

        meta = {
            "task_id": trajectory.task_id,
            "user_request": trajectory.user_request,
            "started_at": trajectory.started_at.isoformat(),
            "completed_at": trajectory.completed_at.isoformat() if trajectory.completed_at else None,
            "final_status": trajectory.final_status,
            "repair_rounds": trajectory.repair_rounds,
            "validation_passed": trajectory.validation_passed,
            "step_count": len(trajectory.steps),
        }

        file_path = self._metadata_path(task_id)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, default=str)
        except OSError as exc:
            logger.error("Failed to write metadata %s: %s", task_id, exc)

    def _metadata_path(self, task_id: str) -> Path:
        """获取任务元数据文件路径。"""
        return self._date_dir() / f"{task_id}.meta.json"

    def _load_metadata(self, task_id: str, file_path: Path) -> Optional[dict]:
        """从元数据文件加载轨迹元数据。

        Args:
            task_id: 任务 ID（用于查内存缓存）
            file_path: JSONL 文件路径（用于推断元数据文件路径）

        Returns:
            元数据字典，或 None
        """
        meta_path = file_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load metadata for %s: %s", task_id, exc)
            return None

    def _load_trajectory_from_file(self, file_path: Path) -> Optional[Trajectory]:
        """从 JSONL 文件加载完整轨迹。

        Args:
            file_path: JSONL 文件路径

        Returns:
            反序列化的 Trajectory 对象
        """
        task_id = file_path.stem

        # 先查内存中的元数据
        cached = self._trajectories.get(task_id)

        steps: list[TrajectoryStep] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                steps.append(TrajectoryStep(**data))

        # 如果有内存缓存，使用其元数据
        if cached is not None:
            return Trajectory(
                task_id=cached.task_id,
                user_request=cached.user_request,
                started_at=cached.started_at,
                completed_at=cached.completed_at,
                steps=steps,
                final_status=cached.final_status,
                repair_rounds=cached.repair_rounds,
                validation_passed=cached.validation_passed,
            )

        # 尝试从元数据文件加载
        meta = self._load_metadata(task_id, file_path)
        if meta is not None:
            started_at = datetime.fromisoformat(meta["started_at"]) if meta.get("started_at") else datetime.now()
            completed_at = datetime.fromisoformat(meta["completed_at"]) if meta.get("completed_at") else None
            return Trajectory(
                task_id=meta.get("task_id", task_id),
                user_request=meta.get("user_request", ""),
                started_at=started_at,
                completed_at=completed_at,
                steps=steps,
                final_status=meta.get("final_status"),
                repair_rounds=meta.get("repair_rounds", 0),
                validation_passed=meta.get("validation_passed", False),
            )

        # 降级：仅返回步骤数据，无元数据
        return Trajectory(
            task_id=task_id,
            user_request="",
            started_at=steps[0].timestamp if steps else datetime.now(),
            steps=steps,
        )
