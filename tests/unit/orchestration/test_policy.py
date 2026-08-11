from codeagent.orchestration.policy import select_run_profile


def test_focused_request_uses_direct_just_in_time_tools() -> None:
    profile = select_run_profile("修复登录按钮点击后没有响应的问题")

    assert profile.workflow == "direct"
    assert profile.context == "minimal"
    assert profile.memory == "off"
    assert profile.reflection == "on_failure"


def test_complex_request_keeps_reviewable_plan_and_full_context() -> None:
    profile = select_run_profile("重构整个认证架构，同时迁移数据库并更新前端和 API")

    assert profile.workflow == "planned"
    assert profile.context == "full"
    assert profile.memory == "relevant"


def test_follow_up_reuses_conversation_without_replanning() -> None:
    profile = select_run_profile(
        "继续修复刚才失败的测试",
        conversation_history=[{"role": "user", "content": "修复解析器"}],
        direct_execution=True,
    )

    assert profile.workflow == "direct"
    assert profile.context == "minimal"
    assert profile.memory == "off"


def test_follow_up_history_does_not_trigger_durable_memory_recall() -> None:
    profile = select_run_profile(
        "再运行一次并返回新结果",
        conversation_history=[
            {"role": "user", "content": "写一个随机数脚本"},
            {"role": "assistant", "content": "已经创建 random_generator.py"},
        ],
        direct_execution=True,
    )

    assert profile.memory == "off"
    assert profile.learning == "off"


def test_only_explicit_durable_preferences_enable_learning() -> None:
    profile = select_run_profile("记住：以后这个项目始终使用 pytest")

    assert profile.learning == "capture"
    assert profile.memory == "relevant"


def test_benchmark_example_uses_direct_controller_without_memory_contamination() -> None:
    profile = select_run_profile("fix the failing parser", benchmark_instance_id="repo__issue-1")

    assert profile.workflow == "direct"
    assert profile.context == "minimal"
    assert profile.memory == "off"
    assert profile.benchmark is True
