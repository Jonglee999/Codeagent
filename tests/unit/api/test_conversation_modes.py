from codeagent.interaction.api.main import _classify_response_mode
from codeagent.orchestration.routing import route_after_context
from codeagent.orchestration.state import AgentState
from codeagent.interaction.api.websocket import _is_user_visible_event


def test_questions_use_chat_without_planning() -> None:
    assert _classify_response_mode("刚才修改了哪些文件？", "auto") == "chat"


def test_code_follow_up_uses_execution() -> None:
    assert _classify_response_mode("再修复一下失败的测试", "auto") == "execute"


def test_frontend_creation_request_uses_execution() -> None:
    assert _classify_response_mode("帮我写一个简单的前端页面看看效果", "auto") == "execute"


def test_write_file_request_uses_execution() -> None:
    assert _classify_response_mode("写一个 index.html 并让我预览", "auto") == "execute"


def test_explicit_mode_overrides_classifier() -> None:
    assert _classify_response_mode("请解释这个错误", "execute") == "execute"


def test_negated_file_action_stays_in_chat() -> None:
    assert _classify_response_mode("请直接回答，不要修改文件。", "auto") == "chat"


def test_follow_up_bypasses_planning_node() -> None:
    state = AgentState(user_request="继续修复", project_root=".", direct_execution=True)
    assert route_after_context(state) == "execution"


def test_polite_question_that_requests_execution_uses_tools() -> None:
    assert _classify_response_mode("能帮我再运行一下吗？", "auto") == "execute"


def test_question_about_a_failure_stays_tool_free() -> None:
    assert _classify_response_mode("这个测试为什么失败？", "auto") == "chat"


def test_english_polite_execution_request_uses_tools() -> None:
    assert _classify_response_mode("Could you run the parser again?", "auto") == "execute"


def test_question_about_helping_developers_stays_in_chat() -> None:
    assert _classify_response_mode("请用一句话说明你如何帮助开发者。", "auto") == "chat"


def test_request_to_develop_a_page_still_uses_tools() -> None:
    assert _classify_response_mode("请帮我开发一个页面。", "auto") == "execute"


def test_internal_memory_events_never_cross_websocket_boundary() -> None:
    assert not _is_user_visible_event({"type": "memory_recalled"})
    assert not _is_user_visible_event({"type": "custom", "visibility": "internal"})
    assert _is_user_visible_event({"type": "tool_result"})
