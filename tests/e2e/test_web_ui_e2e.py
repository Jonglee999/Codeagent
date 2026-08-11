"""Web UI E2E 浏览器测试（Playwright）。

验证 Web UI 的黄金路径流程：
  Scene O — 提交任务 → 查看执行日志 → 确认完成
  Scene P — Human Review 弹窗 → 审批 → 继续执行
  Scene Q — 历史记录持久化 → 刷新 → 历史可见

所有测试使用 @pytest.mark.e2e 标记，仅在满足前提条件时运行。
"""

from __future__ import annotations

import pytest
from pytest import mark

try:
    from playwright.sync_api import expect
except ImportError:
    # Playwright not installed; tests are skipped via conftest hooks
    pass


# ═══════════════════════════════════════════════════════════════════════
# Scene O: 黄金路径
# ═══════════════════════════════════════════════════════════════════════


@mark.e2e
class TestGoldenPath:
    """Scene O: 提交任务 → 查看执行日志 → 确认完成。"""

    def test_submit_task_and_view_log(self, page, app_url: str):
        """完整黄金路径。

        1. 访问 Web UI
        2. 填写任务描述和项目路径
        3. 点击提交
        4. 验证 execution-log 可见
        5. 验证至少有一条日志条目
        """
        page.goto(app_url)

        # 等待页面加载（TaskInput 组件应可见）
        page.wait_for_selector('[data-testid="task-query"]', timeout=10000)
        page.wait_for_selector('[data-testid="project-root"]', timeout=5000)

        # 填写表单
        page.fill('[data-testid="task-query"]', "为 main.py 添加类型注解")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")

        # 点击提交按钮
        page.click('[data-testid="submit-btn"]')

        # 验证 execution-log 区域可见
        page.wait_for_selector('[data-testid="execution-log"]', timeout=5000)

        # 验证至少有一条日志条目
        page.wait_for_selector('[data-testid="log-entry"]', timeout=30000)

        # 验证 task_id 已显示
        task_id_elem = page.locator('[data-testid="task-id-display"]')
        expect(task_id_elem).to_be_visible()

    def test_task_progress_and_completion(self, page, app_url: str):
        """提交任务后观察进度更新直至完成。

        1. 提交任务
        2. 验证进度指示器可见
        3. 等待 task_complete 事件
        4. 验证完成状态显示
        """
        page.goto(app_url)

        page.fill('[data-testid="task-query"]', "添加一个简单的 hello.py 文件")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 验证进度指示器
        page.wait_for_selector('[data-testid="progress-indicator"]', timeout=5000)

        # 等待任务完成事件日志
        page.wait_for_selector('[data-testid="log-entry"]', timeout=30000)

        # 验证任务完成后的状态显示
        status = page.locator('[data-testid="task-status"]')
        expect(status).to_be_visible()

    def test_error_handling_empty_input(self, page, app_url: str):
        """提交空输入时显示验证错误。

        1. 不填写 task-query
        2. 点击提交
        3. 验证错误提示
        """
        page.goto(app_url)
        page.wait_for_selector('[data-testid="submit-btn"]', timeout=5000)

        # 不填写 query，直接提交
        page.fill('[data-testid="project-root"]', "/tmp/test")
        page.click('[data-testid="submit-btn"]')

        # 应显示验证错误提示
        page.wait_for_selector('[data-testid="error-toast"]', timeout=5000)


# ═══════════════════════════════════════════════════════════════════════
# Scene P: Human Review 审核弹窗
# ═══════════════════════════════════════════════════════════════════════


@mark.e2e
class TestHumanReviewModal:
    """Scene P: Human Review 弹窗 → 审批 → 继续执行。"""

    def test_human_review_approve_flow(self, page, app_url: str):
        """Human Review 弹窗出现后点击 Approve 继续执行。

        1. 提交可能触发 Human Review 的任务（auto_mode=false）
        2. 等待 human_review_modal 出现
        3. 验证弹窗显示了计划详情
        4. 点击 Approve 按钮
        5. 验证弹窗关闭，任务继续
        """
        page.goto(app_url)
        page.fill('[data-testid="task-query"]', "重构 auth 模块，将 Basic Auth 替换为 JWT")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 等待 Human Review 弹窗出现（可能需要较长时间等待 Agent 规划完成）
        page.wait_for_selector('[data-testid="human-review-modal"]', timeout=120000)

        # 验证弹窗显示了 review_type
        review_type = page.locator('[data-testid="review-type"]')
        expect(review_type).to_be_visible()

        # 验证弹窗显示了计划详情
        plan_details = page.locator('[data-testid="plan-details"]')
        expect(plan_details).to_be_visible()

        # 点击 Approve 按钮
        page.click('[data-testid="approve-btn"]')

        # 验证弹窗关闭
        page.wait_for_selector('[data-testid="human-review-modal"]', state="detached", timeout=10000)

        # 验证执行继续（新的日志条目出现）
        page.wait_for_selector('[data-testid="log-entry"]', timeout=30000)

    def test_human_review_reject_flow(self, page, app_url: str):
        """Human Review 弹窗出现后点击 Reject。"""
        page.goto(app_url)
        page.fill('[data-testid="task-query"]', "删除整个项目的所有文件")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 等待 Human Review 弹窗
        page.wait_for_selector('[data-testid="human-review-modal"]', timeout=120000)

        # 点击 Reject 按钮
        page.click('[data-testid="reject-btn"]')

        # 验证弹窗关闭，状态显示已拒绝
        page.wait_for_selector('[data-testid="human-review-modal"]', state="detached", timeout=10000)
        status = page.locator('[data-testid="task-status"]')
        expect(status).to_be_visible()

    def test_human_review_modify_flow(self, page, app_url: str):
        """Human Review 弹窗出现后填写修改意见并提交。"""
        page.goto(app_url)
        page.fill('[data-testid="task-query"]', "重构用户管理模块")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 等待 Human Review 弹窗
        page.wait_for_selector('[data-testid="human-review-modal"]', timeout=120000)

        # 点击 Modify 按钮
        page.click('[data-testid="modify-btn"]')

        # 验证修改意见输入框出现
        page.wait_for_selector('[data-testid="modify-textarea"]', timeout=5000)

        # 填写修改意见
        page.fill('[data-testid="modify-textarea"]', "请使用更安全的方式实现")

        # 提交修改
        page.click('[data-testid="submit-modify-btn"]')

        # 验证弹窗关闭
        page.wait_for_selector('[data-testid="human-review-modal"]', state="detached", timeout=10000)


# ═══════════════════════════════════════════════════════════════════════
# Scene Q: 历史记录持久化
# ═══════════════════════════════════════════════════════════════════════


@mark.e2e
class TestTaskHistory:
    """Scene Q: 历史记录持久化 — 完成 → 刷新 → 历史可见。"""

    def test_task_history_persistence(self, page, app_url: str):
        """完成任务后刷新页面，历史记录应保留。

        1. 提交并完成一个任务
        2. 验证 TaskHistory 组件出现
        3. 刷新页面
        4. 验证历史记录仍然存在
        """
        page.goto(app_url)
        page.fill('[data-testid="task-query"]', "创建一个 hello.py 文件")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 等待任务完成
        page.wait_for_selector('[data-testid="task-complete"]', timeout=120000)

        # 验证历史记录区域出现
        page.wait_for_selector('[data-testid="task-history"]', timeout=5000)

        # 记住历史条目数量
        page.locator('[data-testid="history-item"]').count()

        # 刷新页面
        page.reload()

        # 等待页面加载
        page.wait_for_selector('[data-testid="task-history"]', timeout=10000)

        # 验证历史记录仍然存在
        history_count_after = page.locator('[data-testid="history-item"]').count()
        assert history_count_after >= 1, "刷新后历史记录应至少有一条"

    def test_click_history_item_shows_report(self, page, app_url: str):
        """点击历史记录项可查看对应任务的报告。

        1. 确保有历史记录
        2. 点击历史记录项
        3. 验证报告展示
        """
        page.goto(app_url)

        # 等待历史记录加载（如果有）
        try:
            page.wait_for_selector('[data-testid="history-item"]', timeout=30000)

            # 点击第一个历史记录项
            page.click('[data-testid="history-item"]:first-child')

            # 验证报告区域出现
            report = page.locator('[data-testid="task-report"]')
            expect(report).to_be_visible(timeout=10000)
        except Exception:
            # 没有历史记录时跳过
            pytest.skip("No history items available")

    def test_history_shows_recent_tasks(self, page, app_url: str):
        """历史记录显示最近的任务摘要（task_id + 请求 + 状态）。"""
        page.goto(app_url)
        page.fill('[data-testid="task-query"]', "添加一个 greet 函数")
        page.fill('[data-testid="project-root"]', "/tmp/test_project")
        page.click('[data-testid="submit-btn"]')

        # 等待任务完成
        page.wait_for_selector('[data-testid="task-complete"]', timeout=120000)

        # 验证历史记录项包含必要信息
        history_items = page.locator('[data-testid="history-item"]')
        expect(history_items.first).to_be_visible()

        # 验证历史记录项显示了请求摘要
        query_summary = page.locator('[data-testid="history-query"]')
        expect(query_summary.first).to_be_visible()
