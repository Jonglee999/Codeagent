"""命令安全检查 — Layer 1 安全防护。

在命令到达 Docker 容器之前进行静态分析，拦截高风险操作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

RiskLevel = Literal["safe", "suspicious", "dangerous"]


@dataclass
class SafetyResult:
    """命令安全检查结果。"""

    safe: bool
    reason: str = ""
    risk_level: RiskLevel = "safe"
    matched_patterns: list[str] = field(default_factory=list)


class SafetyChecker:
    """命令安全检查器。

    三层静态检查：
    1. 黑名单命令模式（禁止直接使用危险命令）
    2. Shell 注入模式（检测反引号、$()、链式执行等）
    3. 文件系统危险操作（rm -rf 等破坏性操作）
    """

    # ── 命令黑名单 ────────────────────────────────────────────
    # 这些命令作为独立命令或管道第一个命令时被禁止
    BLACKLISTED_COMMANDS: set[str] = {
        "sudo",
        "su",
        "chmod",
        "chown",
        "mkfs",
        "mkfs.ext4",
        "mkfs.xfs",
        "dd",
        "reboot",
        "shutdown",
        "halt",
        "poweroff",
        "init",
        "systemctl",
        "service",
        "kill",
        "pkill",
        "iptables",
        "ufw",
        "passwd",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "fdisk",
        "parted",
        "mount",
        "umount",
        "insmod",
        "rmmod",
        "modprobe",
    }

    # ── 危险模式（正则） ───────────────────────────────────────
    DANGEROUS_PATTERNS: list[re.Pattern] = [
        # 递归强制删除根目录
        re.compile(r"\brm\s+(?:-[rfRF]+|-rf|-fr)\s+/\b"),
        re.compile(r"\brm\s+(?:-[rfRF]+|-rf|-fr)\s+/[*?]"),
        # chmod 777 / 7777 等宽松权限
        re.compile(r"\bchmod\s+[0-7]{3,4}\s"),
        # 写入 /etc/ 系统目录
        re.compile(r"[>|].*/\s*etc/"),
        # fork 炸弹
        re.compile(r":\(\)\s*\{"),
        re.compile(r":\(\s*\)\s*\{"),
        # dd 直接操作块设备
        re.compile(r"\bdd\s+if="),
        # 修改系统关键文件
        re.compile(r"/etc/(passwd|shadow|sudoers|fstab|hosts|resolv\.conf)\b"),
    ]

    # ── Shell 注入模式 ────────────────────────────────────────
    SHELL_INJECTION_PATTERNS: list[re.Pattern] = [
        # 反引号命令执行
        re.compile(r"`[^`]+`"),
        # $() 命令替换
        re.compile(r"\$\([^)]+\)"),
        # 管道到危险命令
        re.compile(r"\|\s*(sudo|su|chmod|chown|dd|sh|bash)(?:\s|$)"),
        # 多命令执行（&& 和 ||）
        re.compile(r"\s+&&\s+(sudo|su|chmod|chown|dd|rm|mkfs|reboot|shutdown)(?:\s|$)"),
        # 后台执行后删除
        re.compile(r"\s*;\s*rm\s+"),
    ]

    # ── 文件系统危险操作 ──────────────────────────────────────
    FILESYSTEM_DANGER_PATTERNS: list[re.Pattern] = [
        # 递归删除
        re.compile(r"\brm\s+-(?:rf|[rR]+[fF]+|[fF]+[rR]+)\b"),
        # 强制覆盖写入系统文件
        re.compile(r"cat\s+.*[>|]{2,}\s*/"),
        # 清空或覆写磁盘
        re.compile(r"\bdd\s+"),
        # 移动/重写系统目录
        re.compile(r"\bmv\s+.*\s+/etc/"),
        # 更改整个文件系统权限
        re.compile(r"\bchmod\s+-R\s+777\s+/"),
        re.compile(r"\bchown\s+-R\s+"),
    ]

    def check_command(self, command: str) -> SafetyResult:
        """执行完整的命令安全检查。

        Args:
            command: 用户输入的终端命令

        Returns:
            SafetyResult: 安全检查结果
        """
        # 空命令安全
        if not command or not command.strip():
            return SafetyResult(safe=True, reason="Empty command", risk_level="safe")

        matched: list[str] = []

        # 1. 检查黑名单命令
        if matched_blacklist := self._check_blacklist(command):
            matched.extend(matched_blacklist)
            return SafetyResult(
                safe=False,
                reason=f"Blacklisted command detected: {matched_blacklist[0]}",
                risk_level="dangerous",
                matched_patterns=matched,
            )

        # 2. 检查危险模式
        if matched_dangerous := self._check_dangerous_patterns(command):
            matched.extend(matched_dangerous)
            return SafetyResult(
                safe=False,
                reason=f"Dangerous pattern detected: {matched_dangerous[0]}",
                risk_level="dangerous",
                matched_patterns=matched,
            )

        # 3. 检查 shell 注入
        if matched_injection := self._check_shell_injection(command):
            matched.extend(matched_injection)
            risk: RiskLevel = "dangerous"
            return SafetyResult(
                safe=False,
                reason=f"Shell injection detected: {matched_injection[0]}",
                risk_level=risk,
                matched_patterns=matched,
            )

        # 4. 检查文件系统危险操作
        if matched_fs := self._check_filesystem_danger(command):
            matched.extend(matched_fs)
            return SafetyResult(
                safe=False,
                reason=f"Dangerous filesystem operation detected: {matched_fs[0]}",
                risk_level="dangerous",
                matched_patterns=matched,
            )

        return SafetyResult(safe=True, reason="Command passed all safety checks", risk_level="safe")

    def _check_blacklist(self, command: str) -> list[str]:
        """检查命令是否在黑名单中。

        提取命令的首个单词，检查是否在黑名单集合中。

        Returns:
            list[str]: 匹配到的黑名单命令列表
        """
        stripped = command.strip().lstrip("$(")
        # 提取第一个"单词"作为命令
        first_word = stripped.split()[0].split("/")[-1] if stripped.split() else ""
        # 去掉可能的引号
        first_word = first_word.strip("\"'`")
        if first_word in self.BLACKLISTED_COMMANDS:
            return [first_word]
        return []

    def _check_dangerous_patterns(self, command: str) -> list[str]:
        """检查是否匹配危险命令模式。"""
        matched: list[str] = []
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(command):
                matched.append(pattern.pattern)
        return matched

    def _check_shell_injection(self, command: str) -> list[str]:
        """检查是否包含 shell 注入模式。"""
        matched: list[str] = []
        for pattern in self.SHELL_INJECTION_PATTERNS:
            if pattern.search(command):
                matched.append(pattern.pattern)
        return matched

    def _check_filesystem_danger(self, command: str) -> list[str]:
        """检查是否包含危险的文件系统操作。"""
        matched: list[str] = []
        for pattern in self.FILESYSTEM_DANGER_PATTERNS:
            if pattern.search(command):
                matched.append(pattern.pattern)
        return matched
