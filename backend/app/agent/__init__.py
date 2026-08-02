# backend/app/agent/__init__.py
"""
Agent 核心模块
包含：工具调用体系、主循环、记忆系统、子智能体编排
"""

from backend.app.agent.agent import AgenticRAG
from backend.app.agent.tools.registry import tool_registry
from backend.app.agent.memory.long_term import LongTermMemoryService, long_term_memory

__all__ = [
    "AgenticRAG",
    "tool_registry",
    "LongTermMemoryService",
    "long_term_memory",
]