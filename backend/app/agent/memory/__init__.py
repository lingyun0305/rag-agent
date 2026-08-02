# backend/app/agent/memory/__init__.py
"""
记忆系统模块
"""

from backend.app.agent.memory.long_term import long_term_memory

__all__ = [
    "long_term_memory",
]