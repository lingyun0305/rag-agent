# backend/app/agent/context/__init__.py
"""
上下文工程模块
包含：上下文压缩、记忆注入、窗口管理
"""

from backend.app.agent.context.compressor import ContextCompressor, context_compressor

__all__ = [
    "ContextCompressor",
    "context_compressor",
]