# backend/app/agent/__init__.py
"""
Agent 核心模块
包含：工具调用体系、主循环、记忆系统、子智能体编排
延迟导入，避免模块加载时触发 Embedding 模型初始化
"""

from typing import Any


class _LazyProxy:
    """延迟导入代理"""

    def __init__(self, module_path: str, attr_name: str):
        self._module_path = module_path
        self._attr_name = attr_name
        self._instance = None

    def _resolve(self):
        if self._instance is None:
            import importlib

            module = importlib.import_module(self._module_path)
            self._instance = getattr(module, self._attr_name)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __repr__(self):
        return repr(self._resolve())


SmartAgent = _LazyProxy("backend.app.agent.agent", "SmartAgent")
tool_registry = _LazyProxy("backend.app.agent.tools.registry", "tool_registry")
LongTermMemoryService = _LazyProxy("backend.app.agent.memory.long_term", "LongTermMemoryService")
long_term_memory = _LazyProxy("backend.app.agent.memory.long_term", "long_term_memory")

__all__ = [
    "SmartAgent",
    "tool_registry",
    "LongTermMemoryService",
    "long_term_memory",
]
