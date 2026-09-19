# backend/app/agent/tools/__init__.py
"""
Tool 工具调用体系
延迟导入
"""

from typing import Any


class _LazyProxy:
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


BaseTool = _LazyProxy("backend.app.agent.tools.base", "BaseTool")
ToolResult = _LazyProxy("backend.app.agent.tools.base", "ToolResult")
ToolRegistry = _LazyProxy("backend.app.agent.tools.registry", "ToolRegistry")
tool_registry = _LazyProxy("backend.app.agent.tools.registry", "tool_registry")

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
]
