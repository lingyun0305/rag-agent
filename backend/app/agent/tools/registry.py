# backend/app/agent/tools/registry.py
"""
Tool 注册中心
统一管理所有工具，支持动态注册、权限控制、热插拔
"""

from typing import Dict, List, Optional
from backend.app.agent.tools.base import BaseTool


class ToolRegistry:
    """
    工具注册中心

    功能：
    - 注册/注销工具
    - 权限分级控制（safe/read/write/admin）
    - 获取 LangChain 工具列表供 Agent 调用
    - 工具热插拔支持
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        注册工具

        Args:
            tool: 工具实例

        Raises:
            ValueError: 工具名已存在或未设置 name
        """
        if not tool.name:
            raise ValueError("工具必须设置 name 属性")
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已存在")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> Optional[BaseTool]:
        """
        注销工具（支持热插拔）

        Args:
            name: 工具名称

        Returns:
            被注销的工具实例，或 None（不存在）
        """
        return self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """
        获取单个工具

        Args:
            name: 工具名称

        Returns:
            工具实例，或 None
        """
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """获取所有工具实例"""
        return list(self._tools.values())

    def get_langchain_tools(self, permission_level: str = "safe") -> List:
        """
        获取 LangChain StructuredTool 列表

        Args:
            permission_level: 权限过滤级别
                - "safe": 只返回安全工具
                - "read": 返回 safe + read
                - "write": 返回 safe + read + write
                - "admin": 返回所有工具

        Returns:
            LangChain StructuredTool 列表
        """
        permission_order = ["safe", "read", "write", "admin"]
        max_index = permission_order.index(permission_level)

        filtered_tools = [
            t for t in self._tools.values()
            if permission_order.index(t.permission) <= max_index
        ]

        return [t.to_langchain_tool() for t in filtered_tools]

    def list_tools(self) -> List[Dict]:
        """
        列出所有工具信息（用于调试/展示）

        Returns:
            工具信息列表
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "permission": t.permission,
                "parameters": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        tools_str = ", ".join(t.name for t in self._tools.values())
        return f"<ToolRegistry: [{tools_str}]>"


# ====== 全局实例 ======

tool_registry = ToolRegistry()


def init_default_tools() -> None:
    """
    初始化默认工具集
    在应用启动时调用
    """
    from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool
    from backend.app.agent.tools.calculator_tool import CalculatorTool
    from backend.app.agent.tools.file_reader_tool import FileReaderTool

    tool_registry.register(KnowledgeSearchTool())
    tool_registry.register(CalculatorTool())
    tool_registry.register(FileReaderTool())