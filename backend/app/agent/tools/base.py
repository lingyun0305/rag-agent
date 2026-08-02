# backend/app/agent/tools/base.py
"""
Tool 抽象基类
定义所有工具的标准接口，支持 LangChain StructuredTool 转换
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class ToolResult(BaseModel):
    """
    工具执行结果的标准格式
    """
    success: bool = Field(description="执行是否成功")
    data: Any = Field(default=None, description="返回数据")
    error: str = Field(default="", description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    def to_string(self) -> str:
        """转换为 LLM 可读的字符串格式"""
        if self.success:
            return str(self.data)
        return f"错误: {self.error}"


class BaseTool(ABC):
    """
    工具抽象基类
    所有 Agent 工具都必须继承此类

    子类需要实现：
    - name: 工具名称（唯一标识）
    - description: 工具描述（给 LLM 看的）
    - parameters_schema: 参数 JSON Schema
    - run(): 异步执行方法
    """

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {}
    permission: str = "safe"  # safe/read/write/admin 四级权限

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """
        工具执行的核心逻辑

        Args:
            **kwargs: 工具参数（由 LLM 根据 parameters_schema 生成）

        Returns:
            ToolResult: 标准化的执行结果
        """
        pass

    def validate_params(self, **kwargs) -> Optional[str]:
        """
        参数校验（可选实现）
        子类可以重写此方法做额外校验

        Args:
            **kwargs: 工具参数

        Returns:
            None: 校验通过
            str: 校验失败的错误信息
        """
        return None

    def to_langchain_tool(self) -> StructuredTool:
        """
        转换为 LangChain StructuredTool
        供 LangGraph Agent 调用

        Returns:
            StructuredTool: LangChain 工具对象
        """

        # 构建 Pydantic 参数模型
        args_schema = self._build_args_schema()

        return StructuredTool.from_function(
            func=self._sync_wrapper,
            coroutine=self.run,
            name=self.name,
            description=self.description,
            args_schema=args_schema,
        )

    def _sync_wrapper(self, **kwargs) -> str:
        """
        同步包装器（用于 StructuredTool 的 func 参数）
        实际调用时会被 coroutine 覆盖，这里仅作为兼容层
        """
        import asyncio
        result = asyncio.run(self.run(**kwargs))
        return result.to_string()

    def _build_args_schema(self) -> Type[BaseModel]:
        """
        根据 parameters_schema 构建 Pydantic 参数模型

        Returns:
            Pydantic 模型类
        """
        if not self.parameters_schema:
            # 无参数的工具返回空模型
            class EmptyArgs(BaseModel):
                pass
            return EmptyArgs

        # 动态构建 Pydantic 模型
        from pydantic import create_model

        fields = {}
        properties = self.parameters_schema.get("properties", {}) # 所有参数的完整定义
        required = self.parameters_schema.get("required", []) # 必填参数列表

        for param_name, param_info in properties.items():
            param_type = self._map_json_type_to_python(param_info.get("type", "string"))
            is_required = param_name in required

            if is_required:
                fields[param_name] = (param_type, Field(
                    description=param_info.get("description", "")
                ))
            else:
                fields[param_name] = (Optional[param_type], Field(
                    default=None,
                    description=param_info.get("description", "")
                ))

        return create_model(f"{self.name}_args", **fields)

    def _map_json_type_to_python(self, json_type: str) -> Type:
        """
        JSON Schema 类型映射到 Python 类型

        Args:
            json_type: JSON Schema 类型字符串

        Returns:
            Python 类型
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)

    def __repr__(self) -> str:
        return f"<Tool: {self.name} (permission={self.permission})>"