# backend/app/agent/tools/base.py
"""
Tool 抽象基类
定义所有工具的标准接口，支持 LangChain StructuredTool 转换
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError


class ToolResult(BaseModel):
    """
    工具执行结果的标准格式
    """

    success: bool = Field(description="执行是否成功")
    data: Any = Field(default=None, description="返回数据")
    error: str = Field(default="", description="错误信息")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    def to_string(self) -> str:
        """转换为 LLM 可读的字符串格式"""
        if not self.success:
            return f"错误: {self.error}"

        data = self.data
        if data is None:
            return "无数据"

        # 空列表（检索无结果）
        if isinstance(data, list) and not data:
            return "未检索到相关内容，知识库中没有找到与查询匹配的文档。"

        if isinstance(data, list) and isinstance(data[0], dict):
            # 知识库检索结果：格式化为清晰的文本
            if "content" in data[0] and "source" in data[0]:
                return self._format_kb_results(data)

        return str(data)

    @staticmethod
    def _format_kb_results(results: list) -> str:
        """将知识库检索结果格式化为 LLM 易读的文本"""
        parts = []
        for i, item in enumerate(results, 1):
            content = item.get("content", "")
            source = item.get("source", "未知来源")
            if not source or source == "未知来源":
                source = "知识库文档"
            score = item.get("score", item.get("rerank_score", 0))
            score_info = f" (相关度: {score:.2f})" if score else ""
            parts.append(f"[{i}] 来源: {source}{score_info}\n{content}")
        return "\n\n".join(parts)


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
    parameters_schema: dict[str, Any] = {}
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

    def validate_params(self, **kwargs) -> str | None:
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

    def validate_args(self, **kwargs) -> tuple[bool, str | None]:
        """
        基于 parameters_schema 做 Pydantic 强校验（类型 / 必填 / 枚举约束）。

        用于纯 ReAct 执行路径，在 tool.run() 之前拦截参数格式错误，
        错误信息可包装为 Observation 回传模型自行修正。

        Returns:
            (是否通过, 错误信息或 None)
        """
        try:
            schema = self._build_args_schema()
            schema.model_validate(kwargs)
            return True, None
        except ValidationError as e:
            msgs = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) or "<root>"
                msgs.append(f"{loc}: {err['msg']}")
            return False, "；".join(msgs)
        except Exception:
            # schema 构建异常（极少见，如空 schema 工具）时不阻断执行
            return True, None

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

    def _build_args_schema(self) -> type[BaseModel]:
        """
        根据 parameters_schema 构建 Pydantic 参数模型。
        支持 enum 约束（转为 Literal），使强校验能拦截非法枚举值。

        Returns:
            Pydantic 模型类
        """
        if not self.parameters_schema:
            # 无参数的工具返回空模型
            class EmptyArgs(BaseModel):
                pass

            return EmptyArgs

        from typing import Literal

        from pydantic import create_model

        fields = {}
        properties = self.parameters_schema.get("properties", {})  # 所有参数的完整定义
        required = self.parameters_schema.get("required", [])  # 必填参数列表

        for param_name, param_info in properties.items():
            # 尊重 enum：转为 Literal，使 Pydantic 能强校验枚举值
            enum_vals = param_info.get("enum")
            if enum_vals:
                py_type = Literal[tuple(enum_vals)]
            else:
                py_type = self._map_json_type_to_python(param_info.get("type", "string"))
            is_required = param_name in required

            if is_required:
                fields[param_name] = (
                    py_type,
                    Field(description=param_info.get("description", "")),
                )
            else:
                fields[param_name] = (
                    Optional[py_type],
                    Field(default=None, description=param_info.get("description", "")),
                )

        return create_model(f"{self.name}_args", **fields)

    def _map_json_type_to_python(self, json_type: str) -> type:
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
