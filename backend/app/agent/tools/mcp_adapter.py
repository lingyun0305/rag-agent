# backend/app/agent/tools/mcp_adapter.py
"""
MCP 通用适配器：自建工具注册中心 → 行业标准 MCP 协议

设计思路（架构层，而非逐个手写包装）：
  ToolRegistry 中每个工具已自带 MCP server 需要的全部元数据
  （name / description / parameters_schema / permission）。
  McpAdapter 做的事就是「遍历注册表 → 自动生成 MCP tool 定义 → 注册到同一个 FastMCP server」：

      ToolRegistry ──McpAdapter.build_server()──> FastMCP server (stdio)
                                                       ├── tools/list  → 全部工具声明
                                                       └── tools/call  → 分发到 BaseTool.run()

收益：
- 一个 server 自动暴露全部工具，以后新增工具零改动（注册进 registry 即自动可见）；
- 任何标准 MCP client（Claude Desktop / 其它 Agent 框架）无需了解本项目内部实现，
  即可发现并调用全部工具；
- 权限分级（safe/read/write/admin）在暴露侧复用，与 Agent 内部同一套语义。

启动方式（stdio 传输，供 MCP client 子进程拉起）：
  venv/Scripts/python.exe backend/app/agent/tools/mcp_adapter.py           # 轻量工具集（calculator/file_reader/oa）
  venv/Scripts/python.exe backend/app/agent/tools/mcp_adapter.py --all     # 含 knowledge_search（依赖检索服务）
"""

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Literal

# 必须先于 backend.* import：被 MCP client 作为独立子进程脚本拉起时，
# 解释器 sys.path 里没有项目根，这里兜底插入（cwd=项目根时是 no-op）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from backend.app.agent.tools.base import BaseTool  # noqa: E402
from backend.app.agent.tools.registry import ToolRegistry  # noqa: E402

# JSON Schema 类型 → Python 类型（与 base.BaseTool._map_json_type_to_python 同一映射）
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_PERMISSION_ORDER = ["safe", "read", "write", "admin"]


class McpAdapter:
    """
    把 ToolRegistry（或任意 BaseTool 列表）适配为 MCP server。

    用法：
        adapter = McpAdapter(registry=tool_registry, max_permission="safe")
        mcp = adapter.build_server()
        mcp.run(transport="stdio")
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        tools: list[BaseTool] | None = None,
        max_permission: str = "admin",
    ):
        """
        Args:
            registry: 工具注册中心（与 tools 二选一）
            tools: 直接给 BaseTool 列表（与 registry 二选一）
            max_permission: 暴露给 MCP 的最高权限级别（safe/read/write/admin），
                            与 registry.get_langchain_tools 的过滤语义一致
        """
        if tools is None:
            if registry is None:
                raise ValueError("registry 与 tools 至少提供一个")
            tools = self._filter_by_permission(registry.get_all(), max_permission)
        self._tools = list(tools)

    @staticmethod
    def _filter_by_permission(tools: list[BaseTool], max_permission: str) -> list[BaseTool]:
        if max_permission not in _PERMISSION_ORDER:
            raise ValueError(f"未知权限级别: {max_permission}")
        max_idx = _PERMISSION_ORDER.index(max_permission)
        return [
            t
            for t in tools
            if t.permission in _PERMISSION_ORDER
            and _PERMISSION_ORDER.index(t.permission) <= max_idx
        ]

    # ====== server 构建 ======

    def build_server(
        self,
        name: str = "rag-agent-tools",
        instructions: str = (
            "本地化 ReAct Agent 系统的工具集。所有工具由项目内 ToolRegistry 自动注册，"
            "参数约束见各工具 inputSchema。"
        ),
    ) -> FastMCP:
        """遍历工具集，全部注册到同一个 FastMCP server 并返回"""
        mcp = FastMCP(name, instructions=instructions)
        for tool in self._tools:
            self._register_tool(mcp, tool)
        return mcp

    def _register_tool(self, mcp: FastMCP, tool: BaseTool) -> None:
        """单个工具：BaseTool → MCP tool 定义（name/description/annotations + 动态签名）"""
        mcp.add_tool(
            self._make_endpoint(tool),
            name=tool.name,
            description=tool.description,
            annotations=ToolAnnotations(
                title=tool.name,
                readOnlyHint=tool.permission == "safe",
            ),
        )

    def _make_endpoint(self, tool: BaseTool):
        """
        为工具生成一个「签名与 parameters_schema 一致」的异步端点函数，
        FastMCP 会据此自动推导 tools/list 返回的 inputSchema。

        这样 parameters_schema 是唯一事实源：注册中心里改 schema，MCP 侧自动跟随。
        """
        params: list[inspect.Parameter] = []
        properties: dict[str, Any] = tool.parameters_schema.get("properties", {})
        required: set[str] = set(tool.parameters_schema.get("required", []))

        for pname, pinfo in properties.items():
            # 注解保持「裸类型」：FastMCP 仅对 annotation is not str 的字段做 JSON 预解析，
            # 若包成 Optional[str] 会导致 params 这类"JSON 字符串参数"被误解析成 dict 而校验失败。
            # 非必填参数只靠 default（Pydantic 默认不校验 default 值），不影响 inputSchema 正确性。
            ann = self._annotation(pinfo)
            if pname in required:
                params.append(
                    inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann)
                )
            else:
                params.append(
                    inspect.Parameter(
                        pname,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=ann,
                        default=pinfo.get("default", None),
                    )
                )

        async def endpoint(**kwargs) -> str:
            result = await tool.run(**kwargs)
            if result.success and isinstance(result.data, (dict, list)):
                # dict/list 用 JSON 输出，保持结构化、对 MCP client 友好
                return json.dumps(result.data, ensure_ascii=False, default=str)
            return result.to_string()

        endpoint.__name__ = f"call_{tool.name}"
        endpoint.__doc__ = (tool.description or "").split("\n")[0]
        endpoint.__signature__ = inspect.Signature(params)
        return endpoint

    @staticmethod
    def _annotation(param_info: dict[str, Any]) -> type:
        enum_vals = param_info.get("enum")
        if enum_vals:
            return Literal[tuple(enum_vals)]  # type: ignore[valid-type]
        return _JSON_TYPE_MAP.get(param_info.get("type", "string"), str)


# ====== 独立启动入口（stdio server） ======


def bootstrap_tools(include_all: bool = False) -> ToolRegistry:
    """
    组装 MCP server 要暴露的工具集。

    默认只注册三个轻量工具（不依赖检索服务，可离线跑通协议全流程）；
    --all 时额外注册 knowledge_search（会拉起检索相关依赖）。
    """
    from backend.app.agent.tools.calculator_tool import CalculatorTool
    from backend.app.agent.tools.file_reader_tool import FileReaderTool
    from backend.app.agent.tools.oa_mock_tool import OaMockTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool())
    registry.register(OaMockTool())

    if include_all:
        from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool

        registry.register(KnowledgeSearchTool())

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Agent MCP server (stdio)")
    parser.add_argument("--all", action="store_true", help="包含 knowledge_search 等全部工具")
    args = parser.parse_args()

    # ⚠️ stdio 传输下 stdout 是 JSONRPC 协议通道，任何日志写 stdout 都会污染协议帧。
    # 项目 logger（backend.app.common.logger）默认往 stdout 打（loguru 全局对象），
    # 这里把它全部摘掉、改挂到 stderr，保证 stdout 只有协议消息。
    from loguru import logger as _loguru

    _loguru.remove()
    _loguru.add(sys.stderr, level="WARNING")

    registry = bootstrap_tools(include_all=args.all)
    adapter = McpAdapter(registry=registry, max_permission="admin")
    mcp = adapter.build_server()

    _loguru.info(f"MCP server 就绪，暴露工具: {[t.name for t in adapter._tools]}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
