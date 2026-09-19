# backend/app/agent/tools/registry.py
"""
Tool 注册中心
统一管理所有工具，支持动态注册、权限控制、热插拔

工具来源：
1. 内置系统工具（Python 类实现）：calculator, knowledge_search, file_reader
2. 自定义工具（数据库配置）：通过 code_path 指定 Python 模块或 HTTP API endpoint
"""

import importlib
import json

from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.common.logger import logger


class ToolRegistry:
    """
    工具注册中心

    功能：
    - 注册/注销工具
    - 权限分级控制（safe/read/write/admin）
    - 获取 LangChain 工具列表供 Agent 调用
    - 从数据库动态加载工具
    - 工具热插拔支持
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

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

    def unregister(self, name: str) -> BaseTool | None:
        """
        注销工具（支持热插拔）
        """
        return self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """获取单个工具"""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """获取所有工具实例"""
        return list(self._tools.values())

    def get_langchain_tools(self, permission_level: str = "safe") -> list:
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
            t for t in self._tools.values() if permission_order.index(t.permission) <= max_index
        ]

        return [t.to_langchain_tool() for t in filtered_tools]

    def get_tools_by_names(self, names: list[str]) -> list:
        """
        按名称获取 LangChain 工具列表

        Args:
            names: 工具名称列表

        Returns:
            LangChain StructuredTool 列表
        """
        tools = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool.to_langchain_tool())
            else:
                logger.warning(f"工具 '{name}' 未找到，跳过")
        return tools

    def list_tools(self) -> list[dict]:
        """
        列出所有工具信息（用于调试/展示）
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

    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()

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
    注册所有内置系统工具
    """
    from backend.app.agent.tools.calculator_tool import CalculatorTool
    from backend.app.agent.tools.file_reader_tool import FileReaderTool
    from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool
    from backend.app.agent.tools.oa_mock_tool import OaMockTool

    tool_registry.register(KnowledgeSearchTool())
    tool_registry.register(CalculatorTool())
    tool_registry.register(FileReaderTool())
    tool_registry.register(OaMockTool())

    logger.info(f"默认工具初始化完成: {list(tool_registry._tools.keys())}")


async def load_agent_tools(
    agent_id: str,
    db=None,
    permission_level: str = "safe",
) -> list:
    """
    根据 Agent ID 加载关联的工具列表

    规则：
    - 有 agent_id + db → 只返回 Agent 绑定的工具
    - knowledge_search 工具会注入 Agent 绑定的知识库 ID 列表及描述

    Args:
        agent_id: Agent ID
        db: 数据库会话
        permission_level: 权限级别过滤

    Returns:
        LangChain StructuredTool 列表
    """
    from backend.app.models.agent import AgentTool

    if db is None:
        return []

    try:
        from sqlalchemy import select

        from backend.app.models.agent import Agent, AgentKnowledgeBase
        from backend.app.models.knowledge import KnowledgeBase
        from backend.app.models.tool import Tool

        # 1. 加载 Agent 绑定的知识库 ID 列表及描述
        kb_ids = []
        kb_descriptions = {}  # {kb_id: {"name": ..., "desc": ...}}
        agent_kb_result = await db.execute(
            select(AgentKnowledgeBase).where(AgentKnowledgeBase.agent_id == agent_id)
        )
        agent_kb_bindings = agent_kb_result.scalars().all()
        kb_ids = [binding.kb_id for binding in agent_kb_bindings]

        # 获取每个知识库的名称和描述
        if kb_ids:
            kb_info_result = await db.execute(
                select(KnowledgeBase.id, KnowledgeBase.name, KnowledgeBase.desc).where(
                    KnowledgeBase.id.in_(kb_ids), KnowledgeBase.status == "active"
                )
            )
            for row in kb_info_result:
                kb_descriptions[row.id] = {
                    "name": row.name,
                    "desc": row.desc or "",
                }

        logger.info(f"Agent '{agent_id}' 绑定了 {len(kb_ids)} 个知识库: {kb_ids}")

        # 2. 查询 AgentTool 关联表（获取所有工具，包括系统工具和自定义工具）
        result = await db.execute(select(AgentTool).where(AgentTool.agent_id == agent_id))
        agent_tools = result.scalars().all()

        # 3. 检查 Agent.system_tool_names（新方式：系统工具存储在 Agent 表的 system_tool_names 字段）
        agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent_obj = agent_result.scalar_one_or_none()
        system_tool_names = agent_obj.get_system_tools() if agent_obj else []
        # 工具级配置(按工具名索引), 用于 knowledge_search 等工具的检索参数覆盖
        agent_tool_configs = agent_obj.get_tool_configs() if agent_obj else {}

        # 兼容旧数据：如果 system_tool_names 为空，从 AgentTool 表中加载系统工具的 code_path
        if not system_tool_names and agent_tools:
            tool_ids = [t.tool_id for t in agent_tools]
            tools_result = await db.execute(
                select(Tool).where(
                    Tool.id.in_(tool_ids), Tool.status == "active", Tool.tool_type == "system"
                )
            )
            for t in tools_result.scalars().all():
                if t.code_path and t.code_path not in system_tool_names:
                    system_tool_names.append(t.code_path)
                    logger.info(f"从 AgentTool 表兼容加载系统工具: {t.code_path}")

        # 自动推断：如果 Agent 绑定了知识库，自动添加 knowledge_search 工具
        if kb_ids and "knowledge_search" not in system_tool_names:
            system_tool_names.append("knowledge_search")
            logger.info(f"Agent '{agent_id}' 绑定了知识库，自动添加 knowledge_search 工具")

        # 注意：calculator 不再自动挂载，由用户在 Agent 配置里显式绑定。
        # 这样可以避免工具列表与用户预期不一致。

        # 如果 Agent 没有任何工具绑定，返回空
        if not agent_tools and not system_tool_names:
            return []

        # 4. 获取数据库中绑定的工具详情（自定义工具）
        db_tool_map = {}
        if agent_tools:
            tool_ids = [t.tool_id for t in agent_tools]
            tools_result = await db.execute(
                select(Tool).where(
                    Tool.id.in_(tool_ids), Tool.status == "active", Tool.tool_type == "custom"
                )
            )
            for db_tool in tools_result.scalars().all():
                db_tool_map[db_tool.name] = db_tool

        # 5. 构建最终的工具列表
        langchain_tools = []

        # 5a. 处理 system_tool_names（如 knowledge_search, calculator）
        for tool_name in system_tool_names:
            if tool_name == "knowledge_search":
                # 特殊处理：创建带 kb_ids 和 kb_descriptions 的 KnowledgeSearchTool 实例
                from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool

                kb_tool = KnowledgeSearchTool(
                    kb_ids=kb_ids,
                    kb_descriptions=kb_descriptions if kb_ids else None,
                    retrieval_config=agent_tool_configs.get("knowledge_search"),
                )
                langchain_tools.append(kb_tool.to_langchain_tool())
                logger.info(
                    f"为 Agent 创建带知识库绑定的 KnowledgeSearchTool: kb_ids={kb_ids}, kb_descriptions_count={len(kb_descriptions)}"
                )
            else:
                # 其他系统工具从 registry 获取
                tool_obj = tool_registry.get(tool_name)
                if tool_obj:
                    langchain_tools.append(tool_obj.to_langchain_tool())

        # 5b. 处理 AgentTool 关联表中的自定义工具
        for db_tool in db_tool_map.values():
            if db_tool.tool_type == "custom":
                custom_tool = _load_custom_tool(db_tool)
                if custom_tool:
                    langchain_tools.append(custom_tool.to_langchain_tool())

        if not langchain_tools:
            return []

        logger.info(f"Agent '{agent_id}' 最终加载 {len(langchain_tools)} 个工具")
        return langchain_tools

    except Exception as e:
        logger.error(f"加载 Agent 工具失败: {e}", exc_info=True)
        return []


async def load_agent_base_tools(
    agent_id: str,
    db=None,
    permission_level: str = "safe",
) -> list:
    """
    根据 Agent ID 加载关联的工具列表（返回 BaseTool 实例，供纯 ReAct 直接调用 run()）。

    逻辑与 load_agent_tools 一致，但返回 BaseTool 实例而非 LangChain StructuredTool，
    便于在「解析 <action>/<input> 标签 → 直接 tool.run(**args)」的范式下执行。

    Returns:
        BaseTool 实例列表
    """
    from backend.app.models.agent import AgentTool

    if db is None:
        return []

    try:
        from sqlalchemy import select

        from backend.app.models.agent import Agent, AgentKnowledgeBase
        from backend.app.models.knowledge import KnowledgeBase
        from backend.app.models.tool import Tool

        # 1. 加载 Agent 绑定的知识库 ID 列表及描述
        kb_ids = []
        kb_descriptions = {}
        agent_kb_result = await db.execute(
            select(AgentKnowledgeBase).where(AgentKnowledgeBase.agent_id == agent_id)
        )
        agent_kb_bindings = agent_kb_result.scalars().all()
        kb_ids = [binding.kb_id for binding in agent_kb_bindings]

        if kb_ids:
            kb_info_result = await db.execute(
                select(KnowledgeBase.id, KnowledgeBase.name, KnowledgeBase.desc).where(
                    KnowledgeBase.id.in_(kb_ids), KnowledgeBase.status == "active"
                )
            )
            for row in kb_info_result:
                kb_descriptions[row.id] = {
                    "name": row.name,
                    "desc": row.desc or "",
                }

        logger.info(f"Agent '{agent_id}' 绑定了 {len(kb_ids)} 个知识库: {kb_ids}")

        # 2. 查询 AgentTool 关联表
        result = await db.execute(select(AgentTool).where(AgentTool.agent_id == agent_id))
        agent_tools = result.scalars().all()

        # 3. 检查 Agent.system_tool_names
        agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent_obj = agent_result.scalar_one_or_none()
        system_tool_names = agent_obj.get_system_tools() if agent_obj else []
        agent_tool_configs = agent_obj.get_tool_configs() if agent_obj else {}

        if not system_tool_names and agent_tools:
            tool_ids = [t.tool_id for t in agent_tools]
            tools_result = await db.execute(
                select(Tool).where(
                    Tool.id.in_(tool_ids), Tool.status == "active", Tool.tool_type == "system"
                )
            )
            for t in tools_result.scalars().all():
                if t.code_path and t.code_path not in system_tool_names:
                    system_tool_names.append(t.code_path)
                    logger.info(f"从 AgentTool 表兼容加载系统工具: {t.code_path}")

        if kb_ids and "knowledge_search" not in system_tool_names:
            system_tool_names.append("knowledge_search")
            logger.info(f"Agent '{agent_id}' 绑定了知识库，自动添加 knowledge_search 工具")

        if not agent_tools and not system_tool_names:
            return []

        # 4. 自定义工具详情
        db_tool_map = {}
        if agent_tools:
            tool_ids = [t.tool_id for t in agent_tools]
            tools_result = await db.execute(
                select(Tool).where(
                    Tool.id.in_(tool_ids), Tool.status == "active", Tool.tool_type == "custom"
                )
            )
            for db_tool in tools_result.scalars().all():
                db_tool_map[db_tool.name] = db_tool

        # 5. 构建 BaseTool 实例列表
        base_tools = []

        for tool_name in system_tool_names:
            if tool_name == "knowledge_search":
                from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool

                kb_tool = KnowledgeSearchTool(
                    kb_ids=kb_ids,
                    kb_descriptions=kb_descriptions if kb_ids else None,
                    retrieval_config=agent_tool_configs.get("knowledge_search"),
                )
                base_tools.append(kb_tool)
                logger.info(
                    f"为 Agent 创建带知识库绑定的 KnowledgeSearchTool(BaseTool): kb_ids={kb_ids}"
                )
            else:
                tool_obj = tool_registry.get(tool_name)
                if tool_obj:
                    base_tools.append(tool_obj)

        for db_tool in db_tool_map.values():
            if db_tool.tool_type == "custom":
                custom_tool = _load_custom_tool(db_tool)
                if custom_tool:
                    base_tools.append(custom_tool)

        if not base_tools:
            return []

        logger.info(f"Agent '{agent_id}' 最终加载 {len(base_tools)} 个 BaseTool")
        return base_tools

    except Exception as e:
        logger.error(f"加载 Agent BaseTool 失败: {e}", exc_info=True)
        return []


def _load_custom_tool(db_tool) -> BaseTool | None:
    """
    从数据库 Tool 记录动态加载自定义工具

    支持两种方式：
    1. Python 模块：code_path 指向可导入的 Python 类路径
       例如 "backend.app.agent.tools.custom.my_tool.MyTool"
    2. HTTP API：code_path 为 URL，动态生成 APICallTool

    Args:
        db_tool: Tool ORM 记录

    Returns:
        BaseTool 实例
    """

    # 判断 code_path 是 Python 模块还是 HTTP URL
    code_path = db_tool.code_path or ""

    if code_path.startswith("http://") or code_path.startswith("https://"):
        # HTTP API 工具
        return _create_http_tool(db_tool)
    elif code_path:
        # Python 模块工具
        return _create_python_tool(db_tool)
    else:
        logger.warning(f"工具 '{db_tool.name}' 没有有效的 code_path")
        return None


def _create_python_tool(db_tool) -> BaseTool | None:
    """
    从 Python 模块路径加载工具

    code_path 格式: "module.path.ClassName"
    例如: "backend.app.agent.tools.custom.my_tool.MyTool"
    """
    from backend.app.agent.tools.base import BaseTool

    code_path = db_tool.code_path
    try:
        # 解析模块路径和类名
        parts = code_path.rsplit(".", 1)
        if len(parts) != 2:
            logger.error(f"工具 code_path 格式错误: {code_path}")
            return None

        module_path, class_name = parts
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)

        # 实例化工具
        instance = tool_class()
        if not isinstance(instance, BaseTool):
            logger.error(f"类 {class_name} 未继承 BaseTool")
            return None

        return instance

    except (ImportError, AttributeError) as e:
        logger.error(f"加载 Python 工具 '{db_tool.name}' 失败: {e}")
        return None


def _create_http_tool(db_tool) -> BaseTool:
    """
    创建 HTTP API 调用工具

    动态生成一个可通过 HTTP 请求外部 API 的工具
    """
    from backend.app.agent.tools.base import BaseTool

    url = db_tool.code_path
    tool_name = db_tool.name
    tool_desc = db_tool.desc or f"调用外部 API: {url}"
    parameters_schema = {}
    if db_tool.parameters_schema:
        try:
            parameters_schema = json.loads(db_tool.parameters_schema)
        except json.JSONDecodeError:
            pass

    class HTTPAPITool(BaseTool):
        """HTTP API 动态工具"""

        name = tool_name
        description = tool_desc
        permission = "safe"
        parameters_schema = parameters_schema or {
            "type": "object",
            "properties": {
                "params": {
                    "type": "string",
                    "description": "请求参数（JSON 字符串）",
                }
            },
            "required": ["params"],
        }

        async def run(self, params: str = "{}", **kwargs) -> ToolResult:
            import aiohttp

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=json.loads(params) if isinstance(params, str) else params,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        result = await resp.json()
                        return ToolResult(
                            success=True,
                            data=result,
                            metadata={"url": url, "status_code": resp.status},
                        )
            except Exception as e:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"HTTP 调用失败: {str(e)}",
                )

    return HTTPAPITool()
