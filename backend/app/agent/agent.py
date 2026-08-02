# backend/app/agent/agent.py
"""
Agentic RAG 智能体模块
支持 ReAct 模式：单 Agent + Tool 调用
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pathlib import Path

from backend.app.utils.models_factory import model_factory
from backend.app.agent.tools.registry import tool_registry, init_default_tools
from backend.app.core.config import settings


# ====== 状态定义 ======

class AgentState(TypedDict):
    """
    Agent 状态结构
    """
    messages: Annotated[List[BaseMessage], "对话消息列表"]
    user_id: int
    session_id: str
    called_tools: List[str]  # 记录调用的工具名称（用于评测）
    tool_results: List[Dict]  # 工具调用结果（用于评测）
    llm_provider: Optional[str]  # 当前使用的模型提供商
    usage_metadata: Dict[str, Any]  # 累积的token使用统计


# ====== Agent 主类 ======

class AgenticRAG:
    """
    Agentic RAG 智能体

    功能：
    - ReAct 模式：Think → Act → Observe 循环
    - 自动判断是否需要调用工具
    - 支持多工具串行调用
    - 记录调用链路（用于评测）
    """

    def __init__(
        self,
        llm_provider: str = None,
        temperature: float = 0.7,
        permission_level: str = "safe",
    ):
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER
        """
        初始化 Agent

        Args:
            llm_provider: LLM 提供商（qwen/doubao）
            temperature: 生成温度
            permission_level: 工具权限级别（safe/read/write/admin）
        """
        # 初始化工具
        init_default_tools()

        # 保存配置
        self.default_provider = llm_provider
        self.temperature = temperature
        self.permission_level = permission_level

        # 获取 LLM 并绑定工具
        self.llm = model_factory.get_llm_model(
            provider=llm_provider,
            temperature=temperature
        )
        self.tools = tool_registry.get_langchain_tools(permission_level)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 系统提示词
        self.system_prompt = self._load_system_prompt()

        # 构建 LangGraph
        self._build_graph()

    def _get_llm_with_tools(self, llm_provider: Optional[str] = None):
        provider = llm_provider or self.default_provider
        if provider == self.default_provider and self.llm_with_tools:
            return self.llm_with_tools
        llm = model_factory.get_llm_model(provider=provider, temperature=self.temperature)
        return llm.bind_tools(self.tools)

    def _load_system_prompt(self) -> str:
        """
        加载系统提示词

        Returns:
            系统提示词文本
        """
        prompt_path = Path("backend/prompts/agent_system.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        else:
            # 默认提示词（兼容）
            return """你是企业研发辅助智能体。请根据用户问题自主选择合适的工具并完成任务。

【可用工具】
1. knowledge_search：检索企业知识库
2. calculator：数学计算
3. file_reader：读取本地文档

【规则】
- 必须基于工具结果回答，禁止编造
- 工具可串行调用
- 回答末尾标注来源"""

    def _build_graph(self):
        """
        构建 LangGraph 工作流

        流程：
        1. agent: LLM 决策（是否调用工具）
        2. tools: 执行工具调用
        3. 循环直到 LLM 不再调用工具
        4. END: 返回最终答案
        """
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("agent", self._call_llm)
        workflow.add_node("tools", ToolNode(self.tools))

        # 设置入口
        workflow.set_entry_point("agent")

        # 添加条件边：判断是否需要继续调用工具
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        # 工具执行后回到 agent 节点
        workflow.add_edge("tools", "agent")

        # 编译图
        self.graph = workflow.compile()

    async def _call_llm(self, state: AgentState) -> Dict[str, Any]:
        """
        调用 LLM 进行决策

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        # 构建消息列表
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

        # 获取动态 LLM（支持运行时切换）
        llm_provider = state.get("llm_provider")
        llm_with_tools = self._get_llm_with_tools(llm_provider)

        # 调用 LLM
        response = await llm_with_tools.ainvoke(messages)

        # 记录调用的工具（用于评测）
        called_tools = state.get("called_tools", [])
        if response.tool_calls:
            for tool_call in response.tool_calls:
                called_tools.append(tool_call["name"])

        # 累积 token 使用量
        usage_metadata = getattr(response, "usage_metadata", None)
        total_usage = state.get("usage_metadata", {})
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
                output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
                total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
            else:
                input_tokens = getattr(usage_metadata, "input_tokens", getattr(usage_metadata, "prompt_tokens", 0))
                output_tokens = getattr(usage_metadata, "output_tokens", getattr(usage_metadata, "completion_tokens", 0))
                total_tokens = getattr(usage_metadata, "total_tokens", input_tokens + output_tokens)
            total_usage["input_tokens"] = total_usage.get("input_tokens", 0) + int(input_tokens or 0)
            total_usage["output_tokens"] = total_usage.get("output_tokens", 0) + int(output_tokens or 0)
            total_usage["total_tokens"] = total_usage.get("total_tokens", 0) + int(total_tokens or 0)
            total_usage["llm_calls"] = total_usage.get("llm_calls", 0) + 1

        return {
            "messages": [response],
            "called_tools": called_tools,
            "usage_metadata": total_usage,
        }

    def _should_continue(self, state: AgentState) -> str:
        """
        判断是否需要继续调用工具

        Args:
            state: 当前状态

        Returns:
            "continue" 或 "end"
        """
        last_message = state["messages"][-1]

        # 如果有 tool_calls，继续执行工具
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # 否则结束
        return "end"

    async def run(
        self,
        query: str,
        user_id: int,
        session_id: str,
        history: Optional[List[Dict]] = None,
        llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行 Agent 对话

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            history: 对话历史（可选）

        Returns:
            {
                "answer": str,  # 最终答案
                "called_tools": list,  # 调用的工具列表
                "tool_results": list,  # 工具执行结果
                "messages": list,  # 完整消息列表
            }
        """
        # 构建初始消息
        messages = []

        # 加入历史消息
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # 加入当前问题
        messages.append(HumanMessage(content=query))

        # 初始化状态
        initial_state = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "called_tools": [],
            "tool_results": [],
            "llm_provider": llm_provider,
            "usage_metadata": {},
        }

        # 执行工作流
        result = await self.graph.ainvoke(initial_state)

        # 提取最终答案
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        return {
            "answer": answer,
            "called_tools": result.get("called_tools", []),
            "tool_results": result.get("tool_results", []),
            "messages": [str(m.content) if hasattr(m, "content") else str(m) for m in result["messages"]],
            "usage_metadata": result.get("usage_metadata", {}),
        }

    async def run_stream(
        self,
        query: str,
        user_id: int,
        session_id: str,
        history: Optional[List[Dict]] = None,
        llm_provider: Optional[str] = None,
        usage_collector: Optional[Dict] = None,
    ):
        """
        流式输出 Agent 对话

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            history: 对话历史
            usage_collector: 用于收集token使用信息的字典（传引用，内部会填充）

        Yields:
            字符串 chunk
        """
        # 构建消息
        messages = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))

        initial_state = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "called_tools": [],
            "tool_results": [],
            "llm_provider": llm_provider,
            "usage_metadata": {},
        }

        # 流式执行
        last_usage = {}
        async for event in self.graph.astream(initial_state):
            # 提取 agent 节点的输出
            if "agent" in event:
                message = event["agent"]["messages"][-1]
                if hasattr(message, "content") and message.content:
                    yield message.content
                # 累积 usage_metadata
                if "usage_metadata" in event["agent"]:
                    last_usage = event["agent"]["usage_metadata"]

        # 将最终统计写入 collector
        if usage_collector is not None:
            usage_collector["usage"] = last_usage

    def get_tools_info(self) -> List[Dict]:
        """
        获取工具信息列表（用于调试）

        Returns:
            工具信息列表
        """
        return tool_registry.list_tools()


# ====== 全局实例（延迟初始化）======

_agentic_rag_instance = None

def get_agentic_rag(llm_provider: str = None):
    global _agentic_rag_instance
    if _agentic_rag_instance is None:
        _agentic_rag_instance = AgenticRAG(llm_provider=llm_provider)
    elif llm_provider is not None and _agentic_rag_instance.default_provider != llm_provider:
        _agentic_rag_instance = AgenticRAG(llm_provider=llm_provider)
    return _agentic_rag_instance

class _AgenticRAGProxy:
    def __getattr__(self, name):
        return getattr(get_agentic_rag(), name)

agentic_rag = _AgenticRAGProxy()