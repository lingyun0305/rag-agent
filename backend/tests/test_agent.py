# backend/tests/test_agent.py
"""
Agent 核心逻辑单元测试
测试工具调用解析、强制知识库触发、系统提示词构建等
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在导入业务模块前，mock 掉会触发模型下载的依赖
sys.modules["backend.app.rag.rerank_service"] = MagicMock()
sys.modules["backend.app.rag.bm25_service"] = MagicMock()
sys.modules["backend.app.rag.hyde_service"] = MagicMock()
sys.modules["backend.app.utils.vector_store"] = MagicMock()
sys.modules["backend.app.utils.embedding"] = MagicMock()

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, PropertyMock
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
import json

from backend.app.agent.agent import SmartAgent, AgentState
from backend.app.agent.tools.registry import tool_registry


# ====== 工具注册（不触发模型下载）======
def _register_test_tools():
    """注册测试所需的工具"""
    from backend.app.agent.tools.calculator_tool import CalculatorTool
    from backend.app.agent.tools.file_reader_tool import FileReaderTool

    tool_registry._tools.clear()
    tool_registry.register(CalculatorTool())
    tool_registry.register(FileReaderTool())


_register_test_tools()


# ====== Fixtures ======
@pytest.fixture
def mock_llm():
    """创建 mock LLM 实例"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="你好，这是测试"))
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_tools():
    """创建模拟工具列表"""
    from backend.app.agent.tools.calculator_tool import CalculatorTool
    from backend.app.agent.tools.file_reader_tool import FileReaderTool

    return [
        CalculatorTool().to_langchain_tool(),
        FileReaderTool().to_langchain_tool(),
    ]


@pytest.fixture
def agent_with_tools(mock_llm, mock_tools):
    """创建带工具的 Agent 实例（带 mock LLM）"""
    with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
        agent = SmartAgent(
            llm_provider="qwen",
            pre_loaded_tools=mock_tools,
            agent_id="test-agent",
        )
    return agent


@pytest.fixture
def agent_without_tools(mock_llm):
    """创建无工具的 Agent 实例（带 mock LLM）"""
    with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
        agent = SmartAgent(
            llm_provider="qwen",
            pre_loaded_tools=[],
            agent_id=None,
        )
    return agent


@pytest.fixture
def agent_instance(mock_llm, mock_tools):
    """通用 Agent 实例"""
    with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
        agent = SmartAgent(
            llm_provider="qwen",
            pre_loaded_tools=mock_tools,
            agent_id="test-agent",
        )
    return agent


# ====== 测试：文本工具调用解析 ======
class TestTextToolCallParsing:
    """测试文本形式的工具调用解析"""

    def test_parse_simple_tool_call_tag(self):
        """解析 <tool_call> 标签包裹的简单工具调用"""
        content = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+3"}}</tool_call>'
        result = SmartAgent._parse_text_tool_calls(content)

        assert result is not None
        assert result[0]["name"] == "calculator"

    def test_parse_no_known_tool(self):
        """未知工具名返回 None"""
        content = '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
        result = SmartAgent._parse_text_tool_calls(content)

        assert result is None

    def test_parse_empty_content(self):
        """空内容返回 None"""
        result = SmartAgent._parse_text_tool_calls("")
        assert result is None

    def test_parse_json_with_noise_prefix(self):
        """处理噪声前缀"""
        content = 'letson {"name": "calculator", "arguments": {"expression": "2+3"}}'
        result = SmartAgent._parse_text_tool_calls(content)

        assert result is not None
        assert result[0]["name"] == "calculator"

    def test_parse_arguments_as_string(self):
        """处理 JSON 字符串格式的 arguments"""
        # 构造实际包含字符串 arguments 的 JSON
        content = '<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"100/5\\"}"}</tool_call>'
        result = SmartAgent._parse_text_tool_calls(content)

        assert result is not None
        assert result[0]["name"] == "calculator"
        assert result[0]["args"]["expression"] == "100/5"

    def test_parse_args_key_alternative(self):
        """处理 args 键名的备选"""
        content = '{"name": "calculator", "args": {"expression": "65+9"}}'
        result = SmartAgent._parse_text_tool_calls(content)

        assert result is not None
        assert result[0]["name"] == "calculator"


# ====== 测试：JSON 对象提取 ======
class TestJSONExtraction:
    """测试 JSON 对象提取"""

    def test_extract_simple_json(self):
        """提取简单 JSON"""
        text = 'some text {"key": "value"} more text'
        result = SmartAgent._extract_json_objects(text)
        assert len(result) == 1
        assert result[0] == '{"key": "value"}'

    def test_extract_nested_json(self):
        """提取嵌套 JSON"""
        text = '{"a": {"b": {"c": 3}}}'
        result = SmartAgent._extract_json_objects(text)
        assert len(result) >= 1

    def test_extract_multiple_json(self):
        """提取多个 JSON 对象"""
        text = '{"first": 1} and {"second": 2}'
        result = SmartAgent._extract_json_objects(text)
        assert len(result) == 2

    def test_extract_no_json(self):
        """无 JSON 时返回空列表"""
        text = 'just plain text without any json'
        result = SmartAgent._extract_json_objects(text)
        assert result == []

    def test_extract_mismatched_braces(self):
        """不匹配的括号处理"""
        text = '{unclosed brace'
        result = SmartAgent._extract_json_objects(text)
        assert result == []


# ====== 测试：失败工具调用检测 ======
class TestFailedToolCallDetection:
    """测试失败工具调用检测"""

    def test_detect_failed_attempt_with_tool_name(self):
        """内容包含工具名时返回 True"""
        content = 'letson {"name": "calculator"'
        result = SmartAgent._is_failed_tool_call_attempt(content)
        assert result is True

    def test_no_failed_attempt_without_brace(self):
        """无花括号时返回 False"""
        content = 'calculator is a tool'
        result = SmartAgent._is_failed_tool_call_attempt(content)
        assert result is False

    def test_no_failed_attempt_for_normal_text(self):
        """正常文本返回 False"""
        content = '你好，今天天气怎么样'
        result = SmartAgent._is_failed_tool_call_attempt(content)
        assert result is False


# ====== 测试：强制知识库触发 ======
class TestForceToolCall:
    """测试强制触发知识库查询"""

    def test_force_hr_keyword(self, agent_with_tools):
        """人事考勤关键词应触发强制调用"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            result = agent_with_tools._check_and_force_tool_call("公司的年假规定是什么？")
            assert result is not None
            assert result[0]["name"] == "knowledge_search"

    def test_force_policy_keyword(self, agent_with_tools):
        """公司制度关键词应触发强制调用"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            result = agent_with_tools._check_and_force_tool_call("公司的报销政策是什么？")
            assert result is not None

    def test_force_product_keyword(self, agent_with_tools):
        """产品相关关键词应触发强制调用"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            result = agent_with_tools._check_and_force_tool_call("产品的使用指南在哪里？")
            assert result is not None

    def test_no_force_for_greeting(self, agent_with_tools):
        """问候语不应触发强制调用"""
        result = agent_with_tools._check_and_force_tool_call("你好")
        assert result is None

    def test_no_force_for_calculation(self, agent_with_tools):
        """纯计算问题应触发 calculator 强制调用，而非知识库"""
        result = agent_with_tools._check_and_force_tool_call("125*8等于多少")
        assert result is not None
        # 验证是 calculator 而非 knowledge_search
        tool_names = [tc["name"] for tc in result]
        assert "calculator" in tool_names

    def test_force_other_hr_keywords(self, agent_with_tools):
        """其他人事关键词"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            queries = [
                "如何申请加班",
                "员工入职流程",
                "转正需要什么材料",
                "绩效考核标准是什么",
            ]
            for query in queries:
                result = agent_with_tools._check_and_force_tool_call(query)
                assert result is not None, f"Query '{query}' should trigger force call"

    def test_force_admin_keywords(self, agent_with_tools):
        """行政办公关键词"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            queries = [
                "如何采购办公用品",
                "合同审批流程",
                "如何开具发票",
            ]
            for query in queries:
                result = agent_with_tools._check_and_force_tool_call(query)
                assert result is not None, f"Query '{query}' should trigger force call"

    def test_force_sales_keywords(self, agent_with_tools):
        """销售与数据关键词"""
        with patch.dict(tool_registry._tools, {"knowledge_search": MagicMock()}):
            queries = [
                "上个月的销售数据",
                "产品销量排名",
                "最新上架的产品",
            ]
            for query in queries:
                result = agent_with_tools._check_and_force_tool_call(query)
                assert result is not None, f"Query '{query}' should trigger force call"


# ====== 测试：ReAct 循环判断 ======
class TestShouldContinue:
    """测试 ReAct 循环判断"""

    def test_continue_when_tool_calls_exist(self, agent_with_tools):
        """有工具调用时继续"""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "calculator", "args": {"expression": "1+2"}, "id": "call_1"}]
                )
            ]
        }
        result = agent_with_tools._should_continue(state)
        assert result == "continue"

    def test_end_when_no_tool_calls(self, agent_with_tools):
        """无工具调用时结束"""
        state = {
            "messages": [AIMessage(content="这是最终回答")]
        }
        result = agent_with_tools._should_continue(state)
        assert result == "end"

    def test_end_when_empty_messages(self, agent_with_tools):
        """空消息列表时结束"""
        state = {"messages": []}
        result = agent_with_tools._should_continue(state)
        assert result == "end"


# ====== 测试：系统提示词构建 ======
class TestSystemPrompt:
    """测试系统提示词构建"""

    def test_prompt_with_tools_contains_tool_list(self, agent_with_tools):
        """有工具时提示词包含工具列表"""
        prompt = agent_with_tools._load_system_prompt()
        assert "【可用工具】" in prompt
        assert "【关键规则" in prompt

    def test_prompt_without_tools_fallback(self, agent_without_tools):
        """无工具时提示词降级"""
        prompt = agent_without_tools._load_system_prompt()
        assert "没有绑定任何工具" in prompt

    def test_prompt_contains_forbidden_rules(self, agent_with_tools):
        """提示词包含禁止行为规则"""
        prompt = agent_with_tools._load_system_prompt()
        assert "禁止" in prompt

    def test_prompt_contains_no_fabrication_rule(self, agent_with_tools):
        """提示词包含禁止编造来源规则"""
        prompt = agent_with_tools._load_system_prompt()
        assert "编造" in prompt or "基于工具返回" in prompt

    def test_prompt_forces_calculator_for_math(self, agent_with_tools):
        """提示词要求计算问题必须用计算器"""
        prompt = agent_with_tools._load_system_prompt()
        assert "calculator" in prompt.lower() or "计算" in prompt

    def test_prompt_requires_internal_thinking(self, agent_with_tools):
        """提示词要求思考过程不输出"""
        prompt = agent_with_tools._load_system_prompt()
        assert "思考过程" in prompt or "内部思考" in prompt

    def test_prompt_prohibits_thought_output(self, agent_with_tools):
        """提示词禁止输出 Thought/Action/Observation"""
        prompt = agent_with_tools._load_system_prompt()
        assert "禁止在回答中写出" in prompt or "不要在回答中" in prompt


# ====== 测试：知识库工具检测 ======
class TestHasKnowledgeSearchTool:
    """测试知识库工具检测"""

    def test_has_knowledge_search_tool(self, agent_with_tools):
        """有 knowledge_search 工具"""
        result = agent_with_tools._has_knowledge_search_tool()
        assert result is False

    def test_no_knowledge_search_tool(self, agent_without_tools):
        """无 knowledge_search 工具"""
        result = agent_without_tools._has_knowledge_search_tool()
        assert result is False


# ====== 测试：Agent 初始化 ======
class TestAgentInit:
    """测试 Agent 初始化"""

    def test_init_with_pre_loaded_tools(self, mock_llm, mock_tools):
        """使用预加载工具初始化"""
        with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
            agent = SmartAgent(
                llm_provider="qwen",
                pre_loaded_tools=mock_tools,
            )
        assert len(agent.tools) == len(mock_tools)

    def test_init_without_agent_id(self, mock_llm, mock_tools):
        """无 agent_id 时使用默认工具"""
        with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
            agent = SmartAgent(
                llm_provider="qwen",
                pre_loaded_tools=mock_tools,
            )
        assert agent.agent_id is None

    def test_init_with_agent_id(self, mock_llm, mock_tools):
        """有 agent_id"""
        with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
            agent = SmartAgent(
                llm_provider="qwen",
                pre_loaded_tools=mock_tools,
                agent_id="custom-agent",
            )
        assert agent.agent_id == "custom-agent"

    def test_permission_level_default(self, mock_llm, mock_tools):
        """默认权限级别为 safe"""
        with patch("backend.app.agent.agent.model_factory.get_llm_model", return_value=mock_llm):
            agent = SmartAgent(
                llm_provider="qwen",
                pre_loaded_tools=mock_tools,
            )
        assert agent.permission_level == "safe"


# ====== 测试：Agent 执行主流程 ======
class TestAgentRun:
    """测试 Agent 执行主流程"""

    @pytest.mark.asyncio
    async def test_run_with_tools(self, agent_with_tools):
        """带工具的 Agent 执行"""
        result = await agent_with_tools.run(
            query="你好",
            user_id="test_user",
            session_id="test_session",
        )

        assert "answer" in result
        assert "called_tools" in result
        assert "usage_metadata" in result

    @pytest.mark.asyncio
    async def test_run_without_tools(self, agent_without_tools):
        """无工具的 Agent 执行"""
        result = await agent_without_tools.run(
            query="你好",
            user_id="test_user",
            session_id="test_session",
        )

        assert "answer" in result
        assert result["called_tools"] == []

    @pytest.mark.asyncio
    async def test_run_with_history(self, agent_with_tools):
        """带对话历史的执行"""
        history = [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]

        result = await agent_with_tools.run(
            query="新问题",
            user_id="test_user",
            session_id="test_session",
            history=history,
        )

        assert "answer" in result


# ====== 测试：工具结果捕获节点 ======
class TestCapturingToolNode:
    """测试工具结果捕获节点"""

    @pytest.mark.asyncio
    async def test_tool_not_found_degradation(self):
        """工具不存在时降级处理"""
        from backend.app.agent.agent import CapturingToolNode

        node = CapturingToolNode([])

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "nonexistent_tool",
                        "args": {"param": "value"},
                        "id": "call_123"
                    }]
                )
            ],
            "tool_results": [],
        }

        result = await node.ainvoke(state)

        assert "messages" in result
        assert "tool_results" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_tool_results_tracking(self):
        """工具结果跟踪"""
        from backend.app.agent.agent import CapturingToolNode

        node = CapturingToolNode([])

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "unknown",
                        "args": {},
                        "id": "track_me"
                    }]
                )
            ],
            "tool_results": [],
        }

        result = await node.ainvoke(state)

        tool_results = result.get("tool_results", [])
        tracked = [t for t in tool_results if t.get("tool_call_id") == "track_me"]
        assert len(tracked) == 1
        assert tracked[0]["status"] == "error"


# ====== 测试：工具信息获取 ======
class TestGetToolsInfo:
    """测试工具信息获取"""

    def test_get_tools_info(self, agent_with_tools):
        """获取工具信息"""
        info = agent_with_tools.get_tools_info()
        assert isinstance(info, list)

    def test_get_tools_info_without_tools(self, agent_without_tools):
        """无工具时返回空列表"""
        info = agent_without_tools.get_tools_info()
        assert info == []