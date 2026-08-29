# backend/tests/test_router.py
"""
智能路由器单元测试
测试规则匹配、LLM 分析解析、路由决策等纯逻辑部分
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.agent.router import IntelligentRouter


@pytest.fixture
def router():
    """创建路由器实例（不初始化 LLM）"""
    r = IntelligentRouter.__new__(IntelligentRouter)
    r.default_provider = "test"
    r._llm = None
    return r


class TestChatKeywordMatching:
    """测试快速规则匹配（chat 关键词）"""

    def test_chat_keyword_hi(self, router):
        result = router._try_chat_keywords("hi")
        assert result is not None
        assert result["mode"] == "chat"

    def test_chat_keyword_hello(self, router):
        result = router._try_chat_keywords("hello")
        assert result is not None
        assert result["mode"] == "chat"

    def test_chat_keyword_chinese(self, router):
        result = router._try_chat_keywords("你好")
        assert result is not None
        assert result["mode"] == "chat"

    def test_chat_keyword_thanks(self, router):
        result = router._try_chat_keywords("谢谢")
        assert result is not None
        assert result["mode"] == "chat"

    def test_long_query_not_chat(self, router):
        """长查询即使包含关键词也不快速匹配 chat"""
        result = router._try_chat_keywords("你好，请问公司的请假流程是什么")
        assert result is None  # 长度 > 20

    def test_empty_query(self, router):
        result = router._try_chat_keywords("")
        assert result is None

    def test_non_chat_query(self, router):
        result = router._try_chat_keywords("计算 125*8")
        assert result is None

    def test_query_with_chat_keyword_but_long(self, router):
        """包含关键词但长度超过限制时返回 None"""
        result = router._try_chat_keywords("谢谢，请帮我查询一下数据")
        assert result is None


class TestParseLLMAnalysis:
    """测试 LLM XML 标签解析"""

    def test_parse_all_true(self, router):
        raw = """<needs_tools>true</needs_tools>
<needs_context>true</needs_context>
<tools_suitable>true</tools_suitable>
<reason>需要查询数据并结合上下文</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is True
        assert result["needs_context"] is True
        assert result["tools_suitable"] is True
        assert result["reason"] == "需要查询数据并结合上下文"

    def test_parse_all_false(self, router):
        raw = """<needs_tools>false</needs_tools>
<needs_context>false</needs_context>
<tools_suitable>na</tools_suitable>
<reason>闲聊无需工具</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is False
        assert result["needs_context"] is False
        assert result["tools_suitable"] is True  # 不需要工具时默认为 True
        assert result["reason"] == "闲聊无需工具"

    def test_parse_needs_tools_no_match(self, router):
        raw = """<needs_tools>true</needs_tools>
<needs_context>false</needs_context>
<tools_suitable>false</tools_suitable>
<reason>需要数据但无合适工具</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is True
        assert result["needs_context"] is False
        assert result["tools_suitable"] is False

    def test_parse_case_insensitive(self, router):
        raw = """<NEEDS_TOOLS>TRUE</NEEDS_TOOLS>
<NEEDS_CONTEXT>FALSE</NEEDS_CONTEXT>
<TOOLS_SUITABLE>TRUE</TOOLS_SUITABLE>
<REASON>测试大小写</REASON>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is True
        assert result["needs_context"] is False

    def test_parse_with_whitespace(self, router):
        raw = """<needs_tools>  true  </needs_tools>
<needs_context>
  false
</needs_context>
<tools_suitable>true</tools_suitable>
<reason>含空白</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is True
        assert result["needs_context"] is False

    def test_parse_invalid_xml(self, router):
        result = router._parse_llm_analysis("这不是有效的 XML 标签")
        assert result is not None  # 返回默认值
        assert result["needs_tools"] is False
        assert result["needs_context"] is False

    def test_parse_partial_xml(self, router):
        """部分标签缺失时使用默认值"""
        raw = """<needs_tools>true</needs_tools>
<reason>只解析部分</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert result["needs_tools"] is True
        assert result["needs_context"] is False  # 默认值

    def test_parse_reason_truncated(self, router):
        """reason 超过 50 字符被截断"""
        long_reason = "这是一个非常非常非常非常非常非常非常非常长的判断理由"
        raw = f"""<needs_tools>false</needs_tools>
<needs_context>false</needs_context>
<tools_suitable>na</tools_suitable>
<reason>{long_reason}</reason>"""
        result = router._parse_llm_analysis(raw)
        assert result is not None
        assert len(result["reason"]) <= 50


class TestBuildNoToolsPrompt:
    """测试无工具降级提示词构建"""

    def test_build_prompt_no_tools(self, router):
        prompt = router._build_no_tools_system_prompt("需要营收数据", [])
        assert "诚实的 AI 助手" in prompt
        assert "需要营收数据" in prompt
        assert "不要编造数据" in prompt

    def test_build_prompt_with_tools(self, router):
        tools = [{"name": "calculator", "description": "计算器"}]
        prompt = router._build_no_tools_system_prompt("需要数据", tools)
        assert "calculator" in prompt

    def test_build_prompt_content(self, router):
        prompt = router._build_no_tools_system_prompt("测试原因", [])
        assert "坦诚告知" in prompt
        assert "通用知识" in prompt
        assert "不要编造" in prompt


class TestRouteIntegration:
    """路由集成测试（可测试部分）"""

    @pytest.mark.asyncio
    async def test_quick_chat_keyword_routes_chat(self, router):
        """chat 关键词快速路径"""
        result = await router.route("你好")
        assert result["mode"] == "chat"
        assert result["confidence"] >= 0.9
        assert result["needs_tools"] is False
        assert result["needs_context"] is False
        assert result["chat_degradation"] is False

    @pytest.mark.asyncio
    async def test_long_query_not_quick_chat(self, router):
        """长查询不走快速 chat 路径（走 LLM 分析）"""
        result = await router.route("你好，请问公司的请假流程是什么？")
        # 结果取决于 LLM，但结构正确
        assert "mode" in result
        assert "reason" in result
        assert "needs_tools" in result
        assert "needs_context" in result
        assert "tools_suitable" in result
        assert "chat_degradation" in result
        assert "system_prompt_override" in result