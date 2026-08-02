# backend/app/agent/router.py
"""
智能路由器
根据用户问题自动选择最合适的处理模式：
- traditional: 传统 RAG（简单知识查询）
- agent: 单 Agent + Tool（需要工具调用）
"""

from typing import Dict, Any, Literal,Optional
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.utils.models_factory import model_factory
from backend.app.common.logger import logger
from backend.app.core.config import settings
from backend.app.utils.vector_store import vector_store_util


class IntelligentRouter:
    """
    智能路由器

    工作原理：
    1. 用一个轻量 LLM 调用判断问题复杂度
    2. 根据复杂度选择处理模式
    3. 兜底规则：判断失败时走 Agent 模式
    """

    ROUTE_PROMPT = """你是模式路由器。请分析用户问题，判断应该用哪种模式处理。

【三种模式】
1. chat: 纯聊天模式
   - 适用：问候、闲聊、不需要知识库或工具
   - 特点：最快，直接LLM回复，不走RAG检索
   - 示例："hi" "你好" "谢谢" "你是谁"

2. traditional: 传统RAG模式
   - 适用：简单的知识查询、文档检索
   - 特点：速度快，只做检索+生成
   - 示例："公司请假流程是什么？" "Python代码规范是什么？"

3. agent: 单Agent模式
   - 适用：需要工具调用（计算器、文件读取）
   - 特点：LLM自主判断何时调用工具
   - 示例："125*8等于多少？" "帮我读取员工手册"

【判断规则】
- 纯闲聊、问候、感谢、身份询问 → chat
- 只需检索知识库 → traditional
- 需要工具但步骤简单 → agent

【输出格式】只输出一个单词：chat / traditional / agent"""

    def __init__(self, llm_provider: str = None):
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER
        self.default_provider = llm_provider
        self._llm = None
    
    def _get_llm(self, llm_provider: Optional[str] = None):
        provider = llm_provider or self.default_provider
        if provider == self.default_provider and self._llm:
            return self._llm
        self._llm = model_factory.get_llm_model(provider=provider, temperature=0.0)
        return self._llm

    async def route(
        self,
        query: str,
        history: list = None,
        llm_provider: str = None,
    ) -> Dict[str, Any]:
        """
        智能路由选择

        Args:
            query: 用户问题
            history: 对话历史（可选，帮助判断上下文）

        Returns:
            {
                "mode": "chat" / "traditional" / "agent",
                "confidence": float,
                "reason": str,
            }
        """
        # 快速规则匹配：纯聊天模式（跳过 LLM 判断，提高响应速度）
        chat_keywords = ["hi", "hello", "你好", "嗨", "在吗", "早上好", "晚上好", "晚安", "谢谢", "谢谢啦", "你是谁", "你叫什么", "你好啊", "嗨喽", "哈喽", "再见", "拜拜", "嗯", "好的", "可以", "行", "没问题", "是的", "没错", "对的", "知道了", "明白"]
        query_lower = query.lower().strip()
        if any(kw in query_lower for kw in chat_keywords) and len(query) <= 20:
            return {
                "mode": "chat",
                "confidence": 0.95,
                "reason": f"快速规则匹配（关键词: {query_lower}）",
            }

        # 快速规则匹配：计算模式
        calc_keywords = ["算", "计算", "等于", "+", "-", "*", "/", "×", "÷", "%", "加", "减", "乘", "除", "平方", "立方", "次方"]
        if any(kw in query_lower for kw in calc_keywords):
            return {
                "mode": "agent",
                "confidence": 0.85,
                "reason": f"快速规则匹配（计算关键词: {query_lower}）",
            }

        # 知识库状态感知：如果知识库为空，跳过 traditional/RAG 模式，直接走 chat
        # 使用轻量级检查（不加载embedding模型）
        try:
            kb_empty = vector_store_util.is_collection_empty()
            if kb_empty:
                logger.info(f"知识库为空，跳过 traditional 模式，路由到 chat")
                return {
                    "mode": "chat",
                    "confidence": 0.80,
                    "reason": "知识库为空，跳过RAG检索",
                }
        except Exception as e:
            logger.warning(f"检查知识库状态失败，继续正常路由: {e}")

        # 构建历史上下文
        history_text = ""
        if history:
            recent = history[-4:]
            history_text = "\n".join([
                f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:100]}"
                for m in recent
            ])

        prompt = f"""请判断以下问题应该用哪种模式处理。

【对话历史】
{history_text if history_text else '（无）'}

【当前问题】
{query}

只输出一个单词：chat / traditional / agent"""

        try:
            llm = self._get_llm(llm_provider)
            response = await llm.ainvoke([
                SystemMessage(content=self.ROUTE_PROMPT),
                HumanMessage(content=prompt),
            ])

            decision = response.content.strip().lower()

            # 解析结果
            mode = self._parse_decision(decision)
            confidence = self._estimate_confidence(query, mode)

            return {
                "mode": mode,
                "confidence": confidence,
                "reason": f"LLM判断: {decision}",
            }

        except Exception as e:
            logger.warning(f"智能路由判断失败，使用兜底规则: {e}")
            # 兜底：用规则判断
            mode = self._fallback_route(query)
            return {
                "mode": mode,
                "confidence": 0.5,
                "reason": f"兜底规则（LLM判断失败）",
            }

    def _parse_decision(self, text: str) -> str:
        """解析 LLM 输出"""
        text = text.lower().strip()
        if "chat" in text:
            return "chat"
        if "traditional" in text:
            return "traditional"
        if "agent" in text:
            return "agent"
        return "chat"

    def _estimate_confidence(self, query: str, mode: str) -> float:
        """估算置信度"""
        chat_keywords = ["你好", "hi", "hello", "嗨", "谢谢", "你是谁", "在吗"]
        calc_keywords = ["算", "计算", "乘", "除", "加", "减", "等于"]
        retrieve_keywords = ["流程", "规范", "是什么", "怎么", "查询"]

        if mode == "chat":
            matches = sum(1 for kw in chat_keywords if kw.lower() in query.lower())
            return min(0.7 + matches * 0.1, 0.95)
        elif mode == "traditional":
            matches = sum(1 for kw in retrieve_keywords if kw in query)
            return min(0.6 + matches * 0.1, 0.95)
        elif mode == "agent":
            matches = sum(1 for kw in calc_keywords if kw in query)
            return min(0.6 + matches * 0.1, 0.95)
        return 0.5

    def _fallback_route(self, query: str) -> str:
        """
        兜底路由规则（不依赖 LLM）
        基于关键词匹配
        """
        chat_signals = ["你好", "hi", "hello", "嗨", "谢谢", "你是谁", "在吗", "早上好", "晚上好"]
        chat_count = sum(1 for s in chat_signals if s.lower() in query.lower())

        agent_signals = ["算", "计算", "乘", "除", "加", "减", "读取", "打开文件"]
        agent_count = sum(1 for s in agent_signals if s in query)

        trad_signals = ["是什么", "流程", "规范", "怎么", "查询", "有没有"]
        trad_count = sum(1 for s in trad_signals if s in query)

        if chat_count >= 1 and len(query) <= 10:
            return "chat"
        if agent_count >= 1:
            return "agent"
        if trad_count >= 1:
            return "traditional"
        return "chat"


# ====== 全局实例（延迟初始化）======

_intelligent_router_instance = None

def get_intelligent_router(llm_provider: str = None):
    global _intelligent_router_instance
    if _intelligent_router_instance is None:
        _intelligent_router_instance = IntelligentRouter(llm_provider=llm_provider)
    elif llm_provider is not None and _intelligent_router_instance.default_provider != llm_provider:
        _intelligent_router_instance = IntelligentRouter(llm_provider=llm_provider)
    return _intelligent_router_instance

class _IntelligentRouterProxy:
    def __getattr__(self, name):
        return getattr(get_intelligent_router(), name)

intelligent_router = _IntelligentRouterProxy()