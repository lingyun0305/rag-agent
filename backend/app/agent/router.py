# backend/app/agent/router.py
"""
智能路由器

路由决策流程（LLM 全维度判断 + 风险信号兜底）：
1. 简单问候/闲聊 → chat 模式（快速规则，节省一次 LLM 调用）
2. LLM 全维度分析：
   - 需要工具 + 有合适工具 → agent 模式
   - 需要上下文 → agent 模式
   - 需要工具 + 无合适工具 → chat 模式 + LLM 解释限制
   - 都不需要 → chat 模式
3. 兜底原则（不在"不确定"时剥夺工具能力）：
   - LLM 分析异常/解析失败 → 判为需要工具+上下文 → agent（有工具时）
   - LLM 判"都不需要"但命中风险信号（指代承接/短问句+历史/数据诉求）→ agent
   - 仅"运行时异常"（LLM 调用挂了等）才退回 chat，避免二次超时把用户彻底卡死
   原因：agent 是 chat 的能力超集（它可以选择不调工具直接回答），
   而 chat 一旦被选中就永久失去知识库检索与工具能力，
   这正是线上"问数据却直接拒答/不查知识库"的根因。
"""

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.common.logger import logger
from backend.app.core.config import settings
from backend.app.utils.models_factory import model_factory
from backend.app.utils.roles import is_user_role, role_label_cn

# ====== 风险信号词表（用于兜底纠正 7B 的路由误判）======

# 承接/指代词：出现即说明这句话脱离上文无法理解
CONTEXT_REF_WORDS = (
    "那",
    "这",
    "它",
    "他",
    "她",
    "上面",
    "上述",
    "刚才",
    "之前",
    "前面",
    "刚才说",
    "接着",
    "继续",
    "还有",
    "呢",
    "换成",
    "改为",
    "同样的",
    "排第几",
    "第几",
    "第一个",
    "第二个",
)

# 数据/计算/内部资料诉求：出现即说明应当具备检索或计算能力
DATA_SEEK_WORDS = (
    "多少",
    "几",
    "多少万",
    "排名",
    "排行",
    "排序",
    "统计",
    "汇总",
    "合计",
    "总和",
    "计算",
    "算一下",
    "帮我算",
    "占比",
    "比例",
    "增长率",
    "同比",
    "环比",
    "平均",
    "人均",
    "总额",
    "总营收",
    "对比",
    "差多少",
    "谁更",
    "哪个更",
    "查询",
    "查一下",
    "查一查",
    "制度",
    "规范",
    "流程",
    "规定",
    "政策",
    "报告",
    "数据表",
    "表格",
)


class IntelligentRouter:
    """
    智能路由器

    核心设计：LLM 全维度分析
    1. 判断是否需要外部工具（数据查询、计算等）
    2. 判断是否依赖对话上下文（指代前文、多轮推理等）
    3. 判断可用工具是否能满足需求
    """

    def __init__(self, llm_provider: str = None):
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER
        self.default_provider = llm_provider
        self._llm = None

    def _get_llm(self, llm_provider: str | None = None):
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
        agent_id: str | None = None,
        db: Any | None = None,
    ) -> dict[str, Any]:
        """
        智能路由选择

        决策流程：
        1. 简单问候 → chat（快速规则）
        2. LLM 全维度分析 → 按结果路由

        Args:
            query: 用户问题
            history: 对话历史 [{role, content}, ...]
            llm_provider: 模型提供商
            agent_id: Agent ID
            db: 数据库会话

        Returns:
            {
                "mode": "chat" / "agent",
                "confidence": float,
                "reason": str,
                "needs_tools": bool,
                "needs_context": bool,
                "tools_suitable": bool,
                "chat_degradation": bool,  # True 表示需要工具但无工具可用
                "system_prompt_override": str  # chat 模式降级时的系统提示
            }
        """
        query_lower = query.lower().strip()

        # ===== 规则 1：简单问候 → chat（快速路径）=====
        chat_result = self._try_chat_keywords(query_lower)
        if chat_result:
            chat_result.update(
                {
                    "needs_tools": False,
                    "needs_context": False,
                    "tools_suitable": False,
                    "chat_degradation": False,
                    "system_prompt_override": None,
                }
            )
            return chat_result

        # ===== 规则 2：获取可用工具列表 =====
        available_tools = await self._get_agent_tool_list(agent_id, db)

        # ===== 规则 3：LLM 全维度分析 =====
        llm_analysis = await self._llm_analyze_query(
            query=query,
            history=history,
            available_tools=available_tools,
            llm_provider=llm_provider,
        )

        needs_tools = llm_analysis.get("needs_tools", False)
        needs_context = llm_analysis.get("needs_context", False)
        tools_suitable = llm_analysis.get("tools_suitable", False)
        reason = llm_analysis.get("reason", "LLM 分析完成")

        logger.info(
            f"LLM 路由分析: query='{query[:50]}' "
            f"needs_tools={needs_tools} needs_context={needs_context} "
            f"tools_suitable={tools_suitable} → {reason}"
        )

        # ===== 规则 4：根据分析结果路由 =====
        # 4a. 需要工具 + 有合适工具 → agent
        if needs_tools and tools_suitable and available_tools:
            return {
                "mode": "agent",
                "confidence": 0.85,
                "reason": f"需要工具且有合适工具可用: {reason}",
                "needs_tools": True,
                "needs_context": needs_context,
                "tools_suitable": True,
                "chat_degradation": False,
                "system_prompt_override": None,
            }

        # 4b. 需要上下文 → agent（agent 可利用多轮对话历史推理）
        if needs_context:
            return {
                "mode": "agent",
                "confidence": 0.75,
                "reason": f"需要上下文进行多轮推理: {reason}",
                "needs_tools": needs_tools,
                "needs_context": True,
                "tools_suitable": tools_suitable,
                "chat_degradation": False,
                "system_prompt_override": None,
            }

        # 4c. 需要工具但无合适工具 → chat 降级
        if needs_tools and (not tools_suitable or not available_tools):
            degradation_prompt = self._build_no_tools_system_prompt(reason, available_tools)
            return {
                "mode": "chat",
                "confidence": 0.70,
                "reason": f"需要工具但无合适工具: {reason}",
                "needs_tools": True,
                "needs_context": False,
                "tools_suitable": False,
                "chat_degradation": True,
                "system_prompt_override": degradation_prompt,
            }

        # 4d. 风险信号兜底：LLM 判"都不需要"，但问题显然依赖上文或需要数据 →
        #     不在"不确定"时把用户降到 chat（chat 会直接失去知识库/工具能力）
        risk = self._detect_risk_signals(query, history)
        if risk and available_tools:
            return {
                "mode": "agent",
                "confidence": 0.60,
                "reason": f"风险信号兜底（{risk}）→ 交 Agent 自主判断: {reason}",
                "needs_tools": True,
                "needs_context": True,
                "tools_suitable": True,
                "chat_degradation": False,
                "system_prompt_override": None,
            }

        # 4e. 不需要工具/上下文 → chat
        return {
            "mode": "chat",
            "confidence": 0.80,
            "reason": f"无需工具和上下文: {reason}",
            "needs_tools": False,
            "needs_context": False,
            "tools_suitable": False,
            "chat_degradation": False,
            "system_prompt_override": None,
        }

    def _detect_risk_signals(self, query: str, history: list | None = None) -> str | None:
        """
        检测"不该被判成纯聊天"的风险信号，命中则返回描述（否则 None）。

        三类信号（都来自线上真实误判）：
        1. 承接/指代词（"那华北排第几？"）——脱离上文无法理解，必须有历史 + 推理能力；
        2. 短问句 + 存在历史——7B 常把"那 X 呢"这类短句判成闲聊；
        3. 数据/内部资料诉求（多少/第几/统计/制度/流程…）——必须有检索或计算能力，
           直接 chat 会造成"明明有知识库却直接拒答"。
        """
        q = (query or "").strip()
        if not q:
            return None

        signals: list[str] = []

        # 是否存在可用的对话历史（role 可能是 user/human，统一走 roles 工具判定）
        has_history = any(is_user_role((m or {}).get("role")) for m in (history or []))

        if has_history and any(w in q for w in CONTEXT_REF_WORDS):
            signals.append("指代/承接上文的问句")

        if has_history and len(q) <= 12:
            signals.append("短问句且存在历史对话")

        if any(w in q for w in DATA_SEEK_WORDS):
            signals.append("含数据/计算/内部资料诉求关键词")

        # 含阿拉伯数字或百分号通常意味着要做数（"华东营收 1280 万排第几"）
        if re.search(r"\d", q):
            signals.append("问句含具体数值")

        return "；".join(signals) if signals else None

    def _build_no_tools_system_prompt(self, reason: str, available_tools: list) -> str:
        """
        构建"需要工具但无工具"降级场景的系统提示词

        当用户问题需要特定工具能力但系统没有可用工具时，
        指导 LLM 诚实告知用户限制，并基于已有知识尽可能回答。
        """
        tools_hint = ""
        if available_tools:
            tool_names = [t["name"] for t in available_tools[:5]]
            tools_hint = f"\n\n当前可用的工具包括: {', '.join(tool_names)}。如果用户的问题不在这些工具能力范围内，请诚实告知。"

        return f"""你是一个诚实的 AI 助手。

用户的问题需要某些工具或外部数据来完整回答，但当前你可能没有合适的工具可用。

请遵循以下原则：
1. 坦诚告知用户：你可能无法通过工具获取所需的精确数据
2. 基于你的通用知识尽可能提供有用的信息和建议
3. 明确指出哪些部分需要额外的数据或工具支持
4. 不要编造数据或假装使用了工具
5. 建议用户可以提供哪些额外信息来获得更好的回答{tools_hint}

具体限制原因：{reason}"""

    def _try_chat_keywords(self, query_lower: str) -> dict[str, Any] | None:
        """
        快速匹配纯聊天场景（问候/寒暄）。

        收紧规则（原实现是 `任意关键词 in query` 且长度 <=20，会把
        "你好，公司的报销制度是什么" 这种真问题吞掉，直接走 chat → 不查知识库）：
        1. 去掉标点/空白后必须"几乎只有问候语"（归一化长度 <= 6）；
        2. 一旦命中数据/计算/内部资料诉求关键词，绝不判闲聊。
        """
        chat_keywords = [
            "hi",
            "hello",
            "你好",
            "嗨",
            "在吗",
            "早上好",
            "晚上好",
            "晚安",
            "谢谢",
            "谢谢啦",
            "你是谁",
            "你叫什么",
            "你好啊",
            "嗨喽",
            "哈喽",
            "再见",
            "拜拜",
            "嗯",
            "好的",
            "可以",
            "行",
            "没问题",
            "是的",
            "没错",
            "对的",
            "知道了",
            "明白",
            "不错",
            "真棒",
            "哈哈",
            "呵呵",
        ]
        # 归一化：去掉标点、空白，只留字符
        normalized = re.sub(r"[\s\.,;:!?，。；：！？、~～\"'“”‘’()（）]", "", query_lower)

        # 命中数据/资料诉求关键词 → 不是闲聊
        if any(w in normalized for w in DATA_SEEK_WORDS):
            return None

        if len(normalized) > 6:
            return None

        if any(
            normalized == kw or (kw in normalized and len(normalized) <= 4) for kw in chat_keywords
        ):
            return {
                "mode": "chat",
                "confidence": 0.95,
                "reason": f"快速规则匹配（纯问候/寒暄: {normalized}）",
            }
        return None

    async def _get_agent_tool_list(
        self,
        agent_id: str | None,
        db: Any | None,
    ) -> list[dict[str, str]]:
        """
        获取 Agent 绑定的所有工具列表

        Returns:
            [{"name": "calculator", "description": "数学计算器"}, ...]
            空列表表示没有绑定任何工具
        """
        if not agent_id or not db:
            return []

        try:
            from sqlalchemy import select

            from backend.app.models.agent import Agent, AgentTool
            from backend.app.models.tool import Tool

            tools_info = []

            # 1. 系统工具（直接存在 Agent.system_tool_names 中）
            agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent_obj = agent_result.scalar_one_or_none()
            if agent_obj:
                system_tool_names = agent_obj.get_system_tools()
                for tool_name in system_tool_names:
                    from backend.app.agent.tools.registry import tool_registry

                    tool_obj = tool_registry.get(tool_name)
                    if tool_obj:
                        tools_info.append(
                            {
                                "name": tool_obj.name,
                                "description": tool_obj.description,
                            }
                        )
                    else:
                        db_tool_result = await db.execute(
                            select(Tool).where(Tool.name == tool_name, Tool.status == "active")
                        )
                        db_tool = db_tool_result.scalar_one_or_none()
                        if db_tool:
                            tools_info.append(
                                {
                                    "name": db_tool.name,
                                    "description": db_tool.desc or "",
                                }
                            )

            # 2. 自定义工具（通过 AgentTool 关联表）
            agent_tools_result = await db.execute(
                select(AgentTool).where(AgentTool.agent_id == agent_id)
            )
            agent_tools = agent_tools_result.scalars().all()

            if agent_tools:
                tool_ids = [t.tool_id for t in agent_tools]
                db_tools_result = await db.execute(
                    select(Tool).where(Tool.id.in_(tool_ids), Tool.status == "active")
                )
                db_tools = db_tools_result.scalars().all()

                for db_tool in db_tools:
                    tools_info.append(
                        {
                            "name": db_tool.name,
                            "description": db_tool.desc or "",
                        }
                    )

            logger.info(
                f"Agent '{agent_id}' 绑定了 {len(tools_info)} 个工具: {[t['name'] for t in tools_info]}"
            )
            return tools_info

        except Exception as e:
            logger.warning(f"获取 Agent 工具列表失败: {e}")
            return []

    async def _llm_analyze_query(
        self,
        query: str,
        history: list[dict] | None = None,
        available_tools: list[dict[str, str]] | None = None,
        llm_provider: str = None,
    ) -> dict[str, Any]:
        """
        LLM 全维度分析用户问题

        分析维度：
        1. needs_tools: 是否需要外部工具（数据查询、计算、外部信息等）
        2. needs_context: 是否依赖对话上下文（指代、多轮推理等）
        3. tools_suitable: 可用工具是否能满足需求

        使用 XML 标签格式（比 JSON 更适合 7B 模型解析）
        """
        available_tools = available_tools or []

        # 构建工具列表描述
        if available_tools:
            tool_desc_lines = []
            for i, t in enumerate(available_tools, 1):
                tool_desc_lines.append(f"  {i}. {t['name']}: {t['description']}")
            tools_text = "\n".join(tool_desc_lines)
        else:
            tools_text = "  (当前无可用工具)"

        # 构建对话历史摘要
        history_text = ""
        if history:
            recent = history[-6:] if len(history) > 6 else history
            history_lines = []
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                prefix = role_label_cn(role)
                history_lines.append(f"  {prefix}: {content}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (无对话历史)"

        prompt = f"""你是一个路由分析专家。请从以下三个维度分析用户问题，然后用 XML 标签输出结论。

【可用工具列表】
{tools_text}

【对话历史摘要】
{history_text}

【用户问题】
{query}

【分析维度说明】

1. needs_tools（是否需要工具）:
   - true: 问题需要外部工具能力（如数据查询、数值计算、知识库检索、外部API等）
   - false: 问题可以直接用常识、对话理解或通用知识回答
   - 判断要点：涉及具体数据/数值/外部信息 → true；闲聊/常识/解释 → false

2. needs_context（是否需要上下文）:
   - true: 问题依赖对话历史（如指代前文"他""这个""刚才的问题"，或需要多轮推理）
   - false: 问题独立完整，无需参考之前的对话
   - 判断要点：出现指代词（他/她/它/这个/那个/刚才/之前/上面）→ 检查是否真的有上下文依赖

3. tools_suitable（可用工具是否匹配需求）:
   - true: 需要工具，且可用工具列表中有能满足需求的工具
   - false: 需要工具，但可用工具列表中没有能满足需求的工具
   - na: 不需要工具（此时此字段无意义）
   - 判断要点：对比用户需求与工具描述，看工具能力是否覆盖需求

【输出格式】
严格按以下 XML 标签格式输出，不要添加任何其他内容：
<needs_tools>true或false</needs_tools>
<needs_context>true或false</needs_context>
<tools_suitable>true或false或na</tools_suitable>
<reason>简短的判断理由（不超过30字）</reason>"""

        llm = self._get_llm(llm_provider)

        for attempt in range(2):
            try:
                response = await llm.ainvoke(
                    [
                        SystemMessage(
                            content="你是路由分析专家。严格按 XML 标签格式输出，不要添加额外内容。"
                        ),
                        HumanMessage(content=prompt),
                    ]
                )

                raw_output = response.content.strip()

                # 解析 XML 标签
                result = self._parse_llm_analysis(raw_output)

                if result is not None:
                    logger.info(
                        f"LLM 路由分析成功: needs_tools={result['needs_tools']} "
                        f"needs_context={result['needs_context']} "
                        f"tools_suitable={result['tools_suitable']}"
                    )
                    return result

                # 解析失败，重试一次
                logger.warning(
                    f"LLM 路由分析输出解析失败 (尝试 {attempt + 1}/2): {raw_output[:100]}"
                )

            except Exception as e:
                logger.warning(f"LLM 路由分析异常 (尝试 {attempt + 1}/2): {e}")

        # 兜底：保守判断（宁可选能力更强的 agent，也不要误降到 chat 而失去工具能力）
        logger.warning("LLM 路由分析全部失败，兜底保守判断 → 倾向 agent")
        return {
            "needs_tools": True,
            "needs_context": True,
            "tools_suitable": bool(available_tools),
            "reason": "LLM 分析失败，保守判断需要工具与上下文",
        }

    def _parse_llm_analysis(self, raw_output: str) -> dict[str, Any] | None:
        """
        解析 LLM 输出的 XML 标签

        支持的格式：
        <needs_tools>true</needs_tools>
        <needs_context>false</needs_context>
        <tools_suitable>na</tools_suitable>
        <reason>无需工具</reason>

        容错：提取 <tag>value</tag> 格式，忽略大小写和空白
        """
        try:
            result = {
                "needs_tools": False,
                "needs_context": False,
                "tools_suitable": False,
                "reason": "LLM 输出解析完成",
            }

            # 提取 needs_tools
            m = re.search(
                r"<needs_tools>\s*(true|false)\s*</needs_tools>", raw_output, re.IGNORECASE
            )
            if m:
                result["needs_tools"] = m.group(1).lower() == "true"

            # 提取 needs_context
            m = re.search(
                r"<needs_context>\s*(true|false)\s*</needs_context>", raw_output, re.IGNORECASE
            )
            if m:
                result["needs_context"] = m.group(1).lower() == "true"

            # 提取 tools_suitable
            m = re.search(
                r"<tools_suitable>\s*(true|false|na)\s*</tools_suitable>", raw_output, re.IGNORECASE
            )
            if m:
                val = m.group(1).lower()
                if val == "true":
                    result["tools_suitable"] = True
                elif val == "false":
                    result["tools_suitable"] = False
                else:
                    result["tools_suitable"] = result["needs_tools"]

            # 提取 reason
            m = re.search(r"<reason>(.*?)</reason>", raw_output, re.IGNORECASE | re.DOTALL)
            if m:
                result["reason"] = m.group(1).strip()[:50]

            # 验证：如果不需要工具，tools_suitable 应该为 True（不影响路由）
            if not result["needs_tools"]:
                result["tools_suitable"] = True

            return result

        except Exception:
            return None


# ====== 全局实例 ======

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
