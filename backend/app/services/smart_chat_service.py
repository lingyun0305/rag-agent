# backend/app/services/smart_chat_service.py
"""
智能路由对话服务
根据用户问题自动选择最合适的处理模式
"""

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.router import intelligent_router
from backend.app.common.logger import logger
from backend.app.core.config import settings
from backend.app.services.agent_chat_service import agent_chat_service
from backend.app.services.attachment_service import chat_attachment_service
from backend.app.services.chat_history_service import chat_history_service
from backend.app.utils.models_factory import model_factory
from backend.app.utils.roles import is_assistant_role, is_user_role


class SmartChatService:
    """
    智能路由对话服务

    工作流程：
    1. 智能路由器判断问题类型
    2. 分发到对应模式处理
    3. 统一格式返回
    """

    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        collection_name: str = "default_collection",
        use_long_term_memory: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """
        智能路由对话

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            collection_name: 知识库集合
            use_long_term_memory: 是否启用长期记忆
            db: 数据库会话
            llm_provider: 模型提供商
            agent_id: Agent ID
            turn_id: 单轮对话ID（绑定本轮附件）

        Returns:
            统一格式的响应
        """
        # 0. 创建/获取会话
        if db:
            await chat_history_service.create_session(
                session_id=session_id,
                user_id=str(user_id),
                agent_id=agent_id,
                title=query[:50] if query else None,
                db=db,
            )

        # 1. 获取对话历史（用于路由上下文分析和最终回答）
        history = await chat_history_service.get_last_n_exchanges(session_id, n=6, db=db)

        # 1.5 构建本轮附件上下文（若有 turn_id）：直接注入，不依赖模型是否主动调工具
        attachment_context = await self._build_attachment_context(
            user_id, session_id, turn_id, query
        )

        # 2. 智能路由判断（传入 history 用于上下文分析）
        route_result = await intelligent_router.route(
            query=query, history=history, llm_provider=llm_provider, agent_id=agent_id, db=db
        )
        mode = route_result["mode"]
        confidence = route_result["confidence"]
        reason = route_result["reason"]
        system_prompt_override = route_result.get("system_prompt_override")
        chat_degradation = route_result.get("chat_degradation", False)

        logger.info(
            "智能路由分发",
            event="route_decision",
            query_preview=query[:50],
            mode=mode,
            confidence=round(confidence, 2),
            reason=reason,
            needs_tools=route_result.get("needs_tools"),
            needs_context=route_result.get("needs_context"),
            chat_degradation=chat_degradation,
        )

        # 3. 分发到对应模式（只支持 chat 和 agent）
        try:
            if mode == "chat":
                result = await self._run_chat(
                    query,
                    user_id,
                    session_id,
                    db,
                    llm_provider,
                    agent_id,
                    history=history,
                    system_prompt_override=system_prompt_override,
                    attachment_context=attachment_context,
                )
            elif mode == "agent":
                result = await self._run_agent(
                    query,
                    user_id,
                    session_id,
                    use_long_term_memory,
                    db,
                    llm_provider,
                    agent_id,
                    attachment_context=attachment_context,
                )
            else:
                # 未知 mode（理论上不会出现）：默认走 agent —— agent 是 chat 的能力超集，
                # 它可以选择不调工具直接回答，而 chat 一旦被选中就永久失去检索/工具能力。
                result = await self._run_agent(
                    query,
                    user_id,
                    session_id,
                    use_long_term_memory,
                    db,
                    llm_provider,
                    agent_id,
                    attachment_context=attachment_context,
                )
                mode = "agent"
        except Exception as e:
            logger.error(
                "智能路由执行失败，降级到chat",
                event="route_fallback",
                original_mode=mode,
                error=str(e),
            )
            # 运行时异常才退回 chat：此处多为 LLM 不可用等基础设施问题，
            # 再试一次 agent 大概率同样失败，而 chat 至少能保证用户拿到一条回复。
            result = await self._run_chat(query, user_id, session_id, db, llm_provider, agent_id)
            mode = "chat_fallback"

        # 3. 统一格式返回
        return {
            "answer": result.get("answer", ""),
            "mode": mode,
            "confidence": confidence,
            "route_reason": reason,
            "called_tools": result.get("called_tools", []),
            "sources": result.get("sources", []),
            "extra": result.get("extra", {}),
            "route_details": {
                "needs_tools": route_result.get("needs_tools"),
                "needs_context": route_result.get("needs_context"),
                "tools_suitable": route_result.get("tools_suitable"),
                "chat_degradation": chat_degradation,
            },
        }

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        collection_name: str = "default_collection",
        use_long_term_memory: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ):
        """
        智能路由对话（流式版）
        路由判断后，直接 yield chunk 给前端
        """
        import time

        total_start = time.time()

        # 0. 创建/获取会话
        if db:
            await chat_history_service.create_session(
                session_id=session_id,
                user_id=str(user_id),
                agent_id=agent_id,
                title=query[:50] if query else None,
                db=db,
            )

        # 1. 获取对话历史（用于路由上下文分析）
        history = await chat_history_service.get_last_n_exchanges(session_id, n=6, db=db)

        # 1.5 构建本轮附件上下文（若有 turn_id）：直接注入，不依赖模型是否主动调工具
        attachment_context = await self._build_attachment_context(
            user_id, session_id, turn_id, query
        )

        # 2. 智能路由判断（传入 history 用于上下文分析）
        route_result = await intelligent_router.route(
            query=query, history=history, llm_provider=llm_provider, agent_id=agent_id, db=db
        )
        mode = route_result["mode"]
        confidence = route_result["confidence"]
        system_prompt_override = route_result.get("system_prompt_override")
        chat_degradation = route_result.get("chat_degradation", False)

        logger.info(
            f"智能路由: query='{query[:50]}...' → mode={mode}, confidence={confidence:.2f}, degradation={chat_degradation}"
        )

        # 3. 分发到对应模式，流式返回
        #    stats 收集：把 answer 文本长度 / token 用量 / LLM 调用次数累积下来，
        #    最后写进 [对话交互] 日志（此前这些字段恒为 0，导致统计链路形同虚设）。
        stats = {"response_length": 0, "total_tokens": 0, "llm_calls": 0}
        yielded_any = False

        try:
            if mode == "agent":
                stream = self._run_agent_stream(
                    query,
                    user_id,
                    session_id,
                    use_long_term_memory,
                    db,
                    llm_provider,
                    agent_id,
                    attachment_context=attachment_context,
                )
            else:
                # chat / 未知 mode 走 chat 分支（未知 mode 已在 route() 层被兜底为 agent，
                # 这里再留一层保护，避免未知字符串导致空响应）
                stream = self._run_chat_stream(
                    query,
                    user_id,
                    session_id,
                    db,
                    llm_provider,
                    agent_id,
                    history=history,
                    system_prompt_override=system_prompt_override,
                    attachment_context=attachment_context,
                )

            async for chunk in stream:
                yielded_any = True
                self._accumulate_stream_stats(stats, chunk)
                yield chunk

        except Exception as e:
            logger.error(f"智能路由流式执行失败 (mode={mode}): {e}")
            # 已经在流中途出错时不再重试，否则用户会看到重复/拼接错乱的答案
            if not yielded_any:
                # 运行时异常退回 chat：此处多为 LLM 不可用等基础设施问题，
                # 重试 agent 同样会失败，chat 至少能保证用户拿到一条回复。
                async for chunk in self._run_chat_stream(
                    query,
                    user_id,
                    session_id,
                    db,
                    llm_provider,
                    agent_id,
                    history=history,
                    system_prompt_override=system_prompt_override,
                ):
                    self._accumulate_stream_stats(stats, chunk)
                    yield chunk

        # 3. 记录总耗时 + 本轮真实 token / 答案长度
        total_duration_ms = (time.time() - total_start) * 1000
        logger.chat_exchange(
            mode=mode,
            query=query,
            response_length=stats["response_length"],
            total_tokens=stats["total_tokens"],
            latency_ms=total_duration_ms,
            session_id=session_id,
            user_id=str(user_id),
            route_confidence=round(confidence, 2),
            model=llm_provider,
            llm_calls=stats["llm_calls"],
        )

    @staticmethod
    def _accumulate_stream_stats(stats: dict, chunk: str) -> None:
        """
        从流式事件里累积统计量（供 [对话交互] 日志使用）。

        事件类型：
        - answer: content 为"全量答案文本"（前端按全量替换渲染），故直接取长度，不做累加
        - stats : Agent 侧算好的 token / LLM 调用次数
        - error : 忽略
        """
        if not chunk:
            return
        try:
            event = json.loads(chunk)
        except (json.JSONDecodeError, TypeError):
            # 非结构化（旧版纯文本）chunk：按增量文本累加长度
            stats["response_length"] += len(chunk)
            return

        if not isinstance(event, dict):
            return

        event_type = event.get("type")
        if event_type == "answer":
            stats["response_length"] = len(event.get("content", "") or "")
        elif event_type == "stats":
            stats["total_tokens"] = int(event.get("total_tokens", 0) or 0)
            stats["llm_calls"] = int(event.get("llm_calls", 0) or 0)

    @staticmethod
    async def _build_attachment_context(user_id, session_id, turn_id, query) -> str | None:
        """
        构建本轮附件上下文（会话级 collection + turn_id 单轮绑定）。
        失败不阻断主流程，仅记录告警并返回 None。
        """
        if not turn_id:
            return None
        try:
            return await chat_attachment_service.build_context(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query or "",
            )
        except Exception as e:
            logger.warning(f"[附件] 构建上下文失败，忽略: {e}")
            return None

    @staticmethod
    def _merge_attachment(query: str, attachment_context: str | None) -> str:
        """把附件参考块拼到用户问题后面（chat 模式直接进 HumanMessage）。"""
        if not attachment_context:
            return query
        return f"{query}\n\n{attachment_context}"

    async def _run_chat_stream(
        self,
        query,
        user_id: str,
        session_id,
        db,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        agent_id=None,
        history=None,
        system_prompt_override=None,
        attachment_context: str | None = None,
    ):
        """
        纯聊天模式（流式）
        """
        import time

        start_time = time.time()
        t0 = time.time()

        logger.info("[计时] _run_chat_stream 开始")

        # 使用传入的 history 或重新获取
        if history is None:
            history = await chat_history_service.get_last_n_exchanges(session_id, n=5, db=db)

        # 提前落库用户问题（流式生成开始前）：保证客户端中途断开/刷新后历史至少保留问题
        try:
            await chat_history_service.save_user_message_only(
                session_id=session_id,
                user_message=query,
                user_id=str(user_id),
                agent_id=agent_id,
                db=db,
            )
        except Exception as _e:
            logger.warning(f"提前保存用户消息失败(非致命): {_e}")

        t1 = time.time()
        logger.info(f"[计时] 获取历史记录完成: {(t1 - t0) * 1000:.0f}ms")

        # 确定系统提示词（优先级：override > Agent 自定义 > 默认）
        system_prompt = None
        if system_prompt_override:
            system_prompt = system_prompt_override
            logger.info("使用路由降级系统提示词（需要工具但无工具可用）")
        elif agent_id and db:
            try:
                from backend.app.models.agent import Agent

                result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = result.scalar_one_or_none()
                if agent and agent.system_prompt:
                    system_prompt = agent.system_prompt
                    logger.info(f"使用Agent自定义系统提示词: {agent_id}")
            except Exception as e:
                logger.warning(f"读取Agent系统提示词失败: {e}")

        # 如果没有自定义提示词，使用默认提示词
        if not system_prompt:
            system_prompt = """你是一个友好的AI助手。请自然地与用户对话。

回答要求：
- 使用规范的 Markdown 格式
- 段落之间空一行
- 重点内容用 **加粗** 标注
- 列表使用 - 或 1. 2. 3.
- 标题使用 ## 或 ###，前后空一行
- 代码使用```代码块```包裹"""

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
        ]
        for msg in history:
            if is_user_role(msg.get("role")):
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif is_assistant_role(msg.get("role")):
                messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=self._merge_attachment(query, attachment_context)))
        t2 = time.time()
        logger.info(f"[计时] 构建消息完成: {(t2 - t1) * 1000:.0f}ms")

        llm = model_factory.get_llm_model(provider=llm_provider, temperature=0.7)
        t3 = time.time()
        logger.info(f"[计时] 获取LLM实例完成: {(t3 - t2) * 1000:.0f}ms")

        full_answer = ""
        usage_metadata = None
        ttft_ms = 0
        first_chunk = True
        logger.info("[计时] 开始调用LLM.stream")

        import json as json_mod

        # chat 模式没有工具调用，不需要渲染思考过程；
        # 如果后续扩展需要显示“AI 正在输入”，由 answer 事件的 streaming 字段承载即可。
        # 因此这里不再发送初始化的 thinking 事件，避免前端误解为 Agent 思考过程。

        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not content:
                continue

            if first_chunk:
                ttft_ms = (time.time() - start_time) * 1000
                first_chunk = False
                t4 = time.time()
                logger.info(
                    f"[计时] 第一个token到达: {(t4 - t3) * 1000:.0f}ms (TTFT: {ttft_ms:.0f}ms)"
                )

            full_answer += content
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

            # 实时推送增量 answer 事件（每 token 都发送，前端按 content 全量替换渲染）
            incremental_event = {
                "type": "answer",
                "content": full_answer,
                "streaming": True,
            }
            yield json_mod.dumps(incremental_event, ensure_ascii=False)

        # 发送最终完成的 answer 事件
        final_event = {
            "type": "answer",
            "content": full_answer,
            "streaming": False,
        }
        yield json_mod.dumps(final_event, ensure_ascii=False)

        # 提取token使用量
        total_tokens = 0
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                input_tokens = usage_metadata.get(
                    "input_tokens", usage_metadata.get("prompt_tokens", 0)
                )
                output_tokens = usage_metadata.get(
                    "output_tokens", usage_metadata.get("completion_tokens", 0)
                )
                total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
            else:
                input_tokens = getattr(
                    usage_metadata, "input_tokens", getattr(usage_metadata, "prompt_tokens", 0)
                )
                output_tokens = getattr(
                    usage_metadata, "output_tokens", getattr(usage_metadata, "completion_tokens", 0)
                )
                total_tokens = getattr(usage_metadata, "total_tokens", input_tokens + output_tokens)

        await chat_history_service.save_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            model_name=llm_provider,
            token_usage=total_tokens,
            db=db,
            skip_user=True,
        )
        t5 = time.time()
        logger.info(f"[计时] 保存历史完成: {(t5 - t4) * 1000:.0f}ms")

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"[计时] _run_chat_stream 总耗时: {duration_ms:.0f}ms")

        # 记录统计日志
        duration_ms = (time.time() - start_time) * 1000
        parts = [
            "【模式】chat",
            f"【模型】{llm_provider}",
        ]
        if ttft_ms > 0:
            parts.append(f"【TTFT】{ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{duration_ms:.0f}ms")
        if usage_metadata:
            parts.append(f"【Token】输入{input_tokens} / 输出{output_tokens} / 总计{total_tokens}")
            parts.append("【LLM调用次数】1")
        logger.info(f"【请求统计】{' | '.join(parts)}")

        # 向客户端推送本次 Token 消耗统计（前端展示用）
        if usage_metadata:
            stats_event = json_mod.dumps(
                {
                    "type": "stats",
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "total_tokens": int(total_tokens or 0),
                    "llm_calls": 1,
                    "duration_ms": round(duration_ms),
                },
                ensure_ascii=False,
            )
            yield stats_event

    async def _run_agent_stream(
        self,
        query,
        user_id: str,
        session_id,
        use_long_term_memory,
        db,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        agent_id=None,
        attachment_context: str | None = None,
    ):
        """Agent 模式（流式）"""
        async for chunk in agent_chat_service.chat_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            use_long_term_memory=use_long_term_memory,
            db=db,
            llm_provider=llm_provider,
            agent_id=agent_id,
            attachment_context=attachment_context,
        ):
            yield chunk

    async def _run_chat(
        self,
        query,
        user_id: str,
        session_id,
        db,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        agent_id=None,
        history=None,
        system_prompt_override=None,
        attachment_context: str | None = None,
    ) -> dict[str, Any]:
        """
        纯聊天模式（非流式）
        - 不走 RAG 检索
        - 不调用工具
        - 直接用 LLM 回复，带对话历史
        """
        # 0. 确定系统提示词（优先级：override > Agent 自定义 > 默认）
        system_prompt = None
        if system_prompt_override:
            system_prompt = system_prompt_override
            logger.info("使用路由降级系统提示词（需要工具但无工具可用）")
        elif agent_id and db:
            try:
                from backend.app.models.agent import Agent

                result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = result.scalar_one_or_none()
                if agent and agent.system_prompt:
                    system_prompt = agent.system_prompt
                    logger.info(f"使用Agent自定义系统提示词: {agent_id}")
            except Exception as e:
                logger.warning(f"读取Agent系统提示词失败: {e}")

        if not system_prompt:
            system_prompt = """你是一个友好的AI助手。请自然地与用户对话。

回答要求：
- 使用规范的 Markdown 格式
- 段落之间空一行
- 重点内容用 **加粗** 标注
- 列表使用 - 或 1. 2. 3.
- 标题使用 ## 或 ###，前后空一行
- 代码使用```代码块```包裹"""

        # 1. 获取对话历史
        if history is None:
            history = await chat_history_service.get_last_n_exchanges(session_id, n=5, db=db)

        # 2. 构建消息（SystemMessage + 历史 + 当前问题）
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
        ]
        for msg in history:
            # 角色判定统一走 roles 工具（同上）
            if is_user_role(msg.get("role")):
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif is_assistant_role(msg.get("role")):
                messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=self._merge_attachment(query, attachment_context)))

        # 3. 调用 LLM 生成回答
        llm = model_factory.get_llm_model(provider=llm_provider, temperature=0.7)
        response = await llm.ainvoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # 提取 token 使用量
        total_tokens = 0
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                total_tokens = usage_metadata.get("total_tokens", 0)
            else:
                total_tokens = getattr(usage_metadata, "total_tokens", 0)

        # 4. 保存对话历史
        await chat_history_service.save_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=answer,
            model_name=llm_provider,
            token_usage=total_tokens,
            db=db,
        )

        return {
            "answer": answer,
            "sources": [],
            "called_tools": [],
        }

    async def _run_agent(
        self,
        query,
        user_id: str,
        session_id,
        use_long_term_memory,
        db,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        agent_id=None,
        attachment_context: str | None = None,
    ) -> dict[str, Any]:
        """Agent 模式"""
        result = await agent_chat_service.chat(
            query=query,
            user_id=user_id,
            session_id=session_id,
            use_long_term_memory=use_long_term_memory,
            db=db,
            llm_provider=llm_provider,
            agent_id=agent_id,
            attachment_context=attachment_context,
        )
        return {
            "answer": result.get("answer", ""),
            "called_tools": result.get("called_tools", []),
            "sources": [],
            "extra": {
                "recalled_memories": result.get("recalled_memories", []),
                "compression_info": result.get("compression_info"),
            },
        }

    def _extract_extra(self, result: dict, mode: str) -> dict:
        """提取附加信息"""
        extra = result.get("extra", {})
        return extra


# ====== 全局实例 ======

smart_chat_service = SmartChatService()
