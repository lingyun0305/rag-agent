# backend/app/services/agent_chat_service.py
"""
Agent 对话业务编排服务
整合 Agent 主循环 + 短期对话历史 + 长期记忆
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.agent import smart_agent
from backend.app.agent.context.compressor import context_compressor
from backend.app.agent.memory.long_term import long_term_memory
from backend.app.common.logger import logger
from backend.app.services.chat_history_service import chat_history_service
from backend.app.utils.request_stats import request_stats_tracker

# 会话历史注入策略（2026-10-13 调整）：
# - 注入上下文的会话历史最多保留近 3 轮（user+assistant）。
# - 若这 3 轮仍超过该 token 预算，则整段摘要压缩（LLM 摘要，不保留近期原文），
#   避免长历史撑爆上下文。
SESSION_HISTORY_KEEP_ROUNDS = 3
SESSION_HISTORY_COMPRESS_TOKEN = 2000


class AgentChatService:
    """
    Agent 对话服务

    职责：
    - 编排 Agent 调用流程
    - 集成短期对话历史（Redis + MySQL）
    - 集成长期记忆召回与写入（ChromaDB + MySQL）
    - 上下文压缩（滑动窗口/摘要）
    - 根据 agent_id 动态加载关联工具
    - 对外提供统一的 chat / chat_stream 接口
    """

    def __init__(self, history_rounds: int = SESSION_HISTORY_KEEP_ROUNDS):
        self.history_rounds = history_rounds
        self._current_agent_id: str | None = None

    async def _compress_session_history(
        self, history: list[dict]
    ) -> tuple[list[dict], dict | None]:
        """会话历史压缩：保留近 ``history_rounds`` 轮；若仍超 token 预算则整段摘要压缩。

        返回 (压缩后历史, 压缩信息或 None)。摘要走 LLM，仅在超预算时触发，开销可控。
        """
        if not history:
            return history, None
        hist_tokens = context_compressor.estimate_tokens(history)
        if hist_tokens <= SESSION_HISTORY_COMPRESS_TOKEN:
            return history, None
        compressed = await context_compressor.compress(
            history,
            strategy="summary",
            keep_recent_rounds=0,
            force=True,
        )
        info = {
            "strategy": compressed["strategy"],
            "reason": (
                f"会话历史超 {SESSION_HISTORY_COMPRESS_TOKEN} token"
                f"（{hist_tokens}）触发整段摘要压缩"
            ),
            "original_tokens": compressed["original_tokens"],
            "compressed_tokens": compressed["compressed_tokens"],
            "compression_ratio": compressed["compression_ratio"],
        }
        return compressed["compressed_messages"], info

    async def _get_agent(
        self,
        agent_id: str | None,
        db: AsyncSession | None = None,
        llm_provider: str | None = None,
    ):
        """
        获取 Agent 实例（异步版本，正确加载关联工具）

        如果 agent_id 变化或首次调用，重新初始化 Agent 以加载关联工具。
        在异步上下文中正确加载工具，避免越权问题。
        """
        global smart_agent
        need_reinit = False

        if agent_id != self._current_agent_id:
            need_reinit = True
            self._current_agent_id = agent_id

        if need_reinit or smart_agent is None:
            from backend.app.agent.agent import SmartAgent

            # 异步加载工具（在异步上下文中正确执行）
            pre_loaded_tools = None
            agent_system_prompt = None
            agent_temperature = 0.7  # 默认温度，下面读到 Agent 自定义值则覆盖

            if agent_id and db:
                # 异步加载 BaseTool 实例（供纯 ReAct 直接 run() 执行）
                pre_loaded_base_tools = await SmartAgent.async_resolve_base_tools(
                    agent_id=agent_id,
                    db=db,
                    permission_level="safe",
                )
                logger.info(
                    f"异步预加载 Agent '{agent_id}' BaseTool: {len(pre_loaded_base_tools)} 个"
                )

                # 加载 Agent 自定义系统提示词
                try:
                    from sqlalchemy import select

                    from backend.app.models.agent import Agent

                    result = await db.execute(select(Agent).where(Agent.id == agent_id))
                    agent_obj = result.scalar_one_or_none()
                    if agent_obj:
                        if agent_obj.system_prompt:
                            agent_system_prompt = agent_obj.system_prompt
                            logger.info(f"加载 Agent '{agent_id}' 自定义系统提示词")
                        if agent_obj.temperature is not None:
                            agent_temperature = agent_obj.temperature
                            logger.info(f"加载 Agent '{agent_id}' 自定义温度: {agent_temperature}")
                except Exception as e:
                    logger.warning(f"加载 Agent 自定义提示词失败: {e}")

                # 异步加载工具
                pre_loaded_tools = await SmartAgent.async_resolve_tools(
                    agent_id=agent_id,
                    db=db,
                    permission_level="safe",
                )
                logger.info(f"异步预加载 Agent '{agent_id}' 工具: {len(pre_loaded_tools)} 个")

                smart_agent = SmartAgent(
                    llm_provider=llm_provider,
                    temperature=agent_temperature,
                    agent_id=agent_id,
                    db=db,
                    pre_loaded_tools=pre_loaded_tools,
                    pre_loaded_base_tools=pre_loaded_base_tools,
                    agent_system_prompt=agent_system_prompt,
                )
            logger.info(f"Agent 重新初始化: agent_id={agent_id}")

        return smart_agent

    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        use_long_term_memory: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str | None = None,
        agent_id: str | None = None,
        attachment_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Agent 对话（非流式）

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            use_long_term_memory: 是否启用长期记忆
            db: 数据库会话

        Returns:
            {
                "answer": str,
                "called_tools": list,
                "tool_results": list,
                "recalled_memories": list,
            }
        """
        async with request_stats_tracker.track(
            mode="agent", model=llm_provider or "", request_id=session_id
        ) as stats:
            # 1. 获取短期对话历史
            history = await chat_history_service.get_last_n_exchanges(
                session_id=session_id,
                n=self.history_rounds,
                db=db,
            )

            # 2. 上下文压缩（保留近 3 轮；超 token 预算则整段摘要）
            compression_info = None
            history, compression_info = await self._compress_session_history(history)

            # 3. 召回长期记忆（可选）
            recalled_memories = []
            if use_long_term_memory:
                try:
                    recalled_memories = await long_term_memory.recall(
                        user_id=user_id,
                        query=query,
                        top_k=3,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(f"长期记忆召回失败，跳过: {e}")

            # 4. 将长期记忆注入到 query 上下文
            enhanced_query = self._inject_memory(query, recalled_memories)
            # 4.1 将本轮附件参考块注入（会话级 collection + turn_id 单轮绑定）
            enhanced_query = self._inject_attachment(enhanced_query, attachment_context)

            # 5. 获取 Agent 实例并调用
            agent = await self._get_agent(agent_id, db, llm_provider)
            result = await agent.run(
                query=enhanced_query,
                user_id=user_id,
                session_id=session_id,
                history=history,
                llm_provider=llm_provider,
            )

            # 统计token
            total_tokens = 0
            if result.get("usage_metadata"):
                request_stats_tracker.add_token_usage(stats, result["usage_metadata"])
                request_stats_tracker.add_extra(
                    stats, "调用工具数", len(result.get("called_tools", []))
                )
                usage = result["usage_metadata"]
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(usage, "total_tokens", 0)

            # 6. 保存短期对话历史
            await chat_history_service.save_exchange(
                session_id=session_id,
                user_message=query,
                assistant_message=result["answer"],
                model_name=llm_provider,
                token_usage=total_tokens,
                db=db,
            )

            # 7. 写入长期记忆（异步，不阻塞返回）
            if use_long_term_memory:
                try:
                    await self._save_to_long_term_memory(
                        user_id=user_id,
                        query=query,
                        answer=result["answer"],
                        session_id=session_id,
                        agent_id=agent_id,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(f"长期记忆写入失败，跳过: {e}")

            return {
                "answer": result["answer"],
                "called_tools": result.get("called_tools", []),
                "tool_results": result.get("tool_results", []),
                "recalled_memories": recalled_memories,
                "compression_info": compression_info,
            }

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        use_long_term_memory: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str | None = None,
        agent_id: str | None = None,
        attachment_context: str | None = None,
    ):
        """
        Agent 对话（流式）

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            use_long_term_memory: 是否启用长期记忆
            db: 数据库会话
            agent_id: Agent ID，用于加载关联工具

        Yields:
            字符串 chunk
        """
        import time

        start_time = time.time()

        # 1. 获取短期对话历史
        history = await chat_history_service.get_last_n_exchanges(
            session_id=session_id,
            n=self.history_rounds,
            db=db,
        )

        # 2. 上下文压缩（保留近 3 轮；超 token 预算则整段摘要）
        history, _ = await self._compress_session_history(history)

        # 提前落库用户问题（流式生成开始前）：保证客户端中途断开/刷新后历史至少保留问题。
        # 注意：必须在 history 抓取/压缩之后、agent.run 之前写入，避免当前问题被重复注入上下文。
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

        # 3. 召回长期记忆
        recalled_memories = []
        if use_long_term_memory:
            try:
                recalled_memories = await long_term_memory.recall(
                    user_id=user_id,
                    query=query,
                    top_k=3,
                    db=db,
                )
            except Exception as e:
                logger.warning(f"长期记忆召回失败，跳过: {e}")

        # 4. 注入长期记忆
        enhanced_query = self._inject_memory(query, recalled_memories)
        # 4.1 将本轮附件参考块注入（会话级 collection + turn_id 单轮绑定）
        enhanced_query = self._inject_attachment(enhanced_query, attachment_context)

        # 5. 流式生成 + 拼接完整答案
        import json as json_mod

        full_answer = ""
        usage_collector = {}
        ttft_ms = 0
        first_chunk = True
        all_thinking_steps = []

        agent = await self._get_agent(agent_id, db, llm_provider)
        try:
            async for chunk in agent.run_stream(
                query=enhanced_query,
                user_id=user_id,
                session_id=session_id,
                history=history,
                llm_provider=llm_provider,
                usage_collector=usage_collector,
            ):
                if first_chunk and chunk:
                    ttft_ms = (time.time() - start_time) * 1000
                    first_chunk = False
                # chunk 现在是 JSON 字符串（结构化事件）
                # 提取 answer 类型事件作为保存到历史的最终答案
                # 收集 thinking/tool_result 事件用于持久化到 task_message
                try:
                    event = json_mod.loads(chunk)
                    event_type = event.get("type", "")
                    if event_type == "answer":
                        full_answer = event.get("content", "")
                    elif event_type in ("thinking", "tool_result"):
                        all_thinking_steps.append(event)
                except (json_mod.JSONDecodeError, TypeError):
                    full_answer += chunk
                yield chunk
        except Exception as e:
            # 流式执行中途异常：保留已收集的 thinking 步骤，避免刷新后思考过程丢失
            logger.error(
                f"Agent 流式执行异常，保留已收集的 {len(all_thinking_steps)} 步思考过程: {e}"
            )

        # 6. 统计token并保存短期对话历史（含思考过程）
        total_tokens = 0
        usage_metadata = usage_collector.get("usage", {})
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                total_tokens = usage_metadata.get("total_tokens", 0)
            else:
                total_tokens = getattr(usage_metadata, "total_tokens", 0)

        await chat_history_service.save_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            model_name=llm_provider,
            token_usage=total_tokens,
            thinking_steps=all_thinking_steps if all_thinking_steps else None,
            db=db,
            skip_user=True,
        )

        # 7. 写入长期记忆
        if use_long_term_memory:
            try:
                await self._save_to_long_term_memory(
                    user_id=user_id,
                    query=query,
                    answer=full_answer,
                    session_id=session_id,
                    agent_id=agent_id,
                    db=db,
                )
            except Exception as e:
                logger.warning(f"长期记忆写入失败，跳过: {e}")

        # 8. 记录统计日志
        duration_ms = (time.time() - start_time) * 1000
        parts = [
            "【模式】agent",
            f"【模型】{llm_provider or ''}",
        ]
        if ttft_ms > 0:
            parts.append(f"【TTFT】{ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{duration_ms:.0f}ms")
        usage_metadata = usage_collector.get("usage", {})
        if usage_metadata:
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)
            llm_calls = usage_metadata.get("llm_calls", 0)
            parts.append(f"【Token】输入{input_tokens} / 输出{output_tokens} / 总计{total_tokens}")
            parts.append(f"【LLM调用次数】{llm_calls}")

            # 向客户端推送本次 Token 消耗统计（前端展示用）
            stats_event = json_mod.dumps(
                {
                    "type": "stats",
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "total_tokens": int(total_tokens or 0),
                    "llm_calls": int(llm_calls or 0),
                    "duration_ms": round(duration_ms),
                },
                ensure_ascii=False,
            )
            yield stats_event
        logger.info(f"【请求统计】{' | '.join(parts)}")

    def _inject_memory(
        self,
        query: str,
        memories: list[dict],
    ) -> str:
        """
        将长期记忆注入到用户问题中

        Args:
            query: 原始问题
            memories: 召回的记忆列表

        Returns:
            增强后的问题
        """
        if not memories:
            return query

        memory_text = "\n".join(f"- {m['content']}" for m in memories[:3])

        return f"""【用户历史记忆】
{memory_text}

【当前问题】
{query}"""

    def _inject_attachment(self, query: str, attachment_context: str | None) -> str:
        """
        将本轮上传的附件参考块拼到 query 末尾。

        附件内容由 smart_chat_service 在入口处统一构建（会话级 collection + turn_id
        单轮绑定），这里只负责把这块"参考资料"接到用户问题后面，让 Agent 在 ReAct
        时能看到——不依赖模型是否主动调用 file_reader 工具，避免路由误判/工具失败导致
        附件被忽略。
        """
        if not attachment_context:
            return query
        return f"{query}\n\n{attachment_context}"

    async def _save_to_long_term_memory(
        self,
        user_id: str,
        query: str,
        answer: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        db: AsyncSession | None = None,
    ):
        """
        将本轮对话写入长期记忆

        写入策略（修复前是"len(query) >= 20 就写"，等于把每轮问答都塞进记忆库，
        召回时全是噪音，而且会把 <thought> 这类内部字段一起存进去）：

        1. **门控**：只有出现"显式记忆意图"（记住/以后/我叫/我的偏好…）才写，
           来源标记 explicit；否则不写，避免记忆库被日常问答污染。
        2. **降噪**：清掉 <thought>/<action>/<input>/<observation> 等内部标签，
           答案截断到 200 字并去掉内部字段。
        """
        content = self._compose_memory_content(query, answer)
        if content is None:
            return

        await long_term_memory.memorize(
            user_id=user_id,
            content=content,
            memory_type="preference",
            source="explicit",
            source_session_id=session_id,
            agent_id=agent_id,
            db=db,
        )

    # 显式记忆意图词：命中才写长期记忆（宁可少记，也不要记一堆噪音）
    _EXPLICIT_MEMORY_WORDS = (
        "记住",
        "请记得",
        "别忘了",
        "以后都",
        "以后请",
        "下次",
        "我叫",
        "我是",
        "我的名字",
        "我习惯",
        "我喜欢",
        "我偏好",
        "我不喜欢",
        "我们是",
        "我们公司",
        "我的部门",
        "我在",
        "不要再",
        "别再",
    )

    # 需要从记忆内容里剔除的内部字段标签
    _INTERNAL_TAGS = ("thought", "action", "input", "observation")

    @classmethod
    def _compose_memory_content(cls, query: str, answer: str) -> str | None:
        """
        组装待写入的记忆文本；若本轮无记忆价值则返回 None。

        Returns:
            记忆文本，或 None（表示跳过写入）
        """
        q = (query or "").strip()
        if not q:
            return None
        if not any(w in q for w in cls._EXPLICIT_MEMORY_WORDS):
            return None

        clean_answer = cls._strip_internal_tags(answer or "")[:200]
        content = f"用户问: {q}\n助手答: {clean_answer}" if clean_answer else f"用户陈述: {q}"
        return content

    @classmethod
    def _strip_internal_tags(cls, text: str) -> str:
        """剔除 <thought>/<action>/<input>/<observation> 等内部字段，保留正文。"""
        import re

        cleaned = text
        for tag in cls._INTERNAL_TAGS:
            cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(rf"</?{tag}>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"【最终答案】|最终答案内容[:：]?|最终答案[:：]?", "", cleaned)
        return cleaned.strip()


# ====== 全局实例 ======

agent_chat_service = AgentChatService()
