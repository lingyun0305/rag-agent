# backend/app/services/agent_chat_service.py
"""
Agent 对话业务编排服务
整合 Agent 主循环 + 短期对话历史 + 长期记忆
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.agent import agentic_rag
from backend.app.services.chat_history_service import chat_history_service
from backend.app.agent.memory.long_term import long_term_memory
from backend.app.agent.context.compressor import context_compressor
from backend.app.common.logger import logger
from backend.app.utils.request_stats import request_stats_tracker


class AgentChatService:
    """
    Agent 对话服务

    职责：
    - 编排 Agent 调用流程
    - 集成短期对话历史（Redis + MySQL）
    - 集成长期记忆召回与写入（ChromaDB + MySQL）
    - 上下文压缩（滑动窗口/摘要）
    - 对外提供统一的 chat / chat_stream 接口
    """

    def __init__(self, history_rounds: int = 5):
        self.history_rounds = history_rounds

    async def chat(
        self,
        query: str,
        user_id: int,
        session_id: str,
        use_long_term_memory: bool = True,
        db: Optional[AsyncSession] = None,
        llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        async with request_stats_tracker.track(mode="agent", model=llm_provider or "", request_id=session_id) as stats:
            # 1. 获取短期对话历史
            history = await chat_history_service.get_last_n_exchanges(
                session_id=session_id,
                n=self.history_rounds,
                db=db,
            )

            # 2. 上下文压缩（智能策略选择）
            compression_info = None
            if len(history) > 0:
                compressed = await context_compressor.smart_compress(
                    history, context_type="chat_history"
                )
                if compressed["strategy"] != "none":
                    history = compressed["compressed_messages"]
                    compression_info = {
                        "strategy": compressed["strategy"],
                        "reason": compressed["reason"],
                        "original_tokens": compressed["original_tokens"],
                        "compressed_tokens": compressed["compressed_tokens"],
                        "compression_ratio": compressed["compression_ratio"],
                    }

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

            # 5. 调用 Agent 主循环
            result = await agentic_rag.run(
                query=enhanced_query,
                user_id=user_id,
                session_id=session_id,
                history=history,
                llm_provider=llm_provider,
            )

            # 统计token
            if result.get("usage_metadata"):
                request_stats_tracker.add_token_usage(stats, result["usage_metadata"])
                request_stats_tracker.add_extra(stats, "调用工具数", len(result.get("called_tools", [])))

            # 6. 保存短期对话历史
            await chat_history_service.save_exchange(
                user_id=user_id,
                session_id=session_id,
                user_message=query,
                assistant_message=result["answer"],
                db=db,
            )

            # 7. 写入长期记忆（异步，不阻塞返回）
            if use_long_term_memory:
                try:
                    await self._save_to_long_term_memory(
                        user_id=user_id,
                        query=query,
                        answer=result["answer"],
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
        user_id: int,
        session_id: str,
        use_long_term_memory: bool = True,
        db: Optional[AsyncSession] = None,
        llm_provider: Optional[str] = None,
    ):
        """
        Agent 对话（流式）

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            use_long_term_memory: 是否启用长期记忆
            db: 数据库会话

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

        # 2. 上下文压缩（如果历史过长）
        if len(history) > 8:
            compressed = await context_compressor.compress(
                history, strategy="sliding_window", keep_rounds=4
            )
            history = compressed["compressed_messages"]

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

        # 5. 流式生成 + 拼接完整答案
        full_answer = ""
        called_tools_count = 0
        usage_collector = {}
        ttft_ms = 0
        first_chunk = True
        async for chunk in agentic_rag.run_stream(
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
            full_answer += chunk
            yield chunk

        # 6. 保存短期对话历史
        await chat_history_service.save_exchange(
            user_id=user_id,
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            db=db,
        )

        # 7. 写入长期记忆
        if use_long_term_memory:
            try:
                await self._save_to_long_term_memory(
                    user_id=user_id,
                    query=query,
                    answer=full_answer,
                    db=db,
                )
            except Exception as e:
                logger.warning(f"长期记忆写入失败，跳过: {e}")

        # 8. 记录统计日志
        duration_ms = (time.time() - start_time) * 1000
        parts = [
            f"【模式】agent",
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
        logger.info(f"【请求统计】{' | '.join(parts)}")

    def _inject_memory(
        self,
        query: str,
        memories: List[Dict],
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

        memory_text = "\n".join(
            f"- {m['content']}" for m in memories[:3]
        )

        return f"""【用户历史记忆】
{memory_text}

【当前问题】
{query}"""

    async def _save_to_long_term_memory(
        self,
        user_id: int,
        query: str,
        answer: str,
        db: Optional[AsyncSession] = None,
    ):
        """
        将本轮对话写入长期记忆

        策略：
        - 用户问题中的偏好/事实 → preference/fact 类型
        - 简单问答不写入（避免记忆噪音）
        """
        # 简化判断：问题长度超过 20 字符才写入（过滤简单闲聊）
        if len(query) < 20:
            return

        # 拼接记忆内容
        memory_content = f"用户问: {query}\n助手答: {answer[:200]}"

        await long_term_memory.memorize(
            user_id=user_id,
            content=memory_content,
            memory_type="fact",
            db=db,
        )


# ====== 全局实例 ======

agent_chat_service = AgentChatService()