# backend/app/services/smart_chat_service.py
"""
智能路由对话服务
根据用户问题自动选择最合适的处理模式
"""

from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.router import intelligent_router
from backend.app.services.chat_service import chat_service
from backend.app.services.agent_chat_service import agent_chat_service
from backend.app.services.chat_history_service import chat_history_service
from backend.app.utils.models_factory import model_factory
from backend.app.core.config import settings
from backend.app.common.logger import logger
from backend.app.utils.request_stats import request_stats_tracker


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
        user_id: int,
        session_id: str,
        collection_name: str = "default_collection",
        use_long_term_memory: bool = True,
        db: Optional[AsyncSession] = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
    ) -> Dict[str, Any]:
        """
        智能路由对话

        Args:
            query: 用户问题
            user_id: 用户 ID
            session_id: 会话 ID
            collection_name: 知识库集合
            use_long_term_memory: 是否启用长期记忆
            db: 数据库会话

        Returns:
            统一格式的响应
        """
        # 1. 智能路由判断
        route_result = await intelligent_router.route(query=query, llm_provider=llm_provider)
        mode = route_result["mode"]
        confidence = route_result["confidence"]
        reason = route_result["reason"]

        logger.info(f"智能路由: query='{query[:50]}...' → mode={mode}, confidence={confidence:.2f}")

        # 2. 分发到对应模式
        try:
            if mode == "chat":
                result = await self._run_chat(
                    query, user_id, session_id, db, llm_provider
                )
            elif mode == "traditional":
                try:
                    result = await self._run_traditional(
                        query, user_id, session_id, collection_name, db, llm_provider
                    )
                except Exception as e:
                    logger.warning(f"traditional模式失败，降级到chat: {e}")
                    result = await self._run_chat(
                        query, user_id, session_id, db, llm_provider
                    )
                    mode = "chat"
            elif mode == "agent":
                result = await self._run_agent(
                    query, user_id, session_id, use_long_term_memory, db, llm_provider
                )
            else:
                result = await self._run_chat(
                    query, user_id, session_id, db, llm_provider
                )
                mode = "chat"

        except Exception as e:
            logger.error(f"智能路由执行失败 (mode={mode}): {e}")
            result = await self._run_chat(
                query, user_id, session_id, db, llm_provider
            )
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
        }

    async def chat_stream(
        self,
        query: str,
        user_id: int,
        session_id: str,
        collection_name: str = "default_collection",
        use_long_term_memory: bool = True,
        db: Optional[AsyncSession] = None,
        llm_provider: str = settings.DEFAULT_LLM_PROVIDER,
    ):
        """
        智能路由对话（流式版）
        路由判断后，直接 yield chunk 给前端
        """
        import time
        total_start = time.time()

        # 1. 智能路由判断
        route_result = await intelligent_router.route(query=query, llm_provider=llm_provider)
        mode = route_result["mode"]
        confidence = route_result["confidence"]
        reason = route_result["reason"]

        logger.info(f"智能路由: query='{query[:50]}...' → mode={mode}, confidence={confidence:.2f}")

        # 2. 分发到对应模式，流式返回
        try:
            if mode == "chat":
                async for chunk in self._run_chat_stream(query, user_id, session_id, db, llm_provider):
                    yield chunk
            elif mode == "traditional":
                async for chunk in self._run_traditional_stream(query, user_id, session_id, collection_name, db, llm_provider):
                    yield chunk
            elif mode == "agent":
                async for chunk in self._run_agent_stream(query, user_id, session_id, use_long_term_memory, db, llm_provider):
                    yield chunk
            else:
                async for chunk in self._run_chat_stream(query, user_id, session_id, db, llm_provider):
                    yield chunk

        except Exception as e:
            logger.error(f"智能路由流式执行失败 (mode={mode}): {e}")
            async for chunk in self._run_chat_stream(query, user_id, session_id, db, llm_provider):
                yield chunk

        # 3. 记录总耗时
        total_duration_ms = (time.time() - total_start) * 1000
        parts = [
            f"【模式】{mode}",
            f"【模型】{llm_provider}",
            f"【总耗时】{total_duration_ms:.0f}ms",
            f"【路由置信度】{confidence:.2f}",
        ]
        logger.info(f"【智能路由统计】{' | '.join(parts)}")

    async def _run_chat_stream(
        self, query, user_id, session_id, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ):
        """
        纯聊天模式（流式）
        """
        import time
        start_time = time.time()
        t0 = time.time()

        logger.info(f"[计时] _run_chat_stream 开始")
        
        history = await chat_history_service.get_last_n_exchanges(
            session_id, n=5, db=db
        )
        t1 = time.time()
        logger.info(f"[计时] 获取历史记录完成: {(t1-t0)*1000:.0f}ms")

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages = [
            SystemMessage(content="""你是一个友好的AI助手。请自然地与用户对话。

回答要求：
- 使用规范的 Markdown 格式
- 段落之间空一行
- 重点内容用 **加粗** 标注
- 列表使用 - 或 1. 2. 3.
- 标题使用 ## 或 ###，前后空一行
- 代码使用```代码块```包裹"""),
        ]
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))
        t2 = time.time()
        logger.info(f"[计时] 构建消息完成: {(t2-t1)*1000:.0f}ms")

        llm = model_factory.get_llm_model(provider=llm_provider, temperature=0.7)
        t3 = time.time()
        logger.info(f"[计时] 获取LLM实例完成: {(t3-t2)*1000:.0f}ms")
        
        full_answer = ""
        usage_metadata = None
        ttft_ms = 0
        first_chunk = True
        logger.info(f"[计时] 开始调用LLM.stream")
        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if first_chunk and content:
                ttft_ms = (time.time() - start_time) * 1000
                first_chunk = False
                t4 = time.time()
                logger.info(f"[计时] 第一个token到达: {(t4-t3)*1000:.0f}ms (TTFT: {ttft_ms:.0f}ms)")
            full_answer += content
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
            yield content

        await chat_history_service.save_exchange(
            user_id=user_id,
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            db=db,
        )
        t5 = time.time()
        logger.info(f"[计时] 保存历史完成: {(t5-t4)*1000:.0f}ms")

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"[计时] _run_chat_stream 总耗时: {duration_ms:.0f}ms")

        # 记录统计日志
        duration_ms = (time.time() - start_time) * 1000
        parts = [
            f"【模式】chat",
            f"【模型】{llm_provider}",
        ]
        if ttft_ms > 0:
            parts.append(f"【TTFT】{ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{duration_ms:.0f}ms")
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
                output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
                total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
            else:
                input_tokens = getattr(usage_metadata, "input_tokens", getattr(usage_metadata, "prompt_tokens", 0))
                output_tokens = getattr(usage_metadata, "output_tokens", getattr(usage_metadata, "completion_tokens", 0))
                total_tokens = getattr(usage_metadata, "total_tokens", input_tokens + output_tokens)
            parts.append(f"【Token】输入{input_tokens} / 输出{output_tokens} / 总计{total_tokens}")
            parts.append(f"【LLM调用次数】1")
        logger.info(f"【请求统计】{' | '.join(parts)}")

    async def _run_traditional_stream(
        self, query, user_id, session_id, collection_name, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ):
        """传统 RAG 模式（流式）"""
        async for chunk in chat_service.chat_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            db=db,
            llm_provider=llm_provider,
        ):
            yield chunk

    async def _run_agent_stream(
        self, query, user_id, session_id, use_long_term_memory, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ):
        """单 Agent 模式（流式）"""
        async for chunk in agent_chat_service.chat_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            use_long_term_memory=use_long_term_memory,
            db=db,
            llm_provider=llm_provider,
        ):
            yield chunk

    async def _run_traditional(
        self, query, user_id, session_id, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ) -> Dict[str, Any]:
        """
        纯聊天模式
        - 不走 RAG 检索
        - 不调用工具
        - 直接用 LLM 回复，带对话历史
        """
        # 1. 获取对话历史
        history = await chat_history_service.get_last_n_exchanges(
            session_id, n=5, db=db
        )

        # 2. 构建消息（SystemMessage + 历史 + 当前问题）
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages = [
            SystemMessage(content="""你是一个友好的AI助手。请自然地与用户对话。

回答要求：
- 使用规范的 Markdown 格式
- 段落之间空一行
- 重点内容用 **加粗** 标注
- 列表使用 - 或 1. 2. 3.
- 标题使用 ## 或 ###，前后空一行
- 代码使用```代码块```包裹"""),
        ]
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))

        # 3. 调用 LLM 生成回答
        llm = model_factory.get_llm_model(provider=llm_provider, temperature=0.7)
        response = await llm.ainvoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # 4. 保存对话历史
        await chat_history_service.save_exchange(
            user_id=user_id,
            session_id=session_id,
            user_message=query,
            assistant_message=answer,
            db=db,
        )

        return {
            "answer": answer,
            "sources": [],
            "called_tools": [],
        }

    async def _run_traditional(
        self, query, user_id, session_id, collection_name, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ) -> Dict[str, Any]:
        """传统 RAG 模式"""
        full_answer = ""
        async for chunk in chat_service.chat_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            db=db,
            llm_provider=llm_provider,
        ):
            full_answer += chunk
        return {
            "answer": full_answer,
            "sources": [],
        }

    async def _run_agent(
        self, query, user_id, session_id, use_long_term_memory, db, llm_provider=settings.DEFAULT_LLM_PROVIDER
    ) -> Dict[str, Any]:
        """单 Agent 模式"""
        result = await agent_chat_service.chat(
            query=query,
            user_id=user_id,
            session_id=session_id,
            use_long_term_memory=use_long_term_memory,
            db=db,
            llm_provider=llm_provider,
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

    def _extract_extra(self, result: Dict, mode: str) -> Dict:
        """提取附加信息"""
        extra = result.get("extra", {})
        if not extra and mode == "traditional":
            extra = {"strategy": "hybrid", "rerank": True}
        return extra


# ====== 全局实例 ======

smart_chat_service = SmartChatService()