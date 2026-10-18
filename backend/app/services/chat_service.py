# backend/app/services/chat_service.py
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.logger import logger
from backend.app.rag.rag_service import rag_service
from backend.app.services.chat_history_service import chat_history_service
from backend.app.utils.request_stats import request_stats_tracker
from backend.app.utils.roles import is_assistant_role, is_user_role
from backend.app.utils.vector_store import vector_store_util


class ChatService:
    """
    RAG 对话编排
    """

    def __init__(self, history_rounds: int = 5):
        self.history_rounds = history_rounds

    async def chat(
        self,
        query: str,
        user_id: str,
        session_id: str,
        collection_name: str = "default_collection",
        strategy: str = "hybrid",
        use_rerank: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        """
        完整对话流程（非流式）
        """
        async with request_stats_tracker.track(
            mode="rag", model=llm_provider or "", request_id=session_id
        ) as stats:
            # 查找历史
            history = await chat_history_service.get_last_n_exchanges(
                session_id,
                n=self.history_rounds,
                db=db,
            )
            # 调用RAG
            result = await rag_service.chat(
                query=query,
                collection_name=collection_name,
                history=history,
                strategy=strategy,
                use_rerank=use_rerank,
                llm_provider=llm_provider,
            )
            # 统计token
            total_tokens = 0
            if result.get("usage_metadata"):
                request_stats_tracker.add_token_usage(stats, result["usage_metadata"])
                usage = result["usage_metadata"]
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(usage, "total_tokens", 0)
            # 保存对话
            await chat_history_service.save_exchange(
                session_id=session_id,
                user_message=query,
                assistant_message=result["answer"],
                model_name=llm_provider,
                token_usage=total_tokens,
                db=db,
            )
            return result

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        collection_name: str = "default_collection",
        strategy: str = "hybrid",
        use_rerank: bool = True,
        db: AsyncSession | None = None,
        llm_provider: str | None = None,
    ):
        """
        完成对话流程（流式）
        """
        import time

        start_time = time.time()
        usage_metadata = None

        # 防御性检查：知识库为空时直接走纯聊天模式
        try:
            kb_empty = vector_store_util.is_collection_empty()
            if kb_empty:
                logger.info("知识库为空，跳过RAG检索，直接走chat模式")
                async for chunk in self._fallback_chat_stream(
                    query, user_id, session_id, db, llm_provider
                ):
                    yield chunk
                return
        except Exception as e:
            logger.warning(f"检查知识库状态失败，继续RAG流程: {e}")

        # 查询历史
        history = await chat_history_service.get_last_n_exchanges(
            session_id=session_id,
            n=self.history_rounds,
            db=db,
        )
        # 流式生成+拼接完整答案
        full_answer = ""
        usage_collector = {}
        ttft_ms = 0
        first_chunk = True
        async for chunk in rag_service.chat_stream(
            query=query,
            collection_name=collection_name,
            history=history,
            strategy=strategy,
            use_rerank=use_rerank,
            llm_provider=llm_provider,
            usage_collector=usage_collector,
        ):
            if first_chunk and chunk:
                ttft_ms = (time.time() - start_time) * 1000
                first_chunk = False
            full_answer += chunk
            yield chunk

        # 提取token使用量
        total_tokens = 0
        usage_metadata = usage_collector.get("usage")
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

        # 保存对话
        await chat_history_service.save_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            model_name=llm_provider,
            token_usage=total_tokens,
            db=db,
        )

        # 记录统计日志
        duration_ms = (time.time() - start_time) * 1000
        parts = [
            "【模式】rag",
            f"【模型】{llm_provider or ''}",
        ]
        if ttft_ms > 0:
            parts.append(f"【TTFT】{ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{duration_ms:.0f}ms")
        if usage_metadata:
            parts.append(f"【Token】输入{input_tokens} / 输出{output_tokens} / 总计{total_tokens}")
            parts.append("【LLM调用次数】1")
        logger.info(f"【请求统计】{' | '.join(parts)}")

    async def _fallback_chat_stream(self, query, user_id: str, session_id, db, llm_provider=None):
        """
        知识库为空时的降级聊天模式
        直接调用LLM，不走RAG检索
        """
        import time

        from langchain_core.messages import AIMessage, HumanMessage

        from backend.app.core.config import settings
        from backend.app.utils.models_factory import model_factory

        start_time = time.time()
        llm = model_factory.get_llm_model(
            provider=llm_provider or settings.DEFAULT_LLM_PROVIDER, temperature=0.7
        )

        history = await chat_history_service.get_last_n_exchanges(
            session_id=session_id,
            n=self.history_rounds,
            db=db,
        )

        from langchain_core.messages import SystemMessage

        messages = [
            SystemMessage(
                content="""你是一个友好的AI助手。请自然地与用户对话。

回答要求：
- 使用规范的 Markdown 格式
- 段落之间空一行
- 重点内容用 **加粗** 标注
- 列表使用 - 或 1. 2. 3.，每个列表项前必须换行
- 标题使用 ## 或 ###，前后空一行
- 代码使用```代码块```包裹"""
            ),
        ]
        if history:
            for msg in history:
                if is_user_role(msg.get("role")):
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif is_assistant_role(msg.get("role")):
                    messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=query))

        full_answer = ""
        usage_metadata = None
        ttft_ms = 0
        first_chunk = True
        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if first_chunk and content:
                ttft_ms = (time.time() - start_time) * 1000
                first_chunk = False
            full_answer += content
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
            yield content

        # 提取token使用量
        total_tokens = 0
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
            db=db,
        )

        duration_ms = (time.time() - start_time) * 1000
        parts = [
            "【模式】chat(fallback)",
            f"【模型】{llm_provider or ''}",
        ]
        if ttft_ms > 0:
            parts.append(f"【TTFT】{ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{duration_ms:.0f}ms")
        logger.info(f"【请求统计】{' | '.join(parts)}")


# ====== 全局实例 ======

chat_service = ChatService()
