# backend/app/route/chat.py
from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse

from backend.db.db_config import get_session
from backend.app.core.security import get_current_user
from backend.app.schemas.chat import ChatRequest,ChatResponse,AgentChatRequest,AgentChatResponse,ToolInfo,SmartChatRequest,SmartChatResponse
from backend.app.services.chat_service import chat_service
from backend.app.services.agent_chat_service import agent_chat_service
from backend.app.services.smart_chat_service import smart_chat_service
from backend.app.agent.tools.registry import tool_registry


router = APIRouter()

# ====== 传统 RAG 模式 ======

@router.post("/chat",summary="RAG问答（支持流式/非流式输出）")
async def chat(
    req:ChatRequest,
    user_id: int = Depends(get_current_user),
    db:AsyncSession = Depends(get_session),
):
    if req.stream:
        async def generate():
            async for chunk in chat_service.chat_stream(
                query=req.query,
                user_id=user_id,
                session_id=req.session_id,
                collection_name=req.collection_name,
                strategy=req.strategy,
                use_rerank=req.use_rerank,
                db=db,
            ):
                yield f"data:{chunk}\n\n" # SSE（Server-Sent Events）协议格式
        return StreamingResponse(generate(),media_type="text/plain")
    else:
        result = await chat_service.chat(
            query=req.query,
            user_id=user_id,
            session_id=req.session_id,
            collection_name=req.collection_name,
            strategy=req.strategy,
            use_rerank=req.use_rerank,
            db=db,
        )
        return ChatResponse(**result)


# ====== Agent 模式 ======

@router.post("/agent/chat", summary="Agent智能体对话（支持流式/非流式输出）")
async def agent_chat(
    req: AgentChatRequest,
    user_id: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Agent 模式对话：
    - LLM 自主判断是否调用工具
    - 支持多工具串行调用
    - 集成短期对话历史 + 长期记忆
    """
    if req.stream:
        async def generate():
            async for chunk in agent_chat_service.chat_stream(
                query=req.query,
                user_id=user_id,
                session_id=req.session_id,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
            ):
                yield f"data:{chunk}\n\n"
        return StreamingResponse(generate(), media_type="text/plain")
    else:
        result = await agent_chat_service.chat(
            query=req.query,
            user_id=user_id,
            session_id=req.session_id,
            use_long_term_memory=req.use_long_term_memory,
            db=db,
        )
        return AgentChatResponse(**result)


@router.get("/agent/tools", summary="获取Agent可用工具列表")
async def list_tools():
    """获取当前 Agent 注册的所有工具信息"""
    tools = tool_registry.list_tools()
    return {"tools": tools, "total": len(tools)}


# ====== 智能路由模式 ======

@router.post("/smart/chat", summary="智能路由对话（自动选择最优模式）")
async def smart_chat(
    req: SmartChatRequest,
    user_id: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    智能路由模式：
    - 系统自动判断问题类型
    - 简单知识查询 → 传统RAG（快）
    - 需要工具调用 → 单Agent（中等）
    """
    if req.stream:
        import json as _json
        async def generate():
            async for chunk in smart_chat_service.chat_stream(
                query=req.query,
                user_id=user_id,
                session_id=req.session_id,
                collection_name=req.collection_name,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
                llm_provider=req.llm_provider,
            ):
                if not chunk:
                    continue
                yield f"data:{_json.dumps({'choices': [{'delta': {'content': chunk}}]}, ensure_ascii=False)}\n\n"
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        return StreamingResponse(generate(), media_type="text/event-stream", headers=stream_headers)
    else:
        result = await smart_chat_service.chat(
            query=req.query,
            user_id=user_id,
            session_id=req.session_id,
            collection_name=req.collection_name,
            use_long_term_memory=req.use_long_term_memory,
            db=db,
            llm_provider=req.llm_provider,
        )
        return SmartChatResponse(**result)


@router.post("/smart/chat/public", summary="智能路由对话（公开接口，无需登录）")
async def smart_chat_public(
    req: SmartChatRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    公开接口，无需登录，用于测试
    """
    if req.stream:
        import json as _json
        async def generate():
            async for chunk in smart_chat_service.chat_stream(
                query=req.query,
                user_id=1,
                session_id=req.session_id,
                collection_name=req.collection_name,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
                llm_provider=req.llm_provider,
            ):
                if not chunk:
                    continue
                # yield f"data:{_json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                yield f"data:{_json.dumps({'choices': [{'delta': {'content': chunk}}]}, ensure_ascii=False)}\n\n"
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        return StreamingResponse(generate(), media_type="text/event-stream",headers=stream_headers)
    else:
        result = await smart_chat_service.chat(
            query=req.query,
            user_id=1,
            session_id=req.session_id,
            collection_name=req.collection_name,
            use_long_term_memory=req.use_long_term_memory,
            db=db,
            llm_provider=req.llm_provider,
        )
        return SmartChatResponse(**result)