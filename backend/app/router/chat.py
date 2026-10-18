# backend/app/route/chat.py
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.tools.registry import tool_registry
from backend.app.core.security import get_current_user
from backend.app.schemas.chat import (
    AgentChatRequest,
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    SmartChatRequest,
    SmartChatResponse,
)
from backend.app.services.agent_chat_service import agent_chat_service
from backend.app.services.attachment_service import chat_attachment_service
from backend.app.services.chat_service import chat_service
from backend.app.services.smart_chat_service import smart_chat_service
from backend.db.db_config import get_session

router = APIRouter()

# ====== 传统 RAG 模式 ======


@router.post("/chat", summary="RAG问答（支持流式/非流式输出）")
async def chat(
    req: ChatRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
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
                llm_provider=req.llm_provider,
            ):
                yield f"data:{chunk}\n\n"  # SSE（Server-Sent Events）协议格式

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        result = await chat_service.chat(
            query=req.query,
            user_id=user_id,
            session_id=req.session_id,
            collection_name=req.collection_name,
            strategy=req.strategy,
            use_rerank=req.use_rerank,
            db=db,
            llm_provider=req.llm_provider,
        )
        return ChatResponse(**result)


# ====== Agent 模式 ======


@router.post("/agent/chat", summary="Agent智能体对话（支持流式/非流式输出）")
async def agent_chat(
    req: AgentChatRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    Agent 模式对话：
    - LLM 自主判断是否调用工具（ReAct 范式）
    - 支持多工具串行调用
    - 集成短期对话历史 + 长期记忆
    - 根据 agent_id 动态加载关联工具
    """
    if req.stream:

        async def generate():
            async for chunk in agent_chat_service.chat_stream(
                query=req.query,
                user_id=user_id,
                session_id=req.session_id,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
                agent_id=req.agent_id,
                llm_provider=req.llm_provider,
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
            agent_id=req.agent_id,
            llm_provider=req.llm_provider,
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
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    智能路由模式（两种模式）：
    - chat: 纯聊天，简单问候直接回复
    - agent: Agent 模式，知识查询/工具调用
    所有知识查询统一走 agent 模式，由 Agent 自主调用 knowledge_search 工具
    """
    if req.stream:

        async def generate():
            async for chunk in smart_chat_service.chat_stream(
                query=req.query,
                user_id=user_id,
                session_id=req.session_id,
                collection_name=req.collection_name,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
                llm_provider=req.llm_provider,
                agent_id=req.agent_id,
                turn_id=req.turn_id,
            ):
                if not chunk:
                    continue
                # 传递结构化事件 JSON（前端自行解析 type 字段）
                yield f"data:{chunk}\n\n"

        stream_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
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
            agent_id=req.agent_id,
            turn_id=req.turn_id,
        )
        return SmartChatResponse(**result)


@router.get("/history/{session_id}", summary="获取会话历史消息（分页加载）")
async def get_chat_history(
    session_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    page: int = 0,
    page_size: int = 10,
):
    """
    获取指定会话的历史消息（支持分页）

    - page: 页码，0 表示最新一页（最近10轮对话）
    - page_size: 每页轮数（每轮=user+ai）
    - 返回按时间升序排列的消息
    - has_more: 是否有更早的历史
    - total_pages: 总页数
    """
    from sqlalchemy import select

    from backend.app.models.chat_history import ChatMessage, ChatSession, TaskMessage

    # 验证会话归属（软删除的会话视同不存在，Agent 删除后其历史不再对用户可见）
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = session_result.scalar_one_or_none()
    if not session or session.status == "deleted":
        return {"messages": [], "has_more": False, "total_pages": 0, "page": 0, "session": None}

    # 获取所有消息
    all_messages_result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    all_messages = all_messages_result.scalars().all()

    # 自定义排序：时间升序，时间相同则 human/user 排在 ai/assistant 前面
    # 原因：AI回复太快，用户消息和AI消息的 create_time 可能相同
    def sort_key(msg):
        # 角色优先级：human/user = 0, ai/assistant/tool = 1
        role_priority = 0 if msg.role in ("human", "user") else 1
        return (msg.create_time, role_priority, msg.id)

    all_messages.sort(key=sort_key)

    # 批量查询助手消息关联的 task_message，组装 thinking_steps
    message_ids = [msg.id for msg in all_messages if msg.role == "ai"]
    task_map = {}
    if message_ids:
        task_result = await db.execute(
            select(TaskMessage)
            .where(TaskMessage.message_id.in_(message_ids))
            .order_by(TaskMessage.step.asc())
        )
        for tr in task_result.scalars().all():
            task_map.setdefault(tr.message_id, []).append(tr)

    if not all_messages:
        return {
            "messages": [],
            "has_more": False,
            "total_pages": 0,
            "page": 0,
            "session": {
                "id": session.id,
                "title": session.title,
                "agent_id": session.agent_id,
                "message_count": session.message_count,
            },
        }

    # 按对话轮次分组（每轮以 human/user 消息开始，到下一个 human/user 之前结束）
    turns = []
    current_turn = []

    for msg in all_messages:
        role = msg.role
        # 如果是用户消息（human/user）且当前轮次不为空，开始新一轮
        if role in ("human", "user") and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)

    # 添加最后一轮
    if current_turn:
        turns.append(current_turn)

    # 计算分页信息
    total_turns = len(turns)
    total_pages = max(1, (total_turns + page_size - 1) // page_size)

    # 从最新页开始，page=0 表示最新一页
    # 计算当前页的起止位置（从末尾往前）
    start_turn_idx = max(0, total_turns - (page + 1) * page_size)
    end_turn_idx = total_turns - page * page_size

    # 确保边界正确
    if page >= total_pages:
        # 请求超过总页数，返回空
        return {
            "messages": [],
            "has_more": False,
            "total_pages": total_pages,
            "page": page,
            "session": {
                "id": session.id,
                "title": session.title,
                "agent_id": session.agent_id,
                "message_count": session.message_count,
            },
        }

    # 获取当前页的轮次
    page_turns = turns[start_turn_idx:end_turn_idx]

    # 将轮次展开为消息列表
    import json

    from backend.app.services.chat_history_service import chat_history_service

    message_list = []
    for turn in page_turns:
        for msg in turn:
            thinking_steps = None
            if msg.role == "ai" and msg.id in task_map:
                steps = chat_history_service.build_thinking_steps(task_map[msg.id])
                if steps:
                    thinking_steps = json.dumps(steps, ensure_ascii=False)
            message_list.append(
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "thinking_steps": thinking_steps,
                    "model_name": msg.model_name,
                    "create_time": (
                        msg.create_time.strftime("%Y-%m-%d %H:%M:%S") if msg.create_time else None
                    ),
                }
            )

    # 判断是否有更早的历史
    has_more = start_turn_idx > 0

    return {
        "messages": message_list,
        "has_more": has_more,
        "total_pages": total_pages,
        "page": page,
        "session": {
            "id": session.id,
            "title": session.title,
            "agent_id": session.agent_id,
            "message_count": session.message_count,
        },
    }


@router.post("/smart/chat/public", summary="智能路由对话（公开接口，无需登录）")
async def smart_chat_public(
    req: SmartChatRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    公开接口，无需登录，用于测试
    """
    if req.stream:

        async def generate():
            async for chunk in smart_chat_service.chat_stream(
                query=req.query,
                user_id="1",
                session_id=req.session_id,
                collection_name=req.collection_name,
                use_long_term_memory=req.use_long_term_memory,
                db=db,
                llm_provider=req.llm_provider,
                agent_id=req.agent_id,
                turn_id=req.turn_id,
            ):
                if not chunk:
                    continue
                yield f"data:{chunk}\n\n"

        stream_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(generate(), media_type="text/event-stream", headers=stream_headers)
    else:
        result = await smart_chat_service.chat(
            query=req.query,
            user_id="1",
            session_id=req.session_id,
            collection_name=req.collection_name,
            use_long_term_memory=req.use_long_term_memory,
            db=db,
            llm_provider=req.llm_provider,
            agent_id=req.agent_id,
            turn_id=req.turn_id,
        )
        return SmartChatResponse(**result)


# ====== 聊天附件（绑单轮 / 会话级 collection） ======


@router.post("/chat/attachment/upload", summary="上传聊天附件（绑单轮）")
async def upload_chat_attachment(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    turn_id: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    """
    上传本轮对话的附件：落盘 + 切片入会话级 collection(session_<sid>)，
    metadata 写死 turn_id 实现单轮绑定。前端上传时与发送消息共用同一个 turn_id。
    """
    meta = await chat_attachment_service.ingest(
        file=file, user_id=user_id, session_id=session_id, turn_id=turn_id
    )
    return {"code": 200, "msg": "ok", "data": meta}


@router.post("/chat/attachment/delete", summary="删除单个附件")
async def delete_chat_attachment(
    session_id: str = Form(...),
    turn_id: str = Form(...),
    file_id: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    result = chat_attachment_service.delete_file(
        user_id=user_id, session_id=session_id, turn_id=turn_id, file_id=file_id
    )
    return {"code": 200, "msg": "ok", "data": result}


@router.post("/chat/attachment/clear", summary="清除某一轮的全部附件")
async def clear_chat_attachment(
    session_id: str = Form(...),
    turn_id: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    result = chat_attachment_service.clear_turn(
        user_id=user_id, session_id=session_id, turn_id=turn_id
    )
    return {"code": 200, "msg": "ok", "data": result}
