# backend/app/route/chat.py
from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

from backend.db.db_config import get_session
from backend.app.core.security import get_current_user
from backend.app.schemas.chat import ChatRequest,ChatResponse
from backend.app.services.chat_service import chat_service


router = APIRouter()

@router.post("/chat",summary="RAG问答（支持流式/非流式输出）")
async def chat(
    req:ChatRequest,
    user_id: int = Depends(get_current_user),
    db:Session = Depends(get_session),
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
        result = chat_service.chat(
            query=req.query,
            user_id=user_id,
            session_id=req.session_id,
            collection_name=req.collection_name,
            strategy=req.strategy,
            use_rerank=req.use_rerank,
            db=db,
        )
        return ChatResponse(**result)