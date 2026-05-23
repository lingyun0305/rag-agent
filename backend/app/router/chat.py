from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse

from backend.db.db_config import get_session
from backend.app.schemas.chat import chatMessage
from backend.app.services.chat_service import chatService

router = APIRouter()

@router.post("/api/chat")
async def chat(
    msg:chatMessage,
    db:Session = Depends(get_session)
):
    return StreamingResponse(
        chatService(msg,db), # 调用生成器
        media_type="text/plain" # 定义前端接收格式为纯文本流，防止流式输出返回格式不对报错
    )