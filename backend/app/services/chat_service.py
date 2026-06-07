# backend/app/services/chat_service.py
from typing import Dict,List,Any,Optional
from sqlalchemy.orm import Session

from backend.app.rag.rag_service import rag_service
from backend.app.services.chat_history_service import chat_history_service


class ChatService:
    """
    对话业务编排服务
    """
    def __init__(self,history_rounds: int = 5):
        self.history_rounds = history_rounds
    
    def chat(
        self,
        query: str,
        user_id: int,
        session_id: str,
        collection_name: str = "default_collection",
        strategy: str = "hybrid",
        use_rerank: bool = True,   
        db:Optional[Session] = None,
    )->Dict[str,Any]:
        """
        完整对话流程（非流式）
        """
        # 查找历史
        history = chat_history_service.get_last_n_exchanges(
            session_id,
            n=self.history_rounds,
            db=db,
        )
        # 调用RAG
        result = rag_service.chat(
            query=query,
            collection_name= collection_name,
            history=history,
            strategy=strategy,
            use_rerank=use_rerank,
        )
        # 保存对话
        chat_history_service.save_exchange(
            user_id=user_id,
            session_id=session_id,
            user_message=query,
            assistant_message=result["answer"],
            db=db,
        )
        return result
    
    async def chat_stream(
        self,
        query: str,
        user_id: int,
        session_id: str,
        collection_name: str = "default_collection",
        strategy: str = "hybrid",
        use_rerank: bool = True,
        db: Optional[Session] =None
    ):
        """
        完成对话流程（流式）
        """
        # 查询历史
        history = chat_history_service.get_last_n_exchanges(
            session_id=session_id,
            n=self.history_rounds,
            db=db,
        )
        # 流式生成+拼接完整答案
        full_answer = ""
        async for chunk in rag_service.chat_stream(
            query=query,
            collection_name=collection_name,
            history=history,
            strategy=strategy,
            use_rerank=use_rerank,
        ):
            full_answer += chunk
            yield chunk
        # 保存对话
        chat_history_service.save_exchange(
            user_id=user_id,
            session_id=session_id,
            user_message=query,
            assistant_message=full_answer,
            db=db,
        )

# ====== 全局实例 ======

chat_service = ChatService()