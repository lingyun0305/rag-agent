# backend/app/services/chat_history_service.py
from typing import List, Optional, Dict
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.models.chat_history import ChatSession, ChatMessage
from backend.app.common.logger import logger
from backend.app.common.exceptions import SQLAlchemyError
from backend.app.core.cache import cache_service


class ChatHistoryService:
    """
    对话历史管理服务
    Redis: 快速读写，做缓存
    MySQL: 持久化存储，做备份
    """
    def __init__(
        self,
        max_history_limit: int = 20,
        redis_expire: int = 60 * 60 * 24 * 7,
    ):
        self.max_history_limit = max_history_limit
        self.redis_expire = redis_expire

    def _key(self, session_id: str) -> str:
        """生成Redis Key"""
        return f"chat:history:{session_id}"

    @staticmethod
    def _map_role(role: str) -> str:
        """兼容旧的 role 命名: user/assistant → human/ai"""
        mapping = {
            "user": "human",
            "assistant": "ai",
            "human": "human",
            "ai": "ai",
            "tool": "tool",
        }
        return mapping.get(role, role)

    async def save_message(
        self,
        user_id: int,
        session_id: str,
        role: str,
        content: str,
        db: Optional[AsyncSession] = None,
    ) -> None:
        """
        保存一条消息
        同时存 Redis 和 MySQL
        """
        mapped_role = self._map_role(role)
        message = {
            "role": mapped_role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
        }
        await self._save_to_redis(session_id, message)

        if db is not None:
            try:
                record = ChatMessage(
                    session_id=session_id,
                    role=mapped_role,
                    content=content,
                )
                db.add(record)
                await db.commit()
            except SQLAlchemyError as e:
                await db.rollback()
                logger.error(f"MySQL写入失败{str(e)}", exc_info=True)

    async def _save_to_redis(self, session_id: str, message: Dict) -> None:
        """写入 Redis"""
        key = self._key(session_id)
        await cache_service.rpush(key, message)
        await cache_service.expire(key, self.redis_expire)

        current_len = await cache_service.llen(key)
        if current_len > self.max_history_limit:
            await cache_service.ltrim(key, -self.max_history_limit, -1)

    async def save_exchange(
        self,
        user_id: int,
        session_id: str,
        user_message: str,
        assistant_message: str,
        db: Optional[AsyncSession] = None,
    ) -> None:
        """保存一轮对话"""
        await self.save_message(user_id, session_id, "human", user_message, db)
        await self.save_message(user_id, session_id, "ai", assistant_message, db)

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> List[Dict]:
        """
        查询历史：Redis优先，Redis没有则查MySQL并回填
        """
        messages = await self._get_from_redis(session_id, limit)
        if messages:
            return messages

        if db is not None:
            messages = await self._get_from_mysql(session_id, limit, db)
            if messages:
                await self._backfill_redis(session_id, messages)

        return messages

    async def _get_from_redis(self, session_id: str, limit: Optional[int]) -> List[Dict]:
        """从 Redis 读取"""
        key = self._key(session_id)
        if not await cache_service.exists(key):
            return []

        if limit:
            raw = await cache_service.lrange(key, -limit, -1)
        else:
            raw = await cache_service.lrange(key, 0, -1)

        return raw

    async def _get_from_mysql(self, session_id: str, limit: Optional[int], db: AsyncSession) -> List[Dict]:
        """从 MySQL 读取"""
        stmt = select(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.create_time.asc())

        if limit:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        records = result.scalars().all()
        return [
            {"role": r.role, "content": r.content, "timestamp": str(r.create_time)}
            for r in records
        ]

    async def _backfill_redis(self, session_id: str, messages: List[Dict]) -> None:
        """MySQL数据回填到Redis"""
        key = self._key(session_id)
        for msg in messages:
            await cache_service.rpush(key, msg)
        await cache_service.expire(key, self.redis_expire)

    async def get_last_n_exchanges(self, session_id: str, n: int = 5, db: Optional[AsyncSession] = None,) -> List[Dict]:
        """获取最近N轮对话"""
        return await self.get_history(session_id, limit=n * 2, db=db)

    async def clear_history(self, session_id: str, db: Optional[AsyncSession] = None) -> bool:
        """清空指定会话"""
        await cache_service.delete(self._key(session_id))
        if db is not None:
            try:
                stmt = select(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                )
                result = await db.execute(stmt)
                records = result.scalars().all()
                for record in records:
                    await db.delete(record)
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"MySQL删除失败: {e}", exc_info=True)
        return True

    async def session_exists(self, session_id: str) -> bool:
        """判断会话是否存在"""
        return await cache_service.exists(self._key(session_id))

    async def get_session_count(self) -> int:
        """获取会话总数"""
        keys = await cache_service.keys("chat:history:*")
        return len(keys)

    async def get_all_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        keys = await cache_service.keys("chat:history:*")
        return [key.replace("chat:history:", "") for key in keys]

    async def get_history_length(self, session_id: str) -> int:
        """获取某会话的消息条数"""
        return await cache_service.llen(self._key(session_id))


chat_history_service = ChatHistoryService()
