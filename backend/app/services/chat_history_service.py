# backend/app/services/chat_history_service.py
from typing import List,Optional,Dict
from datetime import datetime
import redis,json
from sqlalchemy.orm import Session

from backend.app.models.chat_history import ChatHistory
from backend.app.common.logger import logger
from backend.app.common.exceptions import SQLAlchemyError

class ChatHistoryService:
    """
    对话历史管理服务
    Redis: 快速读写，做缓存
    MySQL: 持久化存储，做备份
    """
    def __init__(
        self,
        # Redis 配置
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_expire: int = 60 * 60 * 24 * 7,
        # MySQL 配置
        max_history_limit: int = 20,
    ):
        self.max_history_limit = max_history_limit
        self.redis_expire = redis_expire
        # Redis 连接
        self._redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
        )

    # ====== Key 生成 ======

    def _key(self, session_id: str) -> str:
        """生成Redis Key"""
        return f"chat:history:{session_id}"

    # ====== 保存 ======

    def save_message(
        self, 
        user_id:int,
        session_id: str, 
        role: str, 
        content: str,
        db:Optional[Session] = None,
    ) -> None:
        """
        保存一条消息
        同时存 Redis 和 MySQL
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "user_id":user_id,
        }
        self._save_to_redis(session_id, message)

        if db is not None:
            try:
                record = ChatHistory(
                    user_id = user_id,
                    session_id = session_id,
                    role = role,
                    content = content,
                )
                db.add(record)
                db.commit()
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"MySQL写入失败{str(e)}",exc_info=True)
                

    def _save_to_redis(self, session_id: str, message: Dict) -> None:
        """写入 Redis"""
        key = self._key(session_id)
        self._redis.rpush(key, json.dumps(message, ensure_ascii=False))
        self._redis.expire(key, self.redis_expire)

        # 裁剪
        current_len = self._redis.llen(key)
        if current_len > self.max_history_limit:
            self._redis.ltrim(key, -self.max_history_limit, -1)

    def save_exchange(
        self,
        user_id:int,
        session_id: str,
        user_message: str,
        assistant_message: str,
        db:Optional[Session] = None,
    ) -> None:
        """保存一轮对话"""
        self.save_message(user_id,session_id, "user", user_message,db)
        self.save_message(user_id,session_id, "assistant", assistant_message,db)

    # ====== 查询 ======

    def get_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> List[Dict]:
        """
        查询历史：Redis优先，Redis没有则查MySQL并回填
        """
        # 1. 先查 Redis
        messages = self._get_from_redis(session_id, limit)
        if messages:
            return messages

        if db is not None:
            # 2. Redis没有，查 MySQL
            messages = self._get_from_mysql(session_id, limit,db)

            # 3. 回填 Redis
            if messages:
                self._backfill_redis(session_id, messages)

        return messages

    def _get_from_redis(self, session_id: str, limit: Optional[int]) -> List[Dict]:
        """从 Redis 读取"""
        key = self._key(session_id)
        if not self._redis.exists(key):
            return []

        if limit:
            raw = self._redis.lrange(key, -limit, -1)
        else:
            raw = self._redis.lrange(key, 0, -1)

        return [json.loads(msg) for msg in raw]

    def _get_from_mysql(self, session_id: str, limit: Optional[int],db: Session) -> List[Dict]:
        """从 MySQL 读取"""
        query = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.created_at.asc())

        if limit:
            query = query.limit(limit)

        records = query.all()
        return [
            {"role": r.role, "content": r.content, "timestamp": str(r.created_at)}
            for r in records
        ]

    def _backfill_redis(self, session_id: str, messages: List[Dict]) -> None:
        """MySQL数据回填到Redis"""
        key = self._key(session_id)
        for msg in messages:
            self._redis.rpush(key, json.dumps(msg, ensure_ascii=False))
        self._redis.expire(key, self.redis_expire)

    def get_last_n_exchanges(self, session_id: str, n: int = 5,db: Optional[Session] = None,) -> List[Dict]:
        """获取最近N轮对话"""
        return self.get_history(session_id, limit=n * 2,db=db)

    # ====== 管理 ======

    def clear_history(self, session_id: str,db: Optional[Session] = None) -> bool:
        """清空指定会话"""
        self._redis.delete(self._key(session_id))
        if db is not None:
            try:
                db.query(ChatHistory).filter(
                    ChatHistory.session_id == session_id
                ).delete()
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"MySQL删除失败: {e}")
        return True

    def session_exists(self, session_id: str) -> bool:
        """判断会话是否存在"""
        return self._redis.exists(self._key(session_id)) > 0

    def get_session_count(self) -> int:
        """获取会话总数"""
        return len(self._redis.keys("chat:history:*"))

    def get_all_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        keys = self._redis.keys("chat:history:*")
        # 从key中提取session_id
        return [key.replace("chat:history:", "") for key in keys]

    def get_history_length(self, session_id: str) -> int:
        """获取某会话的消息条数"""
        return self._redis.llen(self._key(session_id))

# ====== 全局实例 ======

chat_history_service = ChatHistoryService()