# backend/app/agent/memory/long_term.py
"""
长期记忆模块
MySQL + ChromaDB 混合架构：
- MySQL: 存储文本内容、元数据、时间戳
- ChromaDB: 存储向量索引，用于语义检索
"""

from datetime import datetime, timedelta
from typing import Any

import chromadb
from chromadb.config import Settings
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.logger import logger
from backend.app.core.config import settings
from backend.app.models.long_term_memory import UserMemory
from backend.app.utils.models_factory import model_factory
from backend.db.db_config import AsyncSessionLocal

# ====== ChromaDB 配置 ======
MEMORY_CHROMA_PATH = settings.VECTOR_STORE_PATH + "/memory"
MEMORY_COLLECTION_NAME = "user_long_term_memory"


class LongTermMemoryService:
    """
    长期记忆服务

    功能：
    - memorize: 写入长期记忆（MySQL + ChromaDB 双写）
    - recall: 召回相关记忆（ChromaDB 向量检索 → MySQL 查详情）
    - forget: 遗忘机制（清理低重要度/过期记忆）
    - importance_score: 评估记忆重要度
    """

    def __init__(self):
        # ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(
            path=MEMORY_CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(name=MEMORY_COLLECTION_NAME)

        # Embedding 模型
        self.embedder = model_factory.get_embedding_model(provider="bge")

    @staticmethod
    def _normalize_memory_type(memory_type: str) -> str:
        """兼容旧的 memory_type 命名：fact/task → preference/event"""
        mapping = {
            "fact": "preference",
            "task": "event",
            "preference": "preference",
            "event": "event",
        }
        return mapping.get(memory_type, "preference")

    async def memorize(
        self,
        user_id: str,
        content: str,
        memory_type: str = "preference",
        importance: float = None,
        db: AsyncSession = None,
        source: str = "auto",
        source_session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """
        写入长期记忆

        Args:
            user_id: 用户 ID
            content: 记忆文本
            memory_type: 记忆类型（preference/event，兼容旧值fact/task）
            importance: 重要度评分（可选，自动评估）
            db: 数据库会话
            source: 来源 auto(AI自动提取) / explicit(用户说记住)
            source_session_id: 来源会话 ID
            agent_id: Agent 专属记忆时绑定的 Agent ID（None=全局记忆）

        Returns:
            {"memory_id": int, "chroma_id": str}
        """
        # 兼容旧类型名
        memory_type = self._normalize_memory_type(memory_type)

        # 自动评估重要度
        if importance is None:
            importance = await self._evaluate_importance(content, memory_type)

        # 低重要度记忆不写入（早期遗忘）
        if importance < 0.3:
            return {"memory_id": None, "chroma_id": None, "reason": "importance_below_threshold"}

        # 生成向量
        embedding = self.embedder.embed_query(content)

        # 【修复】主键必须显式生成：user_memory.id 是 String(12) 主键且表结构无默认值，
        # 此前不传 id → MySQL 报 (1364, "Field 'id' doesn't have a default value")，
        # 上线以来长期记忆**从未真正落库**（只在日志里刷 warning）。
        from backend.app.services.auth import generate_id

        memory_obj = UserMemory(
            id=generate_id(),
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            is_global=agent_id is None,
            agent_id=agent_id,
            source=source,
            source_session_id=source_session_id,
            status="active",
            create_time=datetime.now(),
            access_count=0,
        )

        if db is None:
            async with AsyncSessionLocal() as session:
                session.add(memory_obj)
                await session.commit()
                await session.refresh(memory_obj)
        else:
            db.add(memory_obj)
            await db.commit()
            await db.refresh(memory_obj)

        # 写入 ChromaDB
        chroma_id = f"{user_id}_{memory_obj.id}_{datetime.now().timestamp()}"
        self.collection.add(
            ids=[chroma_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "user_id": user_id,
                    "memory_id": memory_obj.id,
                    "memory_type": memory_type,
                    "importance": importance,
                    "create_time": str(datetime.now()),
                    "status": "active",
                }
            ],
        )

        # 更新 MySQL 的 chroma_id
        memory_obj.chroma_id = chroma_id
        if db is None:
            async with AsyncSessionLocal() as session:
                await session.commit()
        else:
            await db.commit()

        return {
            "memory_id": memory_obj.id,
            "chroma_id": chroma_id,
            "importance": importance,
        }

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_types: list[str] = None,
        db: AsyncSession = None,
    ) -> list[dict[str, Any]]:
        """
        召回相关长期记忆

        Args:
            user_id: 用户 ID
            query: 查询文本
            top_k: 返回数量
            memory_types: 记忆类型过滤（可选，兼容旧值）
            db: 数据库会话

        Returns:
            [{"content": str, "importance": float, "memory_type": str, ...}]
        """
        # 兼容旧类型名
        if memory_types:
            memory_types = [self._normalize_memory_type(t) for t in memory_types]

        # 生成查询向量
        query_embedding = self.embedder.embed_query(query)

        # ChromaDB 向量检索
        # 【修复】Chroma 的 where 只接受"恰好一个操作符"：多个平铺键会直接抛
        # ValueError: Expected where to have exactly one operator，
        # 导致长期记忆召回**必然失败**（写入修好前也一直没暴露）。
        # 正确写法是用 $and 显式组合。
        conditions: list[dict] = [{"user_id": user_id}, {"status": "active"}]
        if memory_types:
            conditions.append({"memory_type": {"$in": memory_types}})
        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        # 解析结果
        memory_ids = []
        if results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                memory_ids.append(meta.get("memory_id"))

        # MySQL 查完整信息
        if not memory_ids:
            return []

        memories = []
        stmt = select(UserMemory).where(
            UserMemory.id.in_(memory_ids), UserMemory.status == "active"
        )
        if db is None:
            async with AsyncSessionLocal() as session:
                result = await session.execute(stmt)
                memory_objs = result.scalars().all()
        else:
            result = await db.execute(stmt)
            memory_objs = result.scalars().all()

        for obj in memory_objs:
            memories.append(
                {
                    "content": obj.content,
                    "importance": obj.importance,
                    "memory_type": obj.memory_type,
                    "created_at": str(obj.create_time),
                    "access_count": (obj.access_count or 0) + 1,
                    "memory_id": obj.id,
                }
            )

        # 【修复】命中统计必须落库：此前只改了 ORM 对象却从未 commit，
        # last_accessed / access_count 永远停在初始值（遗忘策略因此失真）。
        try:
            update_stmt = (
                update(UserMemory)
                .where(UserMemory.id.in_(memory_ids))
                .values(
                    last_accessed=datetime.now(),
                    access_count=UserMemory.access_count + 1,
                )
            )
            if db is None:
                async with AsyncSessionLocal() as session:
                    await session.execute(update_stmt)
                    await session.commit()
            else:
                await db.execute(update_stmt)
                await db.commit()
        except Exception as e:
            logger.warning(f"更新长期记忆命中统计失败（不影响召回）: {e}")

        return memories

    async def forget(
        self,
        user_id: str = None,
        days_old: int = 90,
        importance_threshold: float = 0.3,
        db: AsyncSession = None,
    ) -> dict[str, Any]:
        """
        遗忘机制：清理低重要度/过期记忆

        Args:
            user_id: 用户 ID（可选，清理指定用户的记忆）
            days_old: 过期天数阈值
            importance_threshold: 重要度阈值
            db: 数据库会话

        Returns:
            {"deleted_mysql": int, "deleted_chroma": int}
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # 构建查询条件
        conditions = [
            UserMemory.status == "active",
            (UserMemory.importance < importance_threshold) | (UserMemory.create_time < cutoff_date),
        ]
        if user_id:
            conditions.append(UserMemory.user_id == user_id)

        # 查询待遗忘记忆
        stmt = select(UserMemory).where(*conditions)
        if db is None:
            async with AsyncSessionLocal() as session:
                result = await session.execute(stmt)
                memories_to_delete = result.scalars().all()
        else:
            result = await db.execute(stmt)
            memories_to_delete = result.scalars().all()

        # ChromaDB 删除
        chroma_ids = [m.chroma_id for m in memories_to_delete if m.chroma_id]
        if chroma_ids:
            self.collection.delete(ids=chroma_ids)

        # MySQL 标记为 forgotten（软删除）
        memory_ids = [m.id for m in memories_to_delete]
        if memory_ids:
            update_stmt = (
                update(UserMemory).where(UserMemory.id.in_(memory_ids)).values(status="forgotten")
            )
            if db is None:
                async with AsyncSessionLocal() as session:
                    await session.execute(update_stmt)
                    await session.commit()
            else:
                await db.execute(update_stmt)
                await db.commit()

        return {
            "deleted_mysql": len(memory_ids),
            "deleted_chroma": len(chroma_ids),
            "reason": "low_importance_or_expired",
        }

    async def _evaluate_importance(self, content: str, memory_type: str) -> float:
        """
        评估记忆重要度

        Args:
            content: 记忆内容
            memory_type: 记忆类型（规范化后的值）

        Returns:
            重要度评分（0-1）
        """
        # 简化规则：preference 高重要度，event 中高
        type_scores = {
            "preference": 0.8,
            "event": 0.7,
        }
        base_score = type_scores.get(memory_type, 0.5)

        # 基于内容长度微调（长内容可能更重要）
        length_bonus = min(len(content) / 500, 0.2)

        return min(base_score + length_bonus, 1.0)

    async def get_user_stats(
        self,
        user_id: str,
        db: AsyncSession = None,
    ) -> dict[str, Any]:
        """
        获取用户记忆统计信息

        Args:
            user_id: 用户 ID
            db: 数据库会话

        Returns:
            {"total": int, "by_type": dict, "avg_importance": float}
        """
        stmt = select(UserMemory).where(
            UserMemory.user_id == user_id, UserMemory.status == "active"
        )
        if db is None:
            async with AsyncSessionLocal() as session:
                result = await session.execute(stmt)
                memories = result.scalars().all()
        else:
            result = await db.execute(stmt)
            memories = result.scalars().all()

        # 统计
        by_type = {}
        total_importance = 0
        for m in memories:
            by_type[m.memory_type] = by_type.get(m.memory_type, 0) + 1
            total_importance += m.importance

        return {
            "total": len(memories),
            "by_type": by_type,
            "avg_importance": total_importance / len(memories) if memories else 0,
        }


# ====== 全局实例 ======

long_term_memory = LongTermMemoryService()
