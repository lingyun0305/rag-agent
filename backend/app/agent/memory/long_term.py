# backend/app/agent/memory/long_term.py
"""
长期记忆模块
MySQL + ChromaDB 混合架构：
- MySQL: 存储文本内容、元数据、时间戳
- ChromaDB: 存储向量索引，用于语义检索
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
import chromadb
from chromadb.config import Settings

from backend.db.db_config import AsyncSessionLocal, Base
from backend.app.models.long_term_memory import UserMemory
from backend.app.utils.models_factory import model_factory
from backend.app.core.config import settings


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
            path=MEMORY_CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME
        )

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
        user_id: int,
        content: str,
        memory_type: str = "preference",
        importance: float = None,
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
        """
        写入长期记忆

        Args:
            user_id: 用户 ID
            content: 记忆文本
            memory_type: 记忆类型（preference/event，兼容旧值fact/task）
            importance: 重要度评分（可选，自动评估）
            db: 数据库会话

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

        # 写入 MySQL
        memory_obj = UserMemory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
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
            metadatas=[{
                "user_id": user_id,
                "memory_id": memory_obj.id,
                "memory_type": memory_type,
                "importance": importance,
                "create_time": str(datetime.now()),
                "status": "active",
            }]
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
        user_id: int,
        query: str,
        top_k: int = 5,
        memory_types: List[str] = None,
        db: AsyncSession = None,
    ) -> List[Dict[str, Any]]:
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
        where_filter = {"user_id": user_id, "status": "active"}
        if memory_types:
            where_filter["memory_type"] = {"$in": memory_types}

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
                UserMemory.id.in_(memory_ids),
                UserMemory.status == "active"
            )
        if db is None:
            async with AsyncSessionLocal() as session:
                result = await session.execute(stmt)
                memory_objs = result.scalars().all()
        else:
            result = await db.execute(stmt)
            memory_objs = result.scalars().all()

        for obj in memory_objs:
            memories.append({
                "content": obj.content,
                "importance": obj.importance,
                "memory_type": obj.memory_type,
                "created_at": str(obj.create_time),
                "access_count": obj.access_count,
                "memory_id": obj.id,
            })

            # 更新访问记录
            obj.last_accessed = datetime.now()
            obj.access_count += 1

        return memories

    async def forget(
        self,
        user_id: int = None,
        days_old: int = 90,
        importance_threshold: float = 0.3,
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
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
            (UserMemory.importance < importance_threshold) |
            (UserMemory.create_time < cutoff_date)
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
            update_stmt = update(UserMemory).where(
                UserMemory.id.in_(memory_ids)
            ).values(status="forgotten")
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
            "reason": "low_importance_or_expired"
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
        user_id: int,
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
        """
        获取用户记忆统计信息

        Args:
            user_id: 用户 ID
            db: 数据库会话

        Returns:
            {"total": int, "by_type": dict, "avg_importance": float}
        """
        stmt = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.status == "active"
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
