# backend/app/rag/agent_rag_service.py
"""
Agent 级 RAG 检索服务
====================
职责：
1. 根据 AgentID 获取关联的知识库列表
2. 验证用户对知识库的访问权限
3. 跨多个知识库进行联合检索
4. 支持 vector/bm25/hybrid 三种检索策略
5. 合并结果、去重、Rerank 精排
"""

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.logger import logger
from backend.app.models.agent import AgentKnowledgeBase
from backend.app.models.knowledge import KbMember, KnowledgeBase
from backend.app.rag.rerank_service import rerank_service
from backend.app.rag.retriever_service import retriever_service


class AgentRAGService:
    """
    Agent RAG 检索编排服务

    使用方式：
        service = AgentRAGService()
        docs = await service.retrieve(
            agent_id="xxx",
            user_id="yyy",
            query="如何申请报销？",
            strategy="hybrid",
            db=db
        )
    """

    def __init__(
        self,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        rerank_top_k: int = 6,
    ):
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k

    # ====== 核心入口 ======

    async def retrieve(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        strategy: str = "hybrid",
        use_rerank: bool = True,
        db: AsyncSession | None = None,
    ) -> list[Document]:
        """
        Agent 级联合检索

        Args:
            agent_id: Agent ID
            user_id: 当前用户 ID
            query: 查询文本
            strategy: 检索策略 ("vector" | "bm25" | "hybrid")
            use_rerank: 是否启用 Rerank
            db: 数据库会话

        Returns:
            合并去重后的文档列表
        """
        # 1. 获取 Agent 关联的知识库列表
        kb_ids = await self._get_agent_kb_ids(agent_id, db)
        if not kb_ids:
            logger.info(f"Agent {agent_id} 未关联任何知识库")
            return []

        # 2. 验证用户权限，过滤无权限的 KB
        authorized_kb_ids = await self._filter_authorized_kbs(kb_ids, user_id, db)
        if not authorized_kb_ids:
            logger.warning(f"用户 {user_id} 对 Agent {agent_id} 的知识库无访问权限")
            return []

        logger.info(
            f"Agent RAG 检索: agent={agent_id}, "
            f"total_kbs={len(kb_ids)}, authorized_kbs={len(authorized_kb_ids)}, "
            f"strategy={strategy}"
        )

        # 3. 执行检索
        if strategy == "vector":
            candidates = await self._multi_kb_vector_search(query, authorized_kb_ids, db)
        elif strategy == "bm25":
            candidates = await self._multi_kb_bm25_search(query, authorized_kb_ids, db)
        else:  # hybrid
            candidates = await self._multi_kb_hybrid_search(query, authorized_kb_ids, db)

        # 4. Rerank 精排
        if use_rerank and len(candidates) > 0:
            candidates = rerank_service.rerank(
                query=query,
                documents=candidates,
                top_k=self.rerank_top_k,
            )

        logger.info(f"Agent RAG 检索完成: 返回 {len(candidates)} 个文档")
        return candidates[: self.rerank_top_k]

    # ====== 多知识库联合检索 ======

    async def _multi_kb_vector_search(
        self,
        query: str,
        kb_ids: list[str],
        db: AsyncSession,
    ) -> list[Document]:
        """
        多知识库语义检索
        遍历每个 KB，在对应的 Collection 中检索
        """
        all_docs: list[Document] = []

        for kb_id in kb_ids:
            try:
                collection_name = self._get_kb_collection_name(kb_id)
                docs = retriever_service.retrieve_vector(
                    query=query,
                    collection_name=collection_name,
                    filter_dict={"kb_id": kb_id},
                )
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"KB {kb_id} 向量检索失败: {e}")
                continue

        return self._deduplicate_docs(all_docs)

    async def _multi_kb_bm25_search(
        self,
        query: str,
        kb_ids: list[str],
        db: AsyncSession,
    ) -> list[Document]:
        """
        多知识库关键词检索
        使用 BM25 按 kb_id 过滤
        """
        from backend.app.rag.bm25_service import bm25_service

        docs = bm25_service.search_by_kb_ids(
            query=query,
            kb_ids=kb_ids,
            top_k=self.bm25_top_k,
        )
        return self._deduplicate_docs(docs)

    async def _multi_kb_hybrid_search(
        self,
        query: str,
        kb_ids: list[str],
        db: AsyncSession,
    ) -> list[Document]:
        """
        多知识库混合检索
        并行执行 vector + bm25，合并去重
        """
        vector_docs = await self._multi_kb_vector_search(query, kb_ids, db)
        bm25_docs = await self._multi_kb_bm25_search(query, kb_ids, db)

        # 合并去重
        all_docs = self._merge_and_deduplicate(vector_docs, bm25_docs)
        return all_docs

    # ====== Agent-KB 关联查询 ======

    async def _get_agent_kb_ids(
        self,
        agent_id: str,
        db: AsyncSession | None,
    ) -> list[str]:
        """获取 Agent 关联的所有知识库 ID"""
        if not db:
            return []

        result = await db.execute(
            select(AgentKnowledgeBase.kb_id).where(AgentKnowledgeBase.agent_id == agent_id)
        )
        return [row[0] for row in result.fetchall()]

    # ====== 权限验证 ======

    async def _filter_authorized_kbs(
        self,
        kb_ids: list[str],
        user_id: str,
        db: AsyncSession | None,
    ) -> list[str]:
        """
        过滤用户有权限访问的知识库

        权限规则：
        1. 用户是 KB 的 owner
        2. 用户在 kb_member 表中（active 状态）
        """
        if not db:
            return []

        authorized: set[str] = set()

        for kb_id in kb_ids:
            if await self._check_kb_permission(kb_id, user_id, db):
                authorized.add(kb_id)

        return list(authorized)

    async def _check_kb_permission(
        self,
        kb_id: str,
        user_id: str,
        db: AsyncSession,
    ) -> bool:
        """检查用户对单个 KB 的访问权限"""
        # 1. 检查是否为 KB 的 owner
        kb_result = await db.execute(
            select(KnowledgeBase.owner_id).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.status == "active",
            )
        )
        kb = kb_result.scalar_one_or_none()
        if kb and kb == user_id:
            return True

        # 2. 检查是否为 KB 的成员
        member_result = await db.execute(
            select(KbMember.id).where(
                KbMember.kb_id == kb_id,
                KbMember.member_id == user_id,
                KbMember.member_type == "user",
                KbMember.status == "active",
            )
        )
        if member_result.scalar_one_or_none():
            return True

        return False

    # ====== 工具方法 ======

    @staticmethod
    def _get_kb_collection_name(kb_id: str) -> str:
        """获取知识库对应的 Collection 名称"""
        return f"kb_{kb_id}"

    @staticmethod
    def _deduplicate_docs(docs: list[Document]) -> list[Document]:
        """文档去重（基于内容 hash）"""
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id") or hash(doc.page_content)
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique

    @staticmethod
    def _merge_and_deduplicate(*doc_lists: list[Document]) -> list[Document]:
        """合并多个文档列表并去重"""
        seen: set[str] = set()
        merged: list[Document] = []
        max_len = max(len(docs) for docs in doc_lists) if doc_lists else 0

        for i in range(max_len):
            for docs in doc_lists:
                if i < len(docs):
                    doc = docs[i]
                    doc_id = doc.metadata.get("chunk_id") or hash(doc.page_content)
                    if doc_id not in seen:
                        seen.add(doc_id)
                        merged.append(doc)
        return merged


# ====== 全局实例 ======

agent_rag_service = AgentRAGService()
