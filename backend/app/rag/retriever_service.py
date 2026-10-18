# backend/app/rag/retriever_service.py
import time
from typing import Any

from langchain_core.documents import Document

from backend.app.common.logger import logger
from backend.app.core.cache import cache_service
from backend.app.rag.bm25_service import bm25_service
from backend.app.rag.hyde_service import hyde_service
from backend.app.rag.rerank_service import rerank_service
from backend.app.utils.vector_store import get_vector_store, vector_store_util


class RetrieverService:
    """
    RAG 检索服务
    """

    def __init__(
        self,
        vector_top_k: int = 10,  # 向量检索召回数量（单路召回，merge 后送 rerank）
        bm25_top_k: int = 10,  # BM25检索召回数量
        rerank_top_k: int = 10,  # 最终返回数量
        use_hybrid: bool = True,  # 是否混合检索
        use_rerank: bool = True,  # 是否启用Rerank
        rrf_k: int = 60,  # RRF 融合平滑常数
        score_threshold: float = 0.0,  # 相似度/相关度分数阈值（0=不过滤）
    ):
        self.vector_store = vector_store_util
        self.bm25 = bm25_service
        self.hyde = hyde_service
        self.rerank = rerank_service

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.rrf_k = rrf_k
        self.score_threshold = score_threshold

    # ====== 检索主入口 ======

    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        collection_name: str = "default_collection",
        use_rerank: bool | None = None,
        filter_dict: dict | None = None,
        skip_cache: bool = False,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """
        检索主入口
        Args:
            query: 用户查询
            strategy: 检索策略 ("vector" | "bm25" | "hybrid" | "hyde")
            collect_name:知识库集合名字
            use_rerank: 是否启用Rerank
            filter_dict: 元数据过滤条件
            skip_cache: 是否跳过缓存
            top_k: 最终返回的文档数量(不传则使用 rerank_top_k)
            score_threshold: 分数阈值，低于该值的文档将被过滤（0.0 表示不过滤）
        Returns:
            相关文档列表
        """
        effective_top_k = top_k if top_k is not None else self.rerank_top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )
        cache_key = f"search:{cache_service.generate_key_hash(query, strategy, collection_name, use_rerank, filter_dict, effective_top_k, effective_threshold)}"

        async def _do_retrieve():
            retrieve_start = time.time()
            should_rerank = use_rerank if use_rerank is not None else self.use_rerank

            if strategy == "vector":
                candidates = self.retrieve_vector(
                    query=query, collection_name=collection_name, filter_dict=filter_dict
                )
            elif strategy == "bm25":
                candidates = self.retrieve_bm25(query=query, collection_name=collection_name)
            elif strategy == "hyde":
                candidates = self.retrieve_hyde(
                    query=query, collection_name=collection_name, filter_dict=filter_dict
                )
            else:
                candidates = self.hybrid_retrieve(
                    query=query, collection_name=collection_name, filter_dict=filter_dict
                )

            # 相似度阈值过滤（只卡向量召回，前置到 rerank 之前，减少 reranker 候选）
            if effective_threshold > 0 and candidates:
                candidates = self._filter_by_score(candidates, effective_threshold)

            if should_rerank and len(candidates) > 0:
                candidates = self.rerank_results(
                    query=query, documents=candidates, top_k=effective_top_k
                )

            latency_ms = (time.time() - retrieve_start) * 1000
            logger.retrieval(
                kb_ids=[collection_name],
                query=query,
                strategy=strategy,
                results_count=len(candidates),
                top_k=effective_top_k,
                latency_ms=latency_ms,
            )

            return [self._doc_to_dict(doc) for doc in candidates[:effective_top_k]]

        cached_docs = await cache_service.get_or_compute(
            key=cache_key,
            ttl=300,
            compute_fn=_do_retrieve,
            skip_cache=skip_cache,
        )

        return [self._dict_to_doc(doc_dict) for doc_dict in cached_docs]

    def _doc_to_dict(self, doc: Document) -> dict:
        """将Document转为字典用于缓存"""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }

    def _dict_to_doc(self, doc_dict: dict) -> Document:
        """将字典转回Document"""
        return Document(
            page_content=doc_dict["page_content"],
            metadata=doc_dict.get("metadata", {}),
        )

    # ====== 各类检索策略 ======

    def retrieve_vector(
        self, query: str, collection_name: str, filter_dict: dict | None = None
    ) -> list[Document]:
        """
        纯向量检索
        Args:
            query: 用户查询
            collect_name:知识库集合名字
            filter_dict: 过滤条件
        Returns:
            相关文档列表
        """
        vector_store = get_vector_store(collection_name=collection_name)
        # 用带分数的检索，把余弦相似度写入 metadata["score"]，供阈值过滤前置使用
        scored = vector_store.search_with_score(
            query=query, top_k=self.vector_top_k, filter_dict=filter_dict
        )
        docs: list[Document] = []
        for item in scored:
            doc, distance = item if isinstance(item, (tuple, list)) else (item, None)
            if distance is not None:
                try:
                    # Chroma 默认 l2 距离（越小越相似）；bge 向量已归一化，
                    # 转成 0~1 余弦相似度：sim = 1 - d^2/2，便于按"相似度阈值"理解
                    sim = 1.0 - (float(distance) ** 2) / 2.0
                    doc.metadata["score"] = sim
                except (TypeError, ValueError):
                    pass
            docs.append(doc)
        self._tag_docs(docs, "vector")
        return docs

    def retrieve_bm25(
        self,
        query: str,
        collection_name: str | None = None,
    ) -> list[Document]:
        """
        纯bm25检索(按 collection 隔离索引)
        Args:
            query: 用户查询
            collection_name: 知识库集合名称(不传则检索默认集合)
        Returns:
            相关文档列表
        """
        docs = self.bm25.search(query=query, collection_name=collection_name, top_k=self.bm25_top_k)
        self._tag_docs(docs, "bm25")
        return docs

    def retrieve_hyde(
        self, query: str, collection_name: str, filter_dict: dict | None = None
    ) -> list[Document]:
        """
        HyDE增强检索
        Args:
            query: 用户查询
            collect_name:知识库集合名字
            filter_dict: 过滤条件
        Returns:
            相关文档列表
        """
        docs = self.hyde.retrieve_with_hyde(
            query=query,
            top_k=self.vector_top_k,
            filter_dict=filter_dict,
            collection_name=collection_name,
        )
        self._tag_docs(docs, "vector")
        return docs

    def hybrid_retrieve(
        self, query: str, collection_name: str, filter_dict: dict | None = None
    ) -> list[Document]:
        """
        混合检索：向量+bm25，合并去重
        Args:
            query: 用户查询
            collect_name:知识库集合名字
            filter_dict: 过滤条件
        Returns:
            相关文档列表
        """
        vector_docs = self.retrieve_vector(query, collection_name, filter_dict)
        bm25_docs = self.retrieve_bm25(query, collection_name=collection_name)
        return self.mergeResults(vector_docs, bm25_docs)

    # ====== 合并结果 ======

    def mergeResults(
        self,
        *doc_lists: list[Document],
        k: int = None,
    ) -> list[Document]:
        """
        使用倒数排名融合(RRF, Reciprocal Rank Fusion)合并多路检索结果。

        RRF 公式:  score(d) = Σ_list  1 / (k + rank_list(d))
        - rank_list(d): 文档 d 在某一路结果中的 1-based 排名
        - k: 平滑常数(默认取 self.rrf_k = 60, 业界常用值), 可通过参数 k 覆盖

        特点:
        1. 只依赖"排名"而非各检索器自身的绝对分数, 因此可无缝融合
           异质检索器(向量相似度 vs BM25 词频分数, 尺度不可直接比较)。
        2. 同一文档若出现在多路中, 分数累加; 排名越靠前, 贡献越大。
        3. 去重后按 RRF 总分降序返回, 并将分数写入 metadata["rrf_score"]。
        """
        if not doc_lists:
            return []

        rrf_k = k if k is not None else self.rrf_k
        rrf_scores: dict[Any, float] = {}
        doc_map: dict[Any, Document] = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list, start=1):  # 1-based 排名
                doc_id = doc.metadata.get("id") or hash(doc.page_content)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                # 保留首次出现的文档对象(向量支路在前, 故保留原始文本而非分词后文本)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # 按 RRF 总分降序排序
        ranked_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        merged = [doc_map[did] for did in ranked_ids]

        # 将 RRF 分数写回 metadata, 便于排查/观察
        for doc in merged:
            doc_id = doc.metadata.get("id") or hash(doc.page_content)
            doc.metadata["rrf_score"] = round(rrf_scores[doc_id], 6)

        return merged

    def _filter_by_score(
        self,
        documents: list[Document],
        score_threshold: float,
    ) -> list[Document]:
        """
        相似度阈值过滤（只卡「向量/语义召回」的文档，前置到 rerank 之前）。

        - 仅对 retrieval_source == "vector" 的文档按余弦相似度 score 过滤；
        - BM25 关键词召回的文档保守保留（避免误删关键词命中的相关分片，
          且 BM25 分数与余弦分数量纲不可比，单一阈值无法公平卡两路）；
        - 向量文档若缺失 score（如 HyDE 未回写分数），保守保留。

        该过滤在 RRF 合并之后、rerank 之前执行，可提前剔除语义弱的分片，
        减少送进 reranker 的候选数量（性能优化）。
        """
        filtered = []
        for doc in documents:
            if doc.metadata.get("retrieval_source") == "vector":
                score = doc.metadata.get("score")
                if score is None:
                    # 向量文档缺失分数，保守保留
                    filtered.append(doc)
                    continue
                try:
                    if float(score) >= score_threshold:
                        filtered.append(doc)
                except (TypeError, ValueError):
                    filtered.append(doc)
            else:
                # BM25 / 未知来源 → 保守保留
                filtered.append(doc)
        return filtered

    def rerank_results(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Rerank精排
        Args:
            query: 原始查询
            documents: 候选文档
            top_k: 重排后返回的文档数(不传则使用 self.rerank_top_k)
        Returns:
            重排序后的Top-K文档
        """
        return self.rerank.rerank(
            query=query,
            documents=documents,
            top_k=top_k if top_k is not None else self.rerank_top_k,
        )

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        score_threshold: float,
    ) -> list[tuple[Document, float]]:
        """
        Rerank并返回分数
        Args:
            query: 原始查询
            documents: 候选文档
            score_threshold: 分数阈值
        Returns:
            (文档, 分数) 元组列表
        """
        return self.rerank.rerank_with_score(
            query=query,
            documents=documents,
            top_k=self.rerank_top_k,
            score_threshold=score_threshold,
        )

    # ====== 便捷方法 ======

    def quick_retrieve(
        self,
        query: str,
    ) -> list[Document]:
        """快速检索（默认配置）"""
        return self.retrieve(query=query, strategy="hybrid")

    def get_status(
        self,
    ) -> dict:
        """获取各组件状态"""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "bm25": self.bm25.get_stats(),
        }

    # ====== 多知识库联合检索 ======

    def multi_kb_retrieve(
        self,
        query: str,
        kb_ids: list[str],
        strategy: str = "hybrid",
        use_rerank: bool | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """
        多知识库联合检索

        Args:
            query: 用户查询
            kb_ids: 知识库ID列表
            strategy: 检索策略 ("vector" | "bm25" | "hybrid" | "hyde")
            use_rerank: 是否启用 Rerank
            top_k: 最终返回的文档数量(不传则使用 rerank_top_k)
            score_threshold: 分数阈值，低于该值的文档将被过滤（0.0 表示不过滤）

        Returns:
            合并去重后的文档列表
        """
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank
        effective_top_k = top_k if top_k is not None else self.rerank_top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        t0 = time.time()

        if strategy == "vector":
            candidates = self.multi_kb_vector_search(query, kb_ids)
        elif strategy == "bm25":
            candidates = self.multi_kb_bm25_search(query, kb_ids)
        elif strategy == "hyde":
            candidates = self.multi_kb_hyde_search(query, kb_ids)
        else:  # hybrid
            candidates = self.multi_kb_hybrid_search(query, kb_ids)

        # 相似度阈值过滤（只卡向量召回，前置到 rerank 之前）
        if effective_threshold > 0 and candidates:
            candidates = self._filter_by_score(candidates, effective_threshold)

        if should_rerank and len(candidates) > 0:
            candidates = self.rerank_results(
                query=query, documents=candidates, top_k=effective_top_k
            )

        latency_ms = (time.time() - t0) * 1000
        logger.retrieval(
            kb_ids=kb_ids,
            query=query,
            strategy=strategy,
            results_count=len(candidates),
            top_k=effective_top_k,
            latency_ms=latency_ms,
            extra={"score_threshold": effective_threshold, "use_rerank": should_rerank},
        )

        return candidates[:effective_top_k]

    def multi_kb_vector_search(
        self,
        query: str,
        kb_ids: list[str],
    ) -> list[Document]:
        """
        多知识库向量检索
        遍历每个 KB，在对应的 Collection 中检索

        注意：每个知识库已经存储在独立的 collection 中，
        不需要额外的 kb_id filter，直接检索整个 collection。
        """
        all_docs: list[Document] = []

        for kb_id in kb_ids:
            collection_name = f"kb_{kb_id}"
            try:
                # 直接检索 collection，不需要额外的 filter
                # 因为每个知识库的文档已经独立存储在对应的 collection 中
                docs = self.retrieve_vector(
                    query=query,
                    collection_name=collection_name,
                    filter_dict=None,  # 不使用 filter
                )
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"知识库 {kb_id} 向量检索失败: {e}")
                continue

        return self._deduplicate(all_docs)

    def multi_kb_bm25_search(
        self,
        query: str,
        kb_ids: list[str],
    ) -> list[Document]:
        """
        多知识库 BM25 检索（结果标记 retrieval_source=bm25）
        """
        docs = self.bm25.search_by_kb_ids(
            query=query,
            kb_ids=kb_ids,
            top_k=self.bm25_top_k,
        )
        self._tag_docs(docs, "bm25")
        return docs

    def multi_kb_hybrid_search(
        self,
        query: str,
        kb_ids: list[str],
    ) -> list[Document]:
        """
        多知识库混合检索
        """
        vector_docs = self.multi_kb_vector_search(query, kb_ids)
        bm25_docs = self.multi_kb_bm25_search(query, kb_ids)
        return self.mergeResults(vector_docs, bm25_docs)

    def multi_kb_hyde_search(
        self,
        query: str,
        kb_ids: list[str],
    ) -> list[Document]:
        """
        多知识库 HyDE 检索。
        先用 LLM 针对 query 生成一篇假设性文档, 再在每一个 kb 对应的
        collection(kb_{kb_id}) 中做向量检索, 最后合并去重。
        (假设性文档只生成一次, 避免每个 KB 都调用一次 LLM)
        """
        try:
            hypothetical_doc = self.hyde.generate_hypothetical_doc(query)
        except Exception as e:
            logger.warning(f"HyDE 假设文档生成失败, 退化为向量检索: {e}")
            return self.multi_kb_vector_search(query, kb_ids)

        all_docs: list[Document] = []
        for kb_id in kb_ids:
            try:
                collection_name = f"kb_{kb_id}"
                docs = self.retrieve_vector(
                    query=hypothetical_doc,
                    collection_name=collection_name,
                    filter_dict=None,
                )
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"知识库 {kb_id} HyDE 向量检索失败: {e}")
                continue

        return self._deduplicate(all_docs)

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        """文档去重"""
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            doc_id = (
                doc.metadata.get("chunk_id") or doc.metadata.get("id") or hash(doc.page_content)
            )
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique

    @staticmethod
    def _tag_docs(docs: list[Document], source: str) -> list[Document]:
        """给召回文档统一打上来源标记，便于阈值过滤时区分向量/BM25。"""
        for doc in docs:
            doc.metadata["retrieval_source"] = source
        return docs


# ====== 全局实例 ======

retriever_service = RetrieverService()
