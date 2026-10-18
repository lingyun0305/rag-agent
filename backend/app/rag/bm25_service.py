# backend/app/rag/bm25_service.py
from typing import Any

import jieba
from langchain_core.documents import Document

from backend.app.common.logger import logger

# 检查 rank_bm25 是否可用，不可用时降级（仅缓存文档、不构建索引）。
_BM25_AVAILABLE = False
try:
    from langchain_community.retrievers import BM25Retriever

    _BM25_AVAILABLE = True
except ImportError:
    pass


class _BM25Collection:
    """
    单个知识库 collection 的 BM25 索引(按 collection_name 隔离)。
    每个知识库对应一个 ChromaDB collection(kb_{kb_id}), 其 BM25 索引也应相互独立,
    避免不同知识库的文档互相串味, 也避免清空某个知识库时误伤其它知识库。
    """

    def __init__(self, top_k: int = 6):
        self.top_k = top_k
        self._documents: list[Document] = []
        self._retriever = None

    @staticmethod
    def _tokenize(text: str) -> str:
        """中文分词（jieba分词后用空格拼接）"""
        return " ".join(jieba.lcut(text))

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        self._documents.extend(docs)
        self._build_index()

    def delete_by_filter(self, filter_dict: dict[str, Any]) -> None:
        if not filter_dict or not self._documents:
            return

        def match(doc: Document) -> bool:
            for key, value in filter_dict.items():
                if doc.metadata.get(key) != value:
                    return True
            return False

        self._documents = [doc for doc in self._documents if match(doc)]
        self._build_index()

    def delete_by_ids(self, ids: list[str]) -> None:
        id_set = set(ids)
        self._documents = [doc for doc in self._documents if doc.metadata.get("id") not in id_set]
        self._build_index()

    def clear(self) -> None:
        self._documents = []
        self._retriever = None

    def _build_index(self) -> None:
        """构建/重建 BM25 索引"""
        if not _BM25_AVAILABLE:
            return
        if not self._documents:
            self._retriever = None
            return
        tokenized_docs = [
            Document(page_content=self._tokenize(doc.page_content), metadata=doc.metadata)
            for doc in self._documents
        ]
        self._retriever = BM25Retriever.from_documents(tokenized_docs)
        self._retriever.k = self.top_k

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        if self._retriever is None:
            return []
        if top_k is not None:
            self._retriever.k = top_k
        return self._retriever.invoke(self._tokenize(query))

    def is_ready(self) -> bool:
        return self._retriever is not None

    def doc_count(self) -> int:
        return len(self._documents)


class BM25Service:
    """
    BM25 关键词检索服务(按 collection 隔离索引)。

    索引按 collection_name 分别维护在 self._collections 中。
    如果 rank_bm25 未安装, 自动降级为仅缓存文档、不构建索引。
    """

    # ====== 初始化 ======
    def __init__(self, top_k: int = 6):
        self.top_k = top_k
        # collection_name -> _BM25Collection
        self._collections: dict[str, _BM25Collection] = {}

    def _get_collection(self, collection_name: str) -> _BM25Collection:
        col = self._collections.get(collection_name)
        if col is None:
            col = _BM25Collection(top_k=self.top_k)
            self._collections[collection_name] = col
        return col

    # ====== 分词处理 ======

    @staticmethod
    def _tokenize(text: str) -> str:
        """中文分词（jieba分词后用空格拼接）"""
        return " ".join(jieba.lcut(text))

    # ====== 索引构建 ======

    def add_documents(self, docs: list[Document], collection_name: str = "default") -> None:
        """向指定 collection 添加文档并重建其 BM25 索引"""
        if not docs:
            return
        self._get_collection(collection_name).add(docs)

    def add_document(self, doc: Document, collection_name: str = "default") -> None:
        """向指定 collection 添加单个文档"""
        self.add_documents([doc], collection_name=collection_name)

    def delete_by_ids(self, ids: list[str], collection_name: str = "default") -> None:
        """根据ID删除指定 collection 中的文档（删除后重建索引）"""
        self._get_collection(collection_name).delete_by_ids(ids)

    def delete_by_filter(
        self, filter_dict: dict[str, Any], collection_name: str = "default"
    ) -> None:
        """根据元数据条件删除指定 collection 中的文档"""
        self._get_collection(collection_name).delete_by_filter(filter_dict)

    def clear(self, collection_name: str | None = None) -> None:
        """
        清空 BM25 索引。
        - collection_name 为 None: 清空所有 collection(保留向后兼容语义)
        - 指定 collection_name: 仅清空该知识库对应的索引
        """
        if collection_name is None:
            self._collections.clear()
        else:
            self._collections.pop(collection_name, None)

    # ====== 检索查询 ======

    def search(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
    ) -> list[Document]:
        """在指定 collection 内做 BM25 关键词检索"""
        return self._get_collection(collection_name).search(query, top_k=top_k)

    def search_with_filter(
        self,
        query: str,
        filter_dict: dict[str, Any] | None = None,
        top_k: int | None = None,
        collection_name: str = "default",
    ) -> list[Document]:
        """
        带过滤条件的 BM25 检索(在指定 collection 内先检索再过滤)
        """
        results = self.search(query, collection_name=collection_name, top_k=top_k)
        if not filter_dict:
            return results

        filtered = []
        for doc in results:
            match = True
            for key, value in filter_dict.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    def search_by_kb_ids(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int | None = None,
        auto_rebuild: bool = True,
    ) -> list[Document]:
        """
        按知识库ID列表检索。
        逐个在 kb_{kb_id} 各自的 collection 中检索并合并去重,
        保证只在指定知识库范围内召回, 不会串到其它知识库。

        auto_rebuild: 当某 KB 的 BM25 索引为空但向量库有数据时，
                      自动从 ChromaDB 重建索引（懒加载，解决重启丢失问题）。
        """
        if not kb_ids:
            return []

        effective_top_k = top_k or self.top_k
        all_results: list[Document] = []
        for kb_id in kb_ids:
            col_name = f"kb_{kb_id}"
            col = self._collections.get(col_name)
            if col is None or not col.is_ready():
                # 懒重建：索引为空时尝试从向量库回填
                if auto_rebuild:
                    rebuilt = self.rebuild_from_vector_store(col_name)
                    if rebuilt > 0:
                        logger.info(f"[BM25] 懒重建 {col_name}: {rebuilt} 条文档")
                        col = self._collections.get(col_name)
                if col is None or not col.is_ready():
                    continue
            all_results.extend(col.search(query, top_k=effective_top_k))

        # 合并去重(同一 chunk 可能出现在多个返回中)
        seen: set = set()
        unique: list[Document] = []
        for doc in all_results:
            doc_id = (
                doc.metadata.get("chunk_id") or doc.metadata.get("id") or hash(doc.page_content)
            )
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)

        return unique[:effective_top_k]

    def search_multi_kb(
        self,
        query: str,
        kb_ids: list[str],
        top_k: int | None = None,
    ) -> list[Document]:
        """多知识库联合检索(按 collection 隔离, 等价于 search_by_kb_ids)"""
        return self.search_by_kb_ids(query, kb_ids, top_k=top_k)

    # ====== 索引重建（从向量库回填） ======

    def rebuild_from_vector_store(self, collection_name: str) -> int:
        """
        从 ChromaDB 向量库读取全量文档，重建指定 collection 的 BM25 内存索引。
        用于服务重启后恢复 BM25 索引（BM25 是纯内存索引，重启即丢失）。

        Returns:
            成功重建的文档数量（0 表示无需重建或重建失败）
        """
        try:
            from backend.app.utils.vector_store import get_vector_store

            vs = get_vector_store(collection_name=collection_name)
            docs = vs.get_all_documents()
            if not docs:
                return 0
            self.add_documents(docs, collection_name=collection_name)
            logger.info(f"[BM25] 索引重建完成: {collection_name}, {len(docs)} 条文档")
            return len(docs)
        except Exception as e:
            logger.warning(f"[BM25] 索引重建失败 ({collection_name}): {e}")
            return 0

    def rebuild_all(self, kb_ids: list[str] = None) -> dict[str, int]:
        """
        批量重建 BM25 索引。

        Args:
            kb_ids: 指定知识库 ID 列表；为空则扫描向量库中所有 kb_* 集合。

        Returns:
            {collection_name: rebuilt_doc_count} 的字典
        """
        results: dict[str, int] = {}
        if kb_ids:
            targets = [f"kb_{kid}" for kid in kb_ids]
        else:
            try:
                from backend.app.utils.vector_store import vector_store_util

                all_cols = vector_store_util.list_collections()
                targets = [c["name"] for c in all_cols if c["name"].startswith("kb_")]
            except Exception:
                targets = []

        for col_name in targets:
            count = self.rebuild_from_vector_store(col_name)
            results[col_name] = count

        logger.info(f"[BM25] 批量重建完成: {results}")
        return results

    # ====== 状态查询 ======

    def get_status(self) -> dict[str, Any]:
        """获取 BM25 索引统计信息(聚合所有 collection)"""
        total_docs = sum(col.doc_count() for col in self._collections.values())
        index_ready = any(col.is_ready() for col in self._collections.values())
        return {
            "document_count": total_docs,
            "collection_count": len(self._collections),
            "index_ready": index_ready,
            "bm25_available": _BM25_AVAILABLE,
        }


# ====== 全局实例 ======

bm25_service = BM25Service()
