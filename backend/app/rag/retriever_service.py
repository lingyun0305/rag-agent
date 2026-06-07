# backend/app/rag/retriever_service.py
from typing import List,Optional,Dict,Set
from langchain_core.documents import Document

from backend.app.rag.bm25_service import bm25_service
from backend.app.rag.hyde_service import hyde_service
from backend.app.rag.rerank_service import rerank_service
from backend.app.utils.vector_store import vector_store_util,get_vector_store

class RetrieverService:
    """
    RAG 检索服务
    """
    def __init__(
        self,
        vector_top_k: int = 20,      # 向量检索召回数量
        bm25_top_k: int = 20,        # BM25检索召回数量
        rerank_top_k: int = 6,       # 最终返回数量
        use_hybrid: bool = True,     # 是否混合检索
        use_rerank: bool = True,     # 是否启用Rerank
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

    # ====== 检索主入口 ======

    def retrieve(
        self,
        query:str,
        strategy:str = "hybrid",
        collection_name:str = "default_collection",
        use_rerank:Optional[bool] = None,
        filter_dict:Optional[Dict] = None,
    )->List[Document]:
        """
        检索主入口
        Args:
            query: 用户查询
            strategy: 检索策略 ("vector" | "bm25" | "hybrid" | "hyde")
            collect_name:知识库集合名字
            use_rerank: 是否启用Rerank
            filter_dict: 元数据过滤条件
        Returns:
            相关文档列表
        """
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank

        if strategy == "vector":
            candidates = self.retrieve_vector(query=query,collection_name = collection_name,filter_dict=filter_dict)
        elif strategy == "bm25":
            candidates = self.retrieve_bm25(query=query)
        elif strategy == "hyde":
            candidates = self.retrieve_hyde(query=query,collection_name = collection_name,filter_dict=filter_dict)
        else:
            candidates = self.hybrid_retrieve(query=query,collection_name = collection_name,filter_dict=filter_dict)

        if should_rerank and len(candidates)>0:
            self.rerank_results(query=query,documents=candidates)
        
        return candidates[:self.rerank_top_k]
    
    # ====== 各类检索策略 ======

    def retrieve_vector(
        self,
        query:str,
        collection_name:str,
        filter_dict:Optional[Dict] = None
    )->List[Document]:
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
        return vector_store.search(
            query=query,
            top_k=self.vector_top_k,
            filter_dict=filter_dict
        )
    
    def retrieve_bm25(
        self,
        query:str,
    )->List[Document]:
        """
        纯bm25检索
        Args:
            query: 用户查询
        Returns:
            相关文档列表
        """
        return self.bm25.search(query=query,top_k=self.bm25_top_k)
    
    def retrieve_hyde(
        self,
        query:str,
        collection_name:str,
        filter_dict:Optional[Dict] = None
    )->List[Document]:
        """
        HyDE增强检索
        Args:
            query: 用户查询
            collect_name:知识库集合名字
            filter_dict: 过滤条件
        Returns:
            相关文档列表
        """
        return self.hyde.retrieve_with_hyde(
            query=query,
            top_k=self.vector_top_k,
            filter_dict=filter_dict,
            collection_name=collection_name
        )
    
    def hybrid_retrieve(
        self,
        query:str,
        collection_name:str,
        filter_dict:Optional[Dict] = None
    )->List[Document]:
        """
        混合检索：向量+bm25，合并去重
        Args:
            query: 用户查询
            collect_name:知识库集合名字
            filter_dict: 过滤条件
        Returns:
            相关文档列表
        """
        vector_docs = self.retrieve_vector(query,collection_name,filter_dict)
        bm25_docs = self.retrieve_bm25(query)
        return self.mergeResults(vector_docs,bm25_docs)
    
    # ====== 合并结果 ======

    def mergeResults(
        self,
        *doc_lists:List[Document],
    ):
        """
        合并多路检索结果并去重
        策略：
        1. 按分数/顺序加权排序
        2. 去重（基于内容或ID）
        3. 返回合并列表
        """
        seen_ids: Set[str] = set()
        merged: List[Document] = []

        max_len = max(len(docs) for docs in doc_lists) if doc_lists else 0

        for i in range(max_len):
            for docs in doc_lists:
                if i < len(docs):
                    doc = docs[i]
                    doc_id = doc.metadata.get("id") or hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append(doc)
        return merged
    
    def rerank_results(
        self,
        query:str,
        documents:List[Document],
    )->List[Document]:
        """
        Rerank精排
        Args:
            query: 原始查询
            documents: 候选文档
        Returns:
            重排序后的Top-K文档
        """
        return self.rerank.rerank(
            query=query,
            documents=documents,
            top_k=self.rerank_top_k
        )

    def rerank_with_scores(
        self,
        query:str,
        documents:List[Document],
        score_threshold:float,
    )->List[tuple[Document,float]]:
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
        query:str,
    )->List[Document]:
        """快速检索（默认配置）"""
        return self.retrieve(query=query,strategy="hybrid")

    def get_status(
        self,
    )->Dict:
        """获取各组件状态"""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "bm25": self.bm25.get_stats(),
        }
    
# ====== 全局实例 ======
 
retriever_service = RetrieverService()