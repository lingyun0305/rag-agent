# backend/app/utils/rerank_service.py
from langchain_core.documents import Document
from typing import List,Optional

from backend.app.utils.models_factory import modelFactory

class RerankService:
    """
    Rerank重排序服务
    作用：对向量/BM25检索召回的候选文档进行精排，提高最终相关性
    原理：Cross-Encoder同时编码query和doc，计算精确匹配分数
    """

    # ====== 初始化 ======

    def __init__(self,provider:str="bge",top_k:int=6):
        """
        初始化Rerank服务
        Args:
            provider: Rerank模型提供商，默认"bge"
            top_k: 默认返回文档数
        """
        self.reranker = modelFactory.get_rerank_model(provider=provider)
        self.default_top_k = top_k

    # ====== 核心重排序方法 ======
    
    def rerank(
        self,
        query:str,
        documents:List[Document],
        top_k:Optional[int]=None
    )->List[Document]:
        """
        对候选文档进行重排序
        Args:
            query: 用户查询
            documents: 候选文档列表（来自向量/BM25检索）
            top_k: 返回Top-K条，默认self.default_top_k
        Returns:
            按相关性分数排序的文档列表
        """
        if not documents:
            return []
        if top_k is None:
            top_k = self.default_top_k
        
        if len(documents) > 100:
            documents = documents[:100]

        reranked_docs = self.reranker.compress_documents(
            documents=documents,
            query = query
        )
        return reranked_docs[:top_k]

    def rerank_with_score(
        self,
        query:str,
        documents:List[Document],
        top_k:Optional[int]=None,
        score_threshold:float = 0.7
    )->List[tuple[Document,float]]:
        """
        重排序并返回分数
        Args:
            query: 用户查询
            documents: 候选文档列表
            top_k: 返回数量
            score_threshold: 分数阈值，过滤低相关性结果
        Returns:
            (文档, 分数) 元组列表
        """
        if not documents:
            return []
        if top_k is None:
            top_k = self.default_top_k
        if len(documents) > 100:
            documents = documents[:100]
        reranked_docs = self.reranker.compress_documents(
            documents=documents,
            query = query,
        )
        results=[]
        for doc in reranked_docs:
            score = doc.metadata.get("relevance_score",0.0)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append((doc,score))
        return results
    
    # ====== 批量处理 ======

    def rerank_batch(
        self,
        queries:List[str],
        document_list:List[List[Document]],
        top_k:Optional[int] = None
    )->List[List[Document]]:
        """
        批量重排序（多个query分别处理）
        Args:
            queries: 查询列表
            documents_list: 每个query对应的候选文档列表
            top_k: 每个query返回数量
        Returns:
            重排序结果列表，与queries一一对应
        """
        results = []
        for query,docs in zip(queries,document_list):
            reranked_docs = self.rerank(query=query,documents=docs,top_k=top_k)
            results.append(reranked_docs)
        return results

# ====== 全局实例 ======

rerank_service = RerankService()