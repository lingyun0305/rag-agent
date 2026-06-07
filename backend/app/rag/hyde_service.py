# backend/app/utils/hyde_service.py
from typing import Optional,List,Dict
from langchain_core.documents import Document

from backend.app.utils.models_factory import modelFactory
from backend.app.utils.vector_store import get_vector_store

class HyDEService:
    """
    HyDE（Hypothetical Document Embeddings）检索服务
    
    原理：先用LLM根据用户问题生成一篇假设性文档，
    再用这篇假设文档去做向量检索，提升检索准确率。
    """
    # ====== 初始化 ======
    def __init__(
        self,
        collection_name:str,
        llm_provider:str = "qwen",
        temperature:float = 0.7
    ):
        self.llm = modelFactory.get_llm_model(provider=llm_provider,temperature=temperature)
        self.vector_store = get_vector_store(collection_name=collection_name)

    # ====== 提示词构建 ======

    def build_hyde_prompt(self,query:str)->str:
        """
        构建让LLM生成假设文档的提示词
        Args:
            query: 用户原始问题 
        Returns:
            构建好的prompt字符串
        """
        prompt = f"""请针对以下问题，写一段详细、专业的回答文档。
要求：
1. 以专业文档的风格撰写，而非对话风格
2. 包含具体的事实、数据、术语
3. 内容要详实，不少于200字
4. 不要说"假设"、"可能"等不确定的词语，直接以确定性的语气撰写

问题：{query}

请直接输出文档内容："""
        return prompt

    # ====== 假设性文档生成 ======

    def generate_hypothetical_doc(self,query:str)->str:
        """
        调用LLM生成假设性文档
        Args:
            query: 用户原始问题
        Returns:
            LLM生成的假设文档内容
        """
        prompt = self.build_hyde_prompt(query=query)
        response = self.llm.invoke(prompt)
        return response.content

    # ====== HyDE检索 ======

    def retrieve_with_hyde(
        self,
        query:str,
        top_k:int = 6,
        filter_dict:Optional[Dict[str,any]] = None,
        collection_name: Optional[str] = None,
    )->List[Document]:
        """
        使用HyDE方式检索相关文档
        流程：用户问题 → LLM生成假设文档 → 用假设文档检索 → 返回结果
        Args:
            query: 用户原始问题
            top_k: 返回文档数量
            filter_dict: 元数据过滤条件
            collection_name: 知识库名称
        Returns:
            相关文档列表
        """    
        hypothetical_doc = self.generate_hypothetical_doc(query)
        vector_store = get_vector_store(collection_name=collection_name)
        results = vector_store.search(
            query=hypothetical_doc,
            top_k=top_k,
            filter_dict=filter_dict
        )
        return results

    def retrieve_with_hyde_and_score(
        self,
        query:str,
        top_k:int = 6,
        score_threshold:float=0.7,
        filter_dict:Optional[Dict[str,any]] = None,
        collection_name: Optional[str] = None
    )->List[tuple[Document,float]]:
        """
        HyDE检索（带相似度分数）
        Args:
            query: 用户原始问题
            top_k: 返回文档数量
            score_threshold: 相似度阈值
            filter_dict: 元数据过滤条件
            collection_name: 知识库名称
        Returns:
            (文档, 分数) 元组列表
        """
        hypothetical_doc = self.generate_hypothetical_doc(query)
        vector_store = get_vector_store(collection_name=collection_name)
        results = self.vector_store.search_with_score(
            query=hypothetical_doc,
            top_k=top_k,
            filter_dict=filter_dict,
            score_threshold= score_threshold
        )
        return results

# ====== 全局实例 ======
hyde_service = HyDEService()