# backend/app/utils/vector_store.py
from typing import Optional,List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import Embeddings
import os

from backend.app.core.config import settings
from backend.app.utils.models_factory import modelFactory

class VectorStoreUtil:
    """
    向量数据库工具类
    支持Chroma向量库的文档增删改查操作
    """
    # ====== 初始化 ======
    def __init__(
        self,
        collection_name:str,
        persist_directory:Optional[str] = None,
        embedding_provider:str = "bge"
    ):
        """
        初始化向量存储工具
        Args:
            collection_name: 集合名称
            persist_directory: 数据持久化目录
            embedding_provider: Embedding模型提供商
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.VECTOR_STORE_PATH or "./chroma_db"
        self.embedding_provider = embedding_provider

        self._client: Optional[Chroma] = None
        self._embedding_function: Optional[Embeddings] = None

    # 初始化向量数据库
    @property
    def embedding_function(self)->Embeddings:
        """懒加载embedding模型"""
        if self._embedding_function is None:
            self._embedding_function = modelFactory.get_embedding_model(self.embedding_provider)
        return self._embedding_function
    
    @property
    def get_client(self)->Chroma:
        """获取或初始化Chroma客户端"""
        if self._client is None:
            self._init_vector_store()
        return self._client

    def _init_vector_store(self):
        """初始化向量数据库"""
        os.makedirs(self.persist_directory, exist_ok=True)
        self._client= Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
    
    # ====== 写入操作 ======

    def add_documents(
        self,
        docs:List[Document],
        ids:Optional[List[str]] = None
    )->List[str]:
        """
        添加文档到向量库
        Args:
            docs: 文档列表
            ids: 自定义文档ID（可选）
        Returns:
            文档ID列表
        """
        if docs is None:
            return []
        else:
            return self.get_client.add_documents(documents=docs,ids=ids)

    def add_texts(
        self,
        texts:List[str],
        metadatas:Optional[List[str]] = None,
        ids:Optional[List[str]] = None
    )->List[str]:
        """
        直接添加文本到向量库
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 自定义ID
        Returns:
            文档ID列表
        """    
        if texts is None:
            return []
        else:
            return self.get_client.add_texts(texts=texts,metadatas=metadatas,ids=ids)
        
    def update_document():
        pass
    
    # ====== 检索操作 ======

    def search(self,query: str,top_k: int = 6)->list[Document]:
        pass
    
    def search_with_score():
        pass

    def mmr_search():
        pass

    # ====== 删除操作 ======

    def delete_by_ids(self,ids:list[str] ):
        pass

    def delete_by_filter():
        pass

    # ====== 管理操作 ======


# 实例化全局向量库
vector_store_util = VectorStoreUtil()