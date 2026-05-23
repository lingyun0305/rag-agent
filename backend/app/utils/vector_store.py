from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.app.core.config import settings

class VectorStoreUtil:
    def __init__(self):
        self.store_type = settings.VECTOR_STORE_TYPE
        self.client : Optional[Chroma] = None

    # 初始化向量数据库
    def init_vector_store(self):

        self.client= Chroma(
            embedding_function= settings.EMBEDDING_MODEL,
            persist_directory="./chroma_db"
        )
    
    # 加入新文档向量
    def add_documents(self,docs:list[Document]):
        if self.client is None:
            self.init_vector_store()
        else:
            self.client.add_documents(docs)
    
    # 向量相似度检索
    def search(self,query: str,top_k: int = 6)->list[Document]:
        if self.client is None:
            return []
        else:
            return self.client.similarity_search(query,k= top_k)

    # 跟据id删除指定向量
    def delete_by_ids(self,ids:list[str] ):
        if self.client and hasattr(self.client,ids):
            self.client.delete(ids)

# 实例化全局向量库
vector_store_util = VectorStoreUtil()