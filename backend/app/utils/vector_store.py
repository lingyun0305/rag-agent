# backend/app/utils/vector_store.py
from typing import Optional,List,Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import Embeddings
import os

from backend.app.core.config import settings
from backend.app.utils.models_factory import modelFactory
from backend.app.common.exceptions import BusinessException
from backend.app.common.logger import logger

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
    
    # ====== 检索操作 ======

    def search(
        self,
        query: str,
        top_k: int = 6,
        filter_dict:Optional[Dict[str,any]] = None
    )->list[Document]:
        """
        相似度检索
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件，如 {"source": "file1.pdf"}
        Returns:
            相关文档列表
        """
        return self.get_client.similarity_search(
            query=query,
            k=top_k,
            filter=filter_dict
        )
    
    def search_with_score(
        self,
        query:str,
        top_k:int = 6,
        filter_dict:Optional[Dict[str,any]]=None,
        score_threshold:Optional[float]=None
    )->List[tuple[Document,float]]:
        """
        相似度检索（带分数）
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
            score_threshold: 分数阈值，过滤低相关性结果
        Returns:
            (文档, 相似度分数) 元组列表
        """
        results = self.get_client.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        if score_threshold is not None:
            results=[(doc,score) for doc,score in results if score >= score_threshold]
        return results

    def mmr_search(
        self,
        query:str,
        top_k:int = 6,
        fetch_k:int = 20,
        lambda_mult:float = 0.5 
    )->List[Document]:
        """
        MMR多样性检索
        Args:
            query: 查询文本
            top_k: 返回结果数量
            fetch_k: 候选池大小
            lambda_mult: 多样性权重，0-1之间，越大多样性越高
        Returns:
            相关文档列表
        """
        return self.get_client.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

    # ====== 删除操作 ======

    def delete_by_ids(self,ids:list[str])->bool:
        """
        根据ID删除文档
        Args:
            ids: 文档ID列表
        Returns:
            是否成功
        """
        if not ids or self._client is None:
            return
        try:
            self._client.delete(ids=ids)
        except Exception as e:
            logger.error("向量数据库删除文档失败：{e}")
            raise BusinessException(
                code=500,
                msg="向量库执行删除操作失败"
            )

    def delete_by_filter(self,filter_dict:Dict[str,any])->bool:
        """
        根据条件删除文档
        Args:
            filter_dict: 过滤条件，如 {"source": "file1.pdf"}
        Returns:
            是否成功
        """
        if self._client is None:
            return False
        try:
            self._client.delete(where=filter_dict)
            return True
        except Exception as e:
            logger.error("向量库按条件删除文档失败：{e}")
            raise BusinessException(
                code=500,
                msg="向量库执行按条件删除操作失败"
            )


    # ====== 更新操作 ======

    def update_document(self,doc_id:str,new_doc:Document)->bool:
        """
        更新指定文档（先删除再添加）
        Args:
            doc_id: 文档ID
            new_doc: 新文档
        Returns:
            是否成功
        """
        try:
            self.delete_by_ids([doc_id])
            self.add_documents(new_doc,doc_id)
            return True
        except Exception as e:
            logger.error("更新文档失败:{e}")
            raise BusinessException(
                code=500,
                msg="向量库执行更新文档失败"
            )

    # ====== 管理操作 ======

    def get_collection_status(self)->Dict[str,any]:
        """
        获取集合统计信息
        Returns:
            统计信息字典
        """
        if self._client is None:
            return {"count":0}
        try:
            count = self._client._collection.count()
            logger.info(f"向量库状态：集合={self.collection_name}, 文档数={count}, 存储路径={self.persist_directory}")
            return {
                "collection_name":self.collection_name,
                "document_count":count,
                "persist_directory":self.persist_directory
            }
        except Exception as e:
            logger.error("获取向量库统计信息失败：{e}")
            raise BusinessException(
                code=500,
                msg="获取向量库统计信息操作失败"
            )
         
    def list_collections(self)->List[str]:
        """列出所有集合名称"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error("向量库获取集合列表失败")
            raise BusinessException(
                code=500,
                msg="向量库获取集合列表操作失败"
            )
        
    def clear_collection(self)->bool:
        """清空当前集合中的所有文档"""
        try:
            self.get_client.delete_collection()
            self._client = None
            logger.info("向量集合已清空，客户端实例已重置")
            return True
        except Exception as e:
            logger.error("清空集合执行异常")
            raise BusinessException(
                code=500,
                msg="清空向量知识库失败"
            )
    
    # ====== 获取向量存储实例 ======
def get_vector_store(
    collection_name:str = "default_collection",
    embedding_provider:str = "bge"
)->VectorStoreUtil:
    """
    获取向量存储实例
    Args:
        collection_name: 集合名称
        embedding_provider: Embedding提供商
    Returns:
        VectorStoreUtil实例
    """
    return VectorStoreUtil(
        collection_name=collection_name,
        embedding_provider=embedding_provider
    )

# 实例化全局向量库
vector_store_util = VectorStoreUtil()