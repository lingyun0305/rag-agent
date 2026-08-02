# backend/app/utils/vector_store.py
from typing import Optional,List,Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import Embeddings
import os
import threading
import uuid

from backend.app.core.config import settings
from backend.app.utils.models_factory import model_factory
from backend.app.common.exceptions import BusinessException
from backend.app.common.logger import logger


class VectorStoreUtil:
    """
    向量数据库工具类
    支持Chroma向量库的文档增删改查操作
    """
    SUMMARY_COLLECTION_NAME = "_kb_summaries_"
    
    # 实例缓存，确保同一个 collection_name 返回同一个实例
    _instances: Dict[str, 'VectorStoreUtil'] = {}
    _lock = threading.Lock()
    
    # ====== 初始化 ======
    def __init__(
        self,
        collection_name:str = "default_collection",
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
        self._summary_client: Optional[Chroma] = None
        self._is_empty_cache: Optional[bool] = None
        self._empty_cache_time: float = 0

    # 初始化向量数据库
    @property
    def embedding_function(self)->Embeddings:
        """懒加载embedding模型"""
        if self._embedding_function is None:
            try:
                self._embedding_function = model_factory.get_embedding_model(self.embedding_provider)
            except Exception as e:
                logger.error(f"Embedding模型加载失败: {e}")
                raise BusinessException(f"Embedding模型加载失败: {e}")
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
        if not docs:
            return []
        if ids is None:
            # 生成唯一的 ID，避免不同文档的分片 ID 冲突
            # ID 格式：{source}_{chunk_index}_{uuid}
            ids = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'unknown') if doc.metadata else 'unknown'
                chunk_index = doc.metadata.get('chunk_index', i) if doc.metadata else i
                unique_id = f"{source}_{chunk_index}_{uuid.uuid4().hex[:8]}"
                ids.append(unique_id)
        logger.info(f"添加文档到向量库: count={len(docs)}, collection={self.collection_name}")
        return self.get_client.add_documents(documents=docs, ids=ids)

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

    def is_collection_empty(self, force_check: bool = False) -> bool:
        """
        检查集合是否为空（轻量级检查，不加载embedding模型）
        使用缓存，默认5分钟内复用结果
        Args:
            force_check: 是否强制重新检查
        Returns:
            True表示空，False表示有文档
        """
        import time
        now = time.time()
        
        if not force_check and self._is_empty_cache is not None:
            if now - self._empty_cache_time < 300:
                return self._is_empty_cache
        
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            try:
                collection = client.get_collection(self.collection_name)
                count = collection.count()
                result = count == 0
                self._is_empty_cache = result
                self._empty_cache_time = now
                return result
            except Exception:
                self._is_empty_cache = True
                self._empty_cache_time = now
                return True
        except Exception as e:
            logger.warning(f"检查集合是否为空失败，假设为空: {e}")
            self._is_empty_cache = True
            self._empty_cache_time = now
            return True

    # ====== 知识库概要管理 ======

    def _get_summary_client(self) -> Chroma:
        """获取概要向量库客户端（懒加载）"""
        if self._summary_client is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._summary_client = Chroma(
                collection_name=self.SUMMARY_COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        return self._summary_client

    def set_kb_summary(self, kb_name: str, summary: str, metadata: Optional[Dict] = None):
        """
        设置知识库概要
        Args:
            kb_name: 知识库名称
            summary: 知识库概要（1-2句话，描述知识库内容范围）
            metadata: 额外元数据（如创建时间、文档数量等）
        """
        client = self._get_summary_client()
        doc_metadata = metadata or {}
        doc_metadata["kb_name"] = kb_name
        
        existing = client.get(ids=[kb_name])
        if existing["ids"]:
            client.update(
                ids=[kb_name],
                documents=[summary],
                metadatas=[doc_metadata]
            )
        else:
            client.add(
                ids=[kb_name],
                documents=[summary],
                metadatas=[doc_metadata]
            )
        logger.info(f"知识库概要已更新: {kb_name}")

    def get_kb_summary(self, kb_name: str) -> Optional[str]:
        """
        获取知识库概要
        Args:
            kb_name: 知识库名称
        Returns:
            概要文本，不存在返回None
        """
        try:
            client = self._get_summary_client()
            result = client.get(ids=[kb_name])
            if result["ids"] and result["documents"]:
                return result["documents"][0]
            return None
        except Exception as e:
            logger.warning(f"获取知识库概要失败: {e}")
            return None

    def search_relevant_kbs(self, query: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        根据查询搜索相关的知识库
        Args:
            query: 用户查询
            top_k: 返回最多多少个相关知识库
            similarity_threshold: 相似度阈值（0-1），低于此值认为不相关
        Returns:
            [{"kb_name": str, "summary": str, "similarity": float}]
        """
        try:
            client = self._get_summary_client()
            results = client.similarity_search_with_score(query, k=top_k)
            
            relevant = []
            for doc, score in results:
                kb_name = doc.metadata.get("kb_name", "")
                summary = doc.page_content
                if score >= similarity_threshold:
                    relevant.append({
                        "kb_name": kb_name,
                        "summary": summary,
                        "similarity": score
                    })
            
            logger.info(f"查询 '{query}' 匹配到 {len(relevant)} 个知识库")
            return relevant
        except Exception as e:
            logger.warning(f"搜索相关知识库失败: {e}")
            return []

    def is_query_relevant_to_kb(self, query: str, kb_name: str = None) -> bool:
        """
        判断查询是否与知识库相关（轻量级检查）
        Args:
            query: 用户查询
            kb_name: 知识库名称（默认检查当前集合）
        Returns:
            True表示相关，False表示不相关
        """
        target_kb = kb_name or self.collection_name
        summaries = self.search_relevant_kbs(query, top_k=5, similarity_threshold=0.3)
        
        for item in summaries:
            if item["kb_name"] == target_kb:
                logger.info(f"查询 '{query}' 与知识库 '{target_kb}' 相关（相似度: {item['similarity']:.3f}）")
                return True
        
        logger.info(f"查询 '{query}' 与知识库 '{target_kb}' 不相关")
        return False
         
    def list_collections(self)->List[Dict]:
        """列出所有集合，返回包含名称、文档数、metadata等信息的列表"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()
            result = []
            for c in collections:
                try:
                    count = c.count()
                    metadata = c.metadata or {}
                except Exception:
                    count = 0
                    metadata = {}
                result.append({
                    "name": c.name,
                    "document_count": count,
                    "metadata": metadata
                })
            return result
        except Exception as e:
            logger.error(f"向量库获取集合列表失败: {e}")
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
    获取向量存储实例（使用缓存，确保同一个 collection_name 返回同一个实例）
    Args:
        collection_name: 集合名称
        embedding_provider: Embedding提供商
    Returns:
        VectorStoreUtil实例
    """
    # 使用缓存键，确保同一个 collection_name + embedding_provider 返回同一个实例
    cache_key = f"{collection_name}_{embedding_provider}"
    
    if cache_key not in VectorStoreUtil._instances:
        with VectorStoreUtil._lock:
            # 双重检查锁定
            if cache_key not in VectorStoreUtil._instances:
                VectorStoreUtil._instances[cache_key] = VectorStoreUtil(
                    collection_name=collection_name,
                    embedding_provider=embedding_provider
                )
                logger.debug(f"创建新的 VectorStoreUtil 实例: {cache_key}")
    
    return VectorStoreUtil._instances[cache_key]

# 实例化全局向量库
vector_store_util = VectorStoreUtil()