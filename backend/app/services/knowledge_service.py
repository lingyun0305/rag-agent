# backend/app/services/knowledge_service.py
import os,time,uuid
from typing import List,Optional,Dict,Any
from langchain_core.documents import Document

from backend.app.utils.vector_store import get_vector_store
from backend.app.rag.bm25_service import bm25_service
from backend.app.utils.text_splitter import text_split_util

class KnowledgeService:
    """
    知识库管理服务
    
    负责文档的离线处理流程：上传 → 解析 → 切分 → 向量化 → 入库
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        upload_dir: str = "./uploads",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir,exist_ok=True)
    
    # ====== 文档解析 ======

    def parse_document(
        self,
        file_path: str,
    )->List[Document]:
        """
        解析文档，提取文本内容
        Args:
            file_path: 文件路径
        Returns:
            文档列表
        """
        docs = text_split_util.load_file_text(file_path)
        filename = os.path.basename(file_path)
        ext = text_split_util.get_ext(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = ext
            doc.metadata["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return docs
        
    # ====== 文本切分 ======

    def split_document(self,docs: List[Document])->List[Document]:
        """切分文档"""
        if not docs:
            return []
        chunks = []
        for doc in docs:
            split_texts = text_split_util.get_split_text(doc.page_content)
            for i,text in enumerate(split_texts):
                chunk = Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_id":str(uuid.uuid4()),
                        "chunk_index": i,
                    }
                )
                chunks.append(chunk)
        return chunks
    
    # ====== 入库 ======

    def index_document(
        self,
        chunks:List[Document],
        collection_name:str = "default_collection",
    )->Dict[str,Any]:
        """存入向量库和BM25索引"""
        if not chunks:
            return {
                "count":0,
                "collection":collection_name,
            }
        vectore_store = get_vector_store(collection_name = collection_name)
        vector_ids = vectore_store.add_documents(chunks)
        bm25_service.add_documents(chunks)
        return {
            "count":len(chunks),
            "collection":collection_name,
            "vector_ids":vector_ids,
        }
    
    # ====== 完整上传流程 ======

    def upload_document(
        self,
        file_path: str,
        collection_name: str = "default_collection"
    )->Dict[str,Any]:
        """解析 → 切分 → 入库""" 
        filename = os.path.basename(file_path)
        docs = self.parse_document(file_path)
        chunks = self.split_document(docs)
        result = self.index_document(chunks,collection_name)
        
        return {
            "filename": filename,
            "original_pages": len(docs),
            "chunks": result["count"],
            "collection": collection_name,
        }
    
    # ====== 删除 ======

    def delete_document(
        self,
        source: str,
        collection_name: str = "default_collection",
    )->bool:
        """根据文件名删除文档"""
        vectore_store = get_vector_store(collection_name=collection_name)
        vectore_store.delete_by_filter({"source":source})
        bm25_service.delete_by_filter({"source":source})
        return True
    
    # ====== 查询 ======

    def get_knowledge_status(
        self,
        collection_name: str = "default_collection",
    )->Dict[str,Any]:
        vector_store = get_vector_store(collection_name)
        return {
            "collection_name": collection_name,
            "vector_store": vector_store.get_collection_status(),
            "bm25":bm25_service.get_status(),
        }
    
    def clear_collections(self,collection_name: str = "default_collection",)->bool:
        vectore_store = get_vector_store(collection_name)
        vectore_store.clear_collection()
        bm25_service.clear()
        return True

    def list_collections(self)->List[str]:
        return get_vector_store().list_collections()
    
# ====== 全局实例 ======

knowledge_service = KnowledgeService()
