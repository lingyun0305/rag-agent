from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional,List
from backend.app.core.config import settings

class embeddingUtils:
    _embedding_model:Optional[HuggingFaceEmbeddings]=None 

    # embedding模型懒加载
    @classmethod
    def get_embedding(cls)->HuggingFaceEmbeddings:
        if cls._embedding_model is None:
            cls._embedding_model = HuggingFaceEmbeddings(
                model_name = settings.EMBEDDING_MODEL,
                model_kwargs = {"device":settings.EMBEDDING_DEVICE},
                encode_kwargs = {"normalize_embeddings":True}
            )
        return cls._embedding_model

    # 批量文档/分片转向量
    @staticmethod
    def get_doc_embeddings(texts:List[str])->List[List[float]]:
        model = embeddingUtils.get_embedding()
        return model.embed_documents(texts)
    
    # 用户Query转向量
    @staticmethod
    def get_query_embeddings(text:str)->List[float]:
        """查询文本向量化"""
        model = embeddingUtils.get_embedding()
        return model.embed_query(text)
