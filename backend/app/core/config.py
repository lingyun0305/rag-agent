from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 向量数据库基础配置
    VECTOR_STORE_TYPE:str = "chroma" 
    CHROMA_PERSIST_DIR:str = "./chroma_db" # 数据持久化目录
    CHROMA_COLLECTION_NAME:Optional[str]=None #集合名称
    VECTOR_STORE_URI:Optional[str]=None
    VECTOR_STORE_PORT:Optional[int]=None

    # Embedding模型配置
    EMBEDDING_MODEL:str = "BAAI/bfe-m3"
    EMBEDDING_API_KEY:Optional[str]=None
    EMBEDDING_DEVICE:str="cpu"

    # 重排序模型
    RERANK_MODEL_NAME:str = "BAAI/bge-m3"

    # 通用检索配置
    ENABLE_HYDE:bool = True
    BASE_RECALL_TOK_K:int = 20

    # 加载RAG配置
    CHUNK_SIZE:int = 512
    CHUNK_OVERLAP:int = 64
    RETRIEVE_TOP_K:int = 20
    KNOWLEDGE_FILE_PATH:str = "./knowledge_base"

    # 加载LLM配置
    LLM_API_KEY:str
    LLM_BASE_URL:str
    LLM_MODEL:str = "qwen-turbo"

    # 加载数据库配置
    MYSQL_USER:str
    MYSQL_PASSWORD:str
    MYSQL_HOST:str = "localhost"     
    MYSQL_PORT:int = 3306
    MYSQL_DATABASE:str

    # 加载JWT 配置
    SECRET_KEY:str
    ALGORITHM:str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:int = 30

    class Config:
        env_file=".env"

settings = Settings()