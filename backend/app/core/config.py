# backend/app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Settings(BaseSettings):
    BASE_DIR: str = BASE_DIR
    
    # 向量数据库基础配置
    VECTOR_STORE_TYPE:str = "chroma" 
    VECTOR_STORE_PATH:str = "./chroma_db" # 数据持久化目录
    CHROMA_COLLECTION_NAME:Optional[str]=None #集合名称
    VECTOR_STORE_URI:Optional[str]=None
    VECTOR_STORE_PORT:Optional[int]=None

    # 加载数据库配置
    MYSQL_USER:str
    MYSQL_PASSWORD:str
    MYSQL_HOST:str = "localhost"     
    MYSQL_PORT:int = 3306
    MYSQL_DATABASE:str

    # 加载Embedding模型配置
    # openaiEmbedding模型
    OPENAI_EMBEDDING_MODEL:str = "text-embedding-3-small"
    OPENAI_EMBEDDING_API_KEY:str = ""
    # bge模型
    HUGGINGFACEEMBEDDINGS_MODEL_NAME:str = "BAAI/bge-large-zh-v1.5"
    HUGGINGFACEEMBEDDINGS_DEVICE:str="cpu"

    # 重排序模型
    # bge模型
    RERANK_MODEL_NAME:str = "BAAI/bge-reranker-v2-m3"
    RERANK_MODEL_DEVICE:str="cpu"

    # 加载LLM配置
    DEFAULT_LLM_PROVIDER:str = "qwen"
    # 千问
    DASHSCOPE_API_KEY:str
    DASHSCOPE_BASE_URL:str
    DASHSCOPE_MODEL:str = "qwen-turbo"
    # 豆包
    ARK_API_KEY:str
    ARK_BASE_URL:str
    ARK_MODEL:str = "doubao-pro-32k"
    # 智谱
    ZHIPU_API_KEY:str
    ZHIPU_BASE_URL:str
    ZHIPU_MODEL:str="glm-4.7-flash"
    
    # 本地模型
    LOCAL_MODEL_BASE_URL:str = "http://localhost:8080/v1"

    # 通用检索配置
    ENABLE_HYDE:bool = True
    BASE_RECALL_TOK_K:int = 20

    # 加载RAG配置
    CHUNK_SIZE:int = 512
    CHUNK_OVERLAP:int = 64
    RETRIEVE_TOP_K:int = 20
    KNOWLEDGE_FILE_PATH:str = "./knowledge_base"
    FILE_STORAGE_PATH:str = "./knowledge_files"  # 原始文件存储目录（按用户隔离）
    UPLOAD_STAGING_DIR:str = "./uploads"  # 上传暂存目录（按用户+批次隔离）

    # 文件上传限制
    MAX_UPLOAD_SIZE_MB:int = 50  # 单个文件最大上传大小（MB）
    MAX_UPLOAD_FILES:int = 5      # 单次批量上传最大文件数
    MAX_STORAGE_PER_USER_MB:int = 500  # 单用户 uploads 目录最大占用（MB）

    # 加载JWT 配置
    SECRET_KEY:str
    ALGORITHM:str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:int = 30

    # HuggingFace 镜像地址
    HF_ENDPOINT: str = "https://hf-mirror.com"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: str = ""

    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file=".env"

settings = Settings()
settings.REDIS_URL = settings.redis_url