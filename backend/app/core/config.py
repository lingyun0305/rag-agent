# backend/app/core/config.py
import os
from typing import Any

from pydantic_settings import BaseSettings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# HuggingFace 模型缓存已迁移至 D 盘（2026-10-15，C 盘原 .cache\huggingface 已备份为 _bak）
# setdefault：终端/系统已显式设置 HF_HOME 时不覆盖
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class Settings(BaseSettings):
    BASE_DIR: str = BASE_DIR

    # 向量数据库基础配置
    VECTOR_STORE_TYPE: str = "chroma"
    VECTOR_STORE_PATH: str = "./chroma_db"  # 数据持久化目录
    CHROMA_COLLECTION_NAME: str | None = None  # 集合名称
    VECTOR_STORE_URI: str | None = None
    VECTOR_STORE_PORT: int | None = None

    # 加载数据库配置
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str

    # 加载Embedding模型配置
    # openaiEmbedding模型
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_API_KEY: str = ""
    # bge模型
    HUGGINGFACEEMBEDDINGS_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    HUGGINGFACEEMBEDDINGS_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # 重排序模型
    # bge模型
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANK_MODEL_DEVICE: str = os.getenv("RERANK_DEVICE", "cpu")

    # 加载LLM配置
    DEFAULT_LLM_PROVIDER: str = "qwen"
    # 千问
    DASHSCOPE_API_KEY: str
    DASHSCOPE_BASE_URL: str
    DASHSCOPE_MODEL: str = "qwen-turbo"
    # 豆包
    ARK_API_KEY: str
    ARK_BASE_URL: str
    ARK_MODEL: str = "doubao-pro-32k"
    # 智谱
    ZHIPU_API_KEY: str
    ZHIPU_BASE_URL: str
    ZHIPU_MODEL: str = "glm-4.7-flash"

    # 本地模型
    LOCAL_MODEL_BASE_URL: str = "http://localhost:8080/v1"
    LOCAL_MODEL_NAME: str = "qwen2.5-7b"
    LOCAL_MODEL_API_KEY: str = "EMPTY"

    # 通用检索配置
    ENABLE_HYDE: bool = True
    BASE_RECALL_TOK_K: int = 20

    # 加载RAG配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RETRIEVE_TOP_K: int = 20
    # 文档原文目录：知识库原始文件（file_reader 白名单目录之一）
    FILE_STORAGE_PATH: str = "./knowledge_files"  # 原始文件存储目录（按用户隔离）
    UPLOAD_STAGING_DIR: str = os.path.join(
        BASE_DIR, "uploads"
    )  # 上传暂存目录 + 聊天附件目录（file_reader 白名单目录之一；基于项目根的绝对路径，避免依赖启动 CWD）

    # 文件上传限制
    MAX_UPLOAD_SIZE_MB: int = 50  # 单个文件最大上传大小（MB）
    MAX_UPLOAD_FILES: int = 5  # 单次批量上传最大文件数
    MAX_STORAGE_PER_USER_MB: int = 500  # 单用户 uploads 目录最大占用（MB）

    # 加载JWT 配置
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # HuggingFace 镜像地址
    HF_ENDPOINT: str = "https://hf-mirror.com"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0
    REDIS_URL: str = ""

    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ReAct Agent 上下文压缩阈值配置（按 llm_provider 索引）
    # 注意：此处「压缩」是长度兜底（fallback），不是常驻机制。
    # - trigger_threshold：轨迹 token 量超过该值才触发 _maybe_compress_context（长度兜底）。
    # - keep_full_steps：仅在「触发压缩」时保留的最近完整轮数（兜底用，非常驻每轮裁剪）。
    # 常驻机制是另一套：_build_system_message 每轮注入 AgentState.render_for_prompt() 要点表，
    # 与轨迹长度无关，二者方向相反、必须解耦（2026-10-11 修正：此前误将压缩改为按轮数常驻）。
    REACT_CONTEXT_COMPRESS_CONFIG: dict[str, dict[str, Any]] = {
        "qwen": {
            "max_context_tokens": 32768,
            "max_output_tokens": 2048,
            "compress_threshold_ratio": 0.75,
            "keep_full_steps": 2,
        },
        "doubao": {
            "max_context_tokens": 32768,
            "max_output_tokens": 2048,
            "compress_threshold_ratio": 0.75,
            "keep_full_steps": 2,
        },
    }

    def get_react_context_config(self, provider: str | None = None) -> dict[str, Any]:
        """
        获取 ReAct 上下文压缩配置。

        优先按具体模型名匹配；否则按 provider 匹配；再否则回退默认 provider。
        并返回计算后的触发阈值字段：
        - available_window = max_context_tokens - max_output_tokens
        - trigger_threshold = available_window × compress_threshold_ratio
        """
        provider = provider or self.DEFAULT_LLM_PROVIDER
        cfg = (
            self.REACT_CONTEXT_COMPRESS_CONFIG.get(provider)
            or self.REACT_CONTEXT_COMPRESS_CONFIG.get(self.DEFAULT_LLM_PROVIDER)
            or {
                "max_context_tokens": 32768,
                "max_output_tokens": 2048,
                "compress_threshold_ratio": 0.75,
                "keep_full_steps": 2,
            }
        )
        cfg = dict(cfg)
        available_window = cfg.get("max_context_tokens", 32768) - cfg.get("max_output_tokens", 2048)
        cfg["available_window"] = available_window
        cfg["trigger_threshold"] = int(available_window * cfg.get("compress_threshold_ratio", 0.75))
        return cfg

    class Config:
        env_file = ".env"


settings = Settings()
settings.REDIS_URL = settings.redis_url
