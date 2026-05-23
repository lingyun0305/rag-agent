# backend/utils/models_factory.py
from langchain_ import ChatDashScope
from langchain_community.embeddings import HuggingFaceEmbeddings,DashScopeEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from backend.app.core.config import (
    LLM_API_KEY,
    LLM_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL
)

# ==========================
# 1. 聊天LLM模型（HyDE、对话、改写）
# ==========================
def get_LLM_model(model_type: str = "qwen"):
    """
    model_type 可选：
    - qwen / qwen_turbo / qwen_plus
    - gpt / gpt35 / gpt4o
    - doubao
    - ollama
    """
    # 通义千问系列
    if model_type in ["qwen", "qwen_turbo"]:
        return ChatDashScope(
            api_key= LLM_API_KEY,
            model= model_type,
            
        )
        


# ==========================
# 2. 向量Embedding模型
# ==========================
def get_embedding_model(model_type: str = "qwen"):
    if model_type == "qwen":
        return DashScopeEmbeddings(
            api_key=EMBEDDING_API_KEY,
            model=EMBEDDING_MODEL
        )

    elif model_type == "local":
        # 本地BGE模型
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )

    raise ValueError("不支持的Embedding模型")


# ==========================
# 3. Rerank 重排序模型
# ==========================
def get_rerank_model(model_type: str = "bge"):
    if model_type == "bge":
        return HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base"
        )

    raise ValueError("不支持的Rerank模型")
