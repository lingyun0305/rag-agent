# backend/utils/models_factory.py
from typing import Dict,Optional
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.app.core.config import settings

class modelFactory:
    # 模型统一配置
    MODEL_CONFIGS = {
        # LLM模型配置
        "llm":{
            "qwen":{
                "model":settings.DASHSCOPE_MODEL,
                "api_key":settings.DASHSCOPE_API_KEY,
                "base_url":settings.DASHSCOPE_BASE_URL
            },
            "doubao":{
                "model":settings.ARK_MODEL,
                "api_key":settings.ARK_API_KEY,
                "base_url":settings.ARK_BASE_URL
            }
        },
        # Embedding模型配置
        "embedding":{
            "openai":{
                "model":settings.OPENAI_EMBEDDING_MODEL,
                "api_key":settings.OPENAI_EMBEDDING_API_KEY
            },
            "bge":{
                "model_name":settings.HUGGINGFACEEMBEDDINGS_MODEL_NAME,
                "model_kwargs":{"device":settings.HUGGINGFACEEMBEDDINGS_DEVICE},
                "encode_kwargs":{"normalize_embeddings":True}
            }
        },
        # Rerank模型配置
        "rerank":{
            "bge":{
                "model_name":settings.RERANK_MODEL_NAME,
                "model_kwargs":{"device":settings.RERANK_MODEL_DEVICE}
            },
        }
    }

    # ====== 模型缓存 ======

    #之后要升级为Redis缓存（分布式共享缓存）
    _llm_instance:Dict[str,ChatOpenAI] = {}
    _embedding_instance:Dict[str,any] = {}
    _rerank_instance:Dict[str,any] = {}

    # ====== 模型工厂 ======
    
    def get_llm_model()->ChatOpenAI:
        pass

    def get_embedding_model()->any:
        pass

    def get_rerank_model()->any:
        pass