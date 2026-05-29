# backend/app/utils/models_factory.py
from typing import Dict
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.app.core.config import settings
from backend.app.common.exceptions import BusinessException

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
    _llm_instances:Dict[str,ChatOpenAI] = {}
    _embedding_instances:Dict[str,any] = {}
    _rerank_instances:Dict[str,any] = {}

    # ====== 模型工厂 ======
    
    def get_llm_model(
            self,
            provider:str = "qwen",
            temperature:float = 0.7,
            **kwargs
    )->ChatOpenAI:
        """
        获取大语言模型实例
        Args:
            provider: 模型提供商 "qwen" 或 "doubao"
            temperature: 创造性程度,0-1之间
        
        Returns:
            ChatOpenAI 实例(兼容OpenAI格式)
        """
        cache_key = f"{provider}_{temperature}"
        if cache_key not in self._llm_instances:
            config = self.MODEL_CONFIGS["llm"][provider]
            params = {
                "model":config["model"],
                "base_url":config["base_url"],
                "api_key":config["api_key"],
                "temperature":temperature
            }
            params.update(kwargs)
            self._llm_instances[cache_key] = ChatOpenAI(**params)

        return self._llm_instances[cache_key]


    def get_embedding_model(self,provider:str = "bge")->any:
        """
        获取Embedding模型实例
        Embedding的作用：将文本转成向量，用于语义检索
        Args:
            provider: "openai"(付费API) 或 "bge"(本地免费)
        
        Returns:
            Embedding模型实例
        """
        if provider not in self._embedding_instances:
            config = self.MODEL_CONFIGS["embedding"][provider]
            if provider == "openai":
                self._embedding_instances[provider] = OpenAIEmbeddings(
                    model=config["model"],
                    api_key=config["api_key"]
                )
            else:
                self._embedding_instances[provider] = HuggingFaceEmbeddings(
                    model_name=config["model_name"],
                    model_kwargs=config["model_kwargs"],
                    encode_kwargs=config["encode_kwargs"]
                )

        return self._embedding_instances[provider]
    

    def get_rerank_model(self,provider:str = "bge")->any:
        """
        获取Rerank模型实例
        Rerank的作用：对检索结果进行精排，提高相关性
        Args:
            provider: 目前只支持 "bge"
        
        Returns:
            Rerank模型实例
        """
        if provider not in self._rerank_instances:
            try: # 延迟导入，用到才加载
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
            except ImportError:
                raise BusinessException(
                        code=500,
                        msg="请安装: pip install langchain-community sentence-transformers langchain-classic"
                    )
            
            config = self.MODEL_CONFIGS["rerank"][provider]
            model = HuggingFaceCrossEncoder(
                model_name=config["model_name"],
                model_kwargs=config["model_kwargs"]
            )
            self._rerank_instances[provider] = CrossEncoderReranker(model=model)

        return self._rerank_instances[provider]
    
    # ====== 便捷函数 ======

    def get_available_llms(self)->list[str]:
        """获取可用的LLM列表"""
        return list(self.MODEL_CONFIGS["llm"].keys())

    def switch_llm(self,provider: str, temperature: float = 0.7):
        """切换当前使用的LLM"""
        if provider not in self.MODEL_CONFIGS["llm"]:
            raise BusinessException(
                    code=400,
                    msg=f"不支持的模型: {provider}，可用: {self.get_available_llms()}"
                )
        return self.get_llm_model(provider,temperature)
    
    # ====== 清理函数 ======

    def clear_model_cache(self):
        """清理所有模型缓存"""
        self._llm_instances.clear()
        self._embedding_instances.clear()
        self._rerank_instances.clear()