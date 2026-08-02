# backend/app/utils/models_factory.py
from typing import Dict,List
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from backend.app.core.config import settings
from backend.app.common.exceptions import BusinessException
from backend.app.core.cache import cache_service

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
            },
            "zhipu":{
                "model":settings.ZHIPU_MODEL,
                "api_key":settings.ZHIPU_API_KEY,
                "base_url":settings.ZHIPU_BASE_URL
            },
            "local":{
                "model":"qwen2.5-7b",
                "api_key":"EMPTY",
                "base_url":"http://localhost:8080/v1"
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

    _llm_instances:Dict[str,ChatOpenAI] = {}
    _embedding_instances:Dict[str,any] = {}
    _rerank_instances:Dict[str,any] = {}
    _use_redis_cache: bool = False

    # ====== 模型工厂 ======
    
    def get_llm_model(
            self,
            provider:str = "qwen",
            temperature:float = 0.7,
            **kwargs
    )->any:
        """
        获取大语言模型实例
        Args:
            provider: 模型提供商 "qwen" / "doubao" / "zhipu" / "local"
            temperature: 创造性程度,0-1之间
        Returns:
            ChatOpenAI实例(兼容OpenAI格式)或本地模型实例
        """
        cache_key = f"{provider}_{temperature}"
        from backend.app.common.logger import logger
        logger.info(f"获取LLM实例: provider={provider}, temperature={temperature}, cache_key={cache_key}")
        if cache_key not in self._llm_instances:
            config = self.MODEL_CONFIGS["llm"][provider]
            
            params = {
                "model":config["model"],
                "base_url":config["base_url"],
                "api_key":config["api_key"],
                "temperature":temperature
            }
            params.update(kwargs)
            if provider in ("doubao", "zhipu", "qwen"):
                params["model_kwargs"] = {"stream_options": {"include_usage": True}}
            self._llm_instances[cache_key] = ChatOpenAI(**params)
            logger.info(f"创建LLM实例: provider={provider}, model={config['model']}, base_url={config['base_url']}")
        else:
            logger.info(f"使用缓存LLM实例: provider={provider}, temperature={temperature}")

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
        from backend.app.common.logger import logger
        if provider not in self._embedding_instances:
            config = self.MODEL_CONFIGS["embedding"][provider]
            if provider == "openai":
                self._embedding_instances[provider] = OpenAIEmbeddings(
                    model=config["model"],
                    api_key=config["api_key"]
                )
            else:
                model_name = config["model_name"]
                import os
                local_model_path = os.path.join(settings.BASE_DIR, "models", model_name.split("/")[-1])
                if os.path.exists(local_model_path):
                    logger.info(f"从本地加载Embedding模型: {local_model_path}")
                    model_name = local_model_path
                else:
                    logger.info(f"从HuggingFace下载Embedding模型: {model_name} (镜像: {settings.HF_ENDPOINT})")
                try:
                    self._embedding_instances[provider] = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=config["model_kwargs"],
                        encode_kwargs=config["encode_kwargs"]
                    )
                    logger.info(f"Embedding模型加载成功: {model_name}")
                except Exception as e:
                    logger.error(f"Embedding模型加载失败: {e}")
                    raise

        return self._embedding_instances[provider]
    

    def get_rerank_model(self,provider:str = "bge")->CrossEncoderReranker:
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

    def get_available_embeddings(self) -> List[Dict[str, str]]:
        """
        获取可用的Embedding模型列表（动态判断配置可用性）
        返回: [{provider, display_name, model_name}]
        """
        from backend.app.common.logger import logger
        available = []
        for provider, cfg in self.MODEL_CONFIGS["embedding"].items():
            try:
                if provider == "openai":
                    # 只有配置了 API key 才算可用
                    api_key = cfg.get("api_key", "") or ""
                    if not api_key or api_key.strip() == "":
                        logger.info(f"[embedding] openai 未配置 API key，跳过")
                        continue
                    model_name = cfg.get("model", "")
                    available.append({
                        "provider": provider,
                        "display_name": f"OpenAI - {model_name}",
                        "model_name": model_name,
                    })
                else:  # bge / huggingface 本地模型
                    model_name = cfg.get("model_name", "")
                    if not model_name:
                        continue
                    available.append({
                        "provider": provider,
                        "display_name": model_name.split("/")[-1] if "/" in model_name else model_name,
                        "model_name": model_name,
                    })
            except Exception as e:
                logger.warning(f"[embedding] 检查 {provider} 可用性时出错: {e}")
                continue
        return available

    def get_default_embedding_provider(self) -> str:
        """获取默认的Embedding provider（优先用bge，否则openai）"""
        avail = self.get_available_embeddings()
        if not avail:
            # 兜底
            return "bge"
        # 优先返回 bge，否则第一个
        for item in avail:
            if item["provider"] == "bge":
                return "bge"
        return avail[0]["provider"]

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

    async def clear_redis_cache(self, pattern: str = "*"):
        """清理Redis缓存"""
        await cache_service.clear_pattern(f"model:{pattern}")

    def enable_redis_cache(self, enable: bool = True):
        """启用/禁用Redis缓存"""
        self._use_redis_cache = enable

# ====== 全局实例 ======

model_factory = modelFactory()