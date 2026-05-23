from pydantic import BaseModel

class RAGTextSplitConfig(BaseModel):
    chunk_size:int = 512
    chunk_overlap:int = 64
    separators:list[str] = ["\n\n","\n","。","，"," "]

class RAGHybridWeightConfig(BaseModel):
    bm25_weight:float = 0.5
    vector_weight:float = 0.5

class RAGSearchConfig(BaseModel):
    rerank_top_n:int = 5
    max_summary_doc_num:int = 3

# 全局RAG高级配置
rag_split_config = RAGTextSplitConfig()
rag_weight_config = RAGHybridWeightConfig()
rag_search_config = RAGSearchConfig()
