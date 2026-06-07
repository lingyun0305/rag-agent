# backend/app/schemas/knowledge.py
from pydantic import BaseModel,Field

# 文件上传请求模型
class FileUploadReq(BaseModel):
    kb_id:int = Field(...,description="知识库ID")

# RAG 问答请求
class RAGQueryReq(BaseModel):
    kb_id:int
    query:str = Field(...,description="用户提问")

# 分片内容结构
class ChunkContentSchema(BaseModel):
    doc_id:int = Field(...,description="来源文档id")
    content:str
    doc_name:str = Field(...,description="来源文档名")

class KnowledgeResponse(BaseModel):
    filename: str
    original_pages: int
    chunks: int
    collection: str

class DeleteRequest(BaseModel):
    sourse: str
    collection: str = "default_collection"

class StatsResponse(BaseModel):
    collection_name: str
    document_count: int
    index_ready: bool