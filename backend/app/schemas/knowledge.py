# backend/app/schemas/knowledge.py
from pydantic import BaseModel,Field
from typing import Optional, List

# 创建知识库请求
class CreateKBReq(BaseModel):
    name: str = Field(..., description="知识库名称")
    desc: Optional[str] = Field("", description="知识库描述")
    folder_id: Optional[str] = Field(None, description="所在文件夹ID")

# 更新知识库请求
class UpdateKBReq(BaseModel):
    name: Optional[str] = Field(None, description="知识库名称")
    desc: Optional[str] = Field(None, description="知识库描述")

# 批量删除文档请求
class BatchDeleteDocsReq(BaseModel):
    doc_ids: List[str] = Field(..., description="要删除的文档ID列表")

# 创建文件夹请求
class CreateFolderReq(BaseModel):
    name: str = Field(..., description="文件夹名称")
    parent_id: Optional[str] = Field(None, description="父文件夹ID")

class RenameReq(BaseModel):
    name: str = Field(..., description="新名称", min_length=1, max_length=100)

# 文件上传请求模型
class FileUploadReq(BaseModel):
    kb_id: str = Field(..., description="知识库ID")

# RAG 问答请求
class RAGQueryReq(BaseModel):
    kb_id: str
    query: str = Field(..., description="用户提问")

# 分片内容结构
class ChunkContentSchema(BaseModel):
    doc_id: str = Field(..., description="来源文档id")
    content: str
    doc_name: str = Field(..., description="来源文档名")

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


# ====== 四步上传流程 ======

class PreviewReq(BaseModel):
    """第三步：数据预览请求"""
    batch_id: str = Field(..., description="上传批次ID")
    chunk_size: int = Field(500, description="分块大小")
    chunk_overlap: int = Field(50, description="重叠大小")
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "，", " "],
        description="分隔符列表（按优先级）"
    )
    top_k: int = Field(6, description="Rerank返回数量")
    score_threshold: float = Field(0.0, description="相似度阈值")
    embedding_model: str = Field("bge", description="嵌入模型名称")


class AdjustPreviewReq(BaseModel):
    """调整参数-数据预览请求"""
    doc_id: str = Field(..., description="文档ID")
    chunk_size: int = Field(500, description="分块大小")
    chunk_overlap: int = Field(50, description="重叠大小")
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "，", " "],
        description="分隔符列表（按优先级）"
    )
    top_k: int = Field(6, description="Rerank返回数量")
    score_threshold: float = Field(0.0, description="相似度阈值")
    embedding_model: str = Field("bge", description="嵌入模型名称")


class ConfirmUploadReq(BaseModel):
    """第四步：确认上传请求"""
    batch_id: str = Field(..., description="上传批次ID")
    kb_id: str = Field(..., description="目标知识库ID")
    cache_key: str = Field(..., description="预览缓存key")