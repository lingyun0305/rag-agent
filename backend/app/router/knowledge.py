# backend/app/router/knowledge.py
from fastapi import APIRouter,Depends,UploadFile,File
from typing import Optional,List
from sqlalchemy.orm import Session
import os

from backend.app.services.knowledge_service import knowledge_service
from backend.app.core.security import get_current_user
from backend.db.db_config import get_session
from backend.app.schemas.knowledge import KnowledgeResponse,DeleteRequest,StatsResponse
from backend.app.common.exceptions import BusinessException

router = APIRouter()

@router.post(
    "/upload",
    response_model=KnowledgeResponse,
    summary="上传文档到知识库"
)
async def upload_document(
    file:UploadFile=File(...),
    collection: str = "default_collection",
    current_user: int = Depends(get_current_user), ###
):
    """
    上传文档并自动解析、切分、入库
    """
    # 保存上传文件到临时目录
    file_path = os.path.join(knowledge_service.upload_dir,file.filename)
    with open(file_path,"wb") as f:
        content = await file.read()
        f.write(content)
    try:
        result = knowledge_service.upload_document(
            file_path = file_path,
            collection_name=collection
        )
        return KnowledgeResponse(**result)
    except ValueError as e:
        raise BusinessException(
            code=400,
            msg=str(e)
        )
    finally:
        # 清理临时文件
        if os.path.exists(path=file_path):
            os.remove(file_path)

# ====== 删除文档 ======

@router.delete(
    "",
    summary="删除知识库文档",
)
async def delete_document(
    request: DeleteRequest,
    current_user: int = Depends(get_current_user)
):
    """
    根据文件名删除指定知识库中的文档
    """
    success = knowledge_service.delete_document(
        source=request.sourse,
        collection_name=request.collection,
    )
    return {"success": success} ###

# ====== 查询统计 ======

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="知识库统计",
)
async def get_stats(
    collection: str = "default_collection",
    current_user: int = Depends(get_current_user),
):
    """获取指定知识库的文档数量和索引状态"""
    stats = knowledge_service.get_knowledge_status(
        collection_name=collection
    )
    return StatsResponse(
        collection_name=stats["collection_name"],
        document_count=stats["vector_store"]["document_count"], ###
        index_ready=stats["bm25"]["index_ready"], ### 
    )

# ====== 知识库列表 ======

@router.get(
    "/collections",
    summary="知识库列表",
)
async def list_collections(
    current_user: int = Depends(get_current_user),
):
    """列出所有知识库集合"""
    return {"collections": knowledge_service.list_collections()}

# ====== 清空知识库 ======

@router.delete(
    "/collection/{collection_name}",
    summary="清空知识库",
)
async def clear_collection(
    collection_name: str,
    current_user: dict = Depends(get_current_user),
):
    """清空指定知识库的所有文档"""
    success = knowledge_service.clear_collection(collection_name)
    return {"success": success}