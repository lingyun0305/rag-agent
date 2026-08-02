# backend/app/router/knowledge.py
from fastapi import APIRouter, Depends, UploadFile, File, Form
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from pydantic import BaseModel
import os
from datetime import datetime

from backend.app.services.knowledge_service import knowledge_service
from backend.app.core.rag_config import rag_split_config
from backend.app.core.security import get_current_user
from backend.app.core.config import settings
from backend.db.db_config import get_session
from backend.app.schemas.knowledge import (
    KnowledgeResponse, DeleteRequest, StatsResponse,
    CreateKBReq, CreateFolderReq, UpdateKBReq, BatchDeleteDocsReq,
    PreviewReq, AdjustPreviewReq, ConfirmUploadReq, RenameReq,
)
from backend.app.common.exceptions import BusinessException
from backend.app.common.logger import logger
from backend.app.models.knowledge import KnowledgeBase, KbFolder, DocumentInfo, KbMember

router = APIRouter()


async def _check_kb_access(db: AsyncSession, kb_id: str, user_id: str) -> Optional[str]:
    """
    检查用户是否有权访问知识库
    返回: 'owner' / 'collaborator' / None(无权限)
    """
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.status == "active"
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        return None

    if kb.owner_id == user_id:
        return "owner"

    member_result = await db.execute(
        select(KbMember).where(
            KbMember.kb_id == kb_id,
            KbMember.member_type == "user",
            KbMember.member_id == user_id,
            KbMember.status == "active"
        )
    )
    member = member_result.scalar_one_or_none()
    if member:
        return "collaborator"

    return None

# ====== 知识库管理 ======

@router.post(
    "/create_kb",
    summary="创建知识库"
)
async def create_knowledge_base(
    req: CreateKBReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    创建知识库（同时创建向量 collection）
    """
    # 检查同名知识库
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.name == req.name,
            KnowledgeBase.owner_id == current_user,
            KnowledgeBase.status == "active"
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise BusinessException(code=400, msg="已存在同名知识库")

    # 创建数据库记录
    from backend.app.services.auth import generate_id
    kb = KnowledgeBase(
        id=generate_id(),
        name=req.name,
        desc=req.desc,
        owner_id=current_user,
        folder_id=req.folder_id,
        collaborator_count=1,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)

    # 创建者自动成为协作者(admin)
    owner_member = KbMember(
        id=generate_id(),
        kb_id=kb.id,
        member_type="user",
        member_id=current_user,
        access_level="admin",
        created_by=current_user,
    )
    db.add(owner_member)
    await db.commit()

    # 创建向量 collection（通过 get_vector_store 自动创建）
    from backend.app.utils.vector_store import get_vector_store
    collection_name = knowledge_service.get_kb_collection_name(kb.id)
    vs = get_vector_store(collection_name=collection_name)
    vs.get_collection_status()  # 触发初始化

    return {
        "id": kb.id,
        "name": kb.name,
        "desc": kb.desc,
        "folder_id": kb.folder_id,
        "collaborator_count": kb.collaborator_count,
        "collection_name": collection_name,
        "message": "知识库创建成功"
    }


@router.get(
    "/get_kbs",
    summary="获取我的知识库列表"
)
async def get_my_knowledge_bases(
    folder_id: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前用户的知识库列表（包含自己创建的 + 作为协作者的），支持按文件夹筛选
    """
    if folder_id is not None and folder_id.strip() == "":
        folder_id = None

    # 子查询：当前用户作为协作者的KB ID
    collab_subquery = select(KbMember.kb_id).where(
        KbMember.member_type == "user",
        KbMember.member_id == current_user,
        KbMember.status == "active"
    )

    # 查询：owner = current_user 或 在协作者列表中
    query = select(KnowledgeBase).where(
        KnowledgeBase.status == "active",
        or_(
            KnowledgeBase.owner_id == current_user,
            KnowledgeBase.id.in_(collab_subquery)
        )
    )
    if folder_id is not None:
        query = query.where(KnowledgeBase.folder_id == folder_id)
    else:
        query = query.where(KnowledgeBase.folder_id.is_(None))

    result = await db.execute(query.order_by(KnowledgeBase.create_time.desc()))
    kbs = result.scalars().all()

    # 获取协作者权限信息
    collab_map = {}
    collab_result = await db.execute(
        select(KbMember.kb_id, KbMember.access_level).where(
            KbMember.member_type == "user",
            KbMember.member_id == current_user,
            KbMember.status == "active"
        )
    )
    for row in collab_result.fetchall():
        collab_map[row[0]] = row[1]  # kb_id -> access_level

    result_list = []
    # 批量获取 owner 信息
    from backend.app.models.user import User
    all_owner_ids = list(set(kb.owner_id for kb in kbs))
    owner_name_map = {}
    if all_owner_ids:
        owner_result = await db.execute(
            select(User.id, User.username, User.nickname).where(User.id.in_(all_owner_ids))
        )
        for row in owner_result.fetchall():
            owner_name_map[row[0]] = row[2] or row[1]

    for kb in kbs:
        doc_result = await db.execute(
            select(func.count()).where(
                DocumentInfo.kb_id == kb.id,
                DocumentInfo.status == "completed"
            )
        )
        doc_count = doc_result.scalar()

        collection_name = knowledge_service.get_kb_collection_name(kb.id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        try:
            vs_status = vs.get_collection_status()
            vector_doc_count = vs_status.get("document_count", 0)
        except Exception as e:
            logger.debug(f"[kb_list] 向量库状态获取失败: {e}")
            vector_doc_count = 0

        is_owner = (kb.owner_id == current_user)
        access_level = "owner" if is_owner else collab_map.get(kb.id, "read")

        result_list.append({
            "id": kb.id,
            "name": kb.name,
            "desc": kb.desc,
            "folder_id": kb.folder_id,
            "collaborator_count": kb.collaborator_count or 1,
            "doc_count": doc_count,
            "vector_doc_count": vector_doc_count,
            "owner_id": kb.owner_id,
            "owner_name": owner_name_map.get(kb.owner_id, "未知用户"),
            "create_time": kb.create_time.strftime("%Y-%m-%d %H:%M:%S") if kb.create_time else None,
            "is_owner": is_owner,
            "access_level": access_level,
        })

    return {"kbs": result_list}


@router.delete(
    "/kb/{kb_id}",
    summary="删除知识库"
)
async def delete_knowledge_base(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除知识库（软删除，同时清空向量数据）
    权限：owner 或 admin 协作者
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")
    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level != "admin":
            raise BusinessException(code=403, msg="仅管理员协作者可删除知识库")

    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.status == "active"
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise BusinessException(code=404, msg="知识库不存在")

    # 清空向量数据
    try:
        knowledge_service.clear_knowledge_base(kb_id)
    except Exception as e:
        # 允许向量库不存在的情况
        pass

    kb.status = "deleted"
    await db.commit()

    return {"success": True, "message": "知识库已删除"}


# ====== 文件夹管理 ======

@router.post(
    "/create_folder",
    summary="创建文件夹"
)
async def create_folder(
    req: CreateFolderReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    创建文件夹
    """
    from backend.app.services.auth import generate_id
    from backend.app.common.logger import logger

    folder = KbFolder(
        id=generate_id(),
        user_id=current_user,
        parent_id=req.parent_id,
        name=req.name,
        collaborator_count=1,
    )
    db.add(folder)
    await db.commit()
    await db.refresh(folder)

    # 创建者自动成为协作者(admin)
    try:
        owner_member = KbMember(
            id=generate_id(),
            kb_id=None,
            folder_id=folder.id,
            member_type="user",
            member_id=current_user,
            access_level="admin",
            created_by=current_user,
        )
        db.add(owner_member)
        await db.commit()
    except Exception as e:
        logger.warning(f"创建文件夹协作者记录失败: {e}")
        await db.rollback()

    return {
        "id": folder.id,
        "name": folder.name,
        "parent_id": folder.parent_id,
        "collaborator_count": folder.collaborator_count,
        "message": "文件夹创建成功"
    }


@router.get(
    "/get_folders",
    summary="获取我的文件夹列表"
)
async def get_my_folders(
    parent_id: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前用户的文件夹列表
    """
    # 空字符串转换为 None
    if parent_id is not None and parent_id.strip() == "":
        parent_id = None

    query = select(KbFolder).where(
        KbFolder.user_id == current_user,
        KbFolder.status == "active"
    )
    if parent_id is not None:
        query = query.where(KbFolder.parent_id == parent_id)
    else:
        query = query.where(KbFolder.parent_id.is_(None))

    result = await db.execute(query.order_by(KbFolder.sort_order.asc(), KbFolder.create_time.desc()))
    folders = result.scalars().all()

    # 批量获取创建者信息
    from backend.app.models.user import User
    owner_ids = list(set(f.user_id for f in folders))
    owner_map = {}
    if owner_ids:
        owner_result = await db.execute(
            select(User.id, User.username, User.nickname).where(User.id.in_(owner_ids))
        )
        for row in owner_result.fetchall():
            owner_map[row[0]] = row[2] or row[1]  # nickname 优先，否则 username

    return {
        "folders": [
            {
                "id": f.id,
                "name": f.name,
                "parent_id": f.parent_id,
                "collaborator_count": f.collaborator_count or 1,
                "owner_id": f.user_id,
                "owner_name": owner_map.get(f.user_id, "未知用户"),
                "create_time": f.create_time.strftime("%Y-%m-%d %H:%M:%S") if f.create_time else None,
            }
            for f in folders
        ]
    }


@router.delete(
    "/folder/{folder_id}",
    summary="删除文件夹"
)
async def delete_folder(
    folder_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除文件夹（要求文件夹内没有知识库）
    """
    result = await db.execute(
        select(KbFolder).where(
            KbFolder.id == folder_id,
            KbFolder.user_id == current_user
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise BusinessException(code=404, msg="文件夹不存在")

    # 检查是否有知识库
    kb_result = await db.execute(
        select(func.count()).where(
            KnowledgeBase.folder_id == folder_id,
            KnowledgeBase.status == "active"
        )
    )
    kb_count = kb_result.scalar()
    if kb_count > 0:
        raise BusinessException(code=400, msg="文件夹内还有知识库，请先移出或删除")

    folder.status = "deleted"
    await db.commit()

    return {"success": True, "message": "文件夹已删除"}


@router.put(
    "/folder/{folder_id}/rename",
    summary="重命名文件夹"
)
async def rename_folder(
    folder_id: str,
    req: RenameReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(KbFolder).where(
            KbFolder.id == folder_id,
            KbFolder.user_id == current_user
        )
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise BusinessException(code=404, msg="文件夹不存在")

    folder.name = req.name
    await db.commit()

    return {"success": True, "message": "重命名成功", "name": folder.name}


@router.post(
    "/kb/{kb_id}/move",
    summary="移动知识库到文件夹"
)
async def move_knowledge_base(
    kb_id: str,
    folder_id: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    移动知识库到指定文件夹（folder_id=None 表示移到根目录）
    """
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.owner_id == current_user
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise BusinessException(code=404, msg="知识库不存在")

    if folder_id is not None:
        folder_result = await db.execute(
            select(KbFolder).where(
                KbFolder.id == folder_id,
                KbFolder.user_id == current_user
            )
        )
        folder = folder_result.scalar_one_or_none()
        if not folder:
            raise BusinessException(code=404, msg="目标文件夹不存在")

    kb.folder_id = folder_id
    await db.commit()

    return {"success": True, "message": "移动成功"}


# ====== 文档上传 ======

@router.post(
    "/upload",
    summary="上传文档到知识库"
)
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    上传文档并自动解析、切分、入库
    """
    # 验证权限（owner 或 write/admin 协作者）
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无权限")
    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level == "read":
            raise BusinessException(code=403, msg="只读协作者无权上传文档")

    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.status == "active"
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise BusinessException(code=404, msg="知识库不存在")

    # 保存到用户隔离目录
    file_path = knowledge_service.get_file_storage_path(
        user_id=current_user,
        folder_id=kb.folder_id,
        filename=file.filename
    )
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = knowledge_service.upload_document(
            file_path=file_path,
            kb_id=kb_id,
            user_id=current_user,
        )

        # 记录文档信息到数据库
        from backend.app.services.auth import generate_id
        doc_info = DocumentInfo(
            id=generate_id(),
            kb_id=kb_id,
            file_name=file.filename,
            file_type=os.path.splitext(file.filename)[1].lower().lstrip("."),
            file_size=os.path.getsize(file_path),
            file_path=file_path,
            chunk_count=result.get("chunks", 0),
            status="completed",
        )
        db.add(doc_info)
        await db.commit()

        return {
            "success": True,
            "filename": result.get("filename", file.filename),
            "chunks": result.get("chunks", 0),
            "message": "上传成功"
        }
    except ValueError as e:
        raise BusinessException(code=400, msg=str(e))
    except Exception as e:
        raise BusinessException(code=500, msg=f"上传处理失败: {str(e)}")


# ====== 删除文档 ======

@router.delete(
    "/document",
    summary="删除知识库文档"
)
async def delete_document(
    request: DeleteRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    根据文件名删除指定知识库中的文档
    """
    success = knowledge_service.delete_document(
        source=request.sourse,
        kb_id=request.collection.replace("kb_", "") if request.collection.startswith("kb_") else request.collection,
    )
    return {"success": success}


# ====== 查询统计 ======

@router.get(
    "/stats/{kb_id}",
    response_model=StatsResponse,
    summary="知识库统计"
)
async def get_stats(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """获取指定知识库的文档数量和索引状态"""
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")

    stats = knowledge_service.get_knowledge_status(kb_id=kb_id)
    return StatsResponse(
        collection_name=stats["collection_name"],
        document_count=stats["vector_store"]["document_count"],
        index_ready=stats["bm25"]["index_ready"],
    )


# ====== 知识库列表（兼容旧接口） ======

@router.get(
    "/collections",
    summary="知识库列表（兼容）"
)
async def list_collections(
    current_user: str = Depends(get_current_user),
):
    """列出所有知识库集合（包含统计信息）"""
    return {"collections": knowledge_service.list_collections()}


# ====== 设置知识库概要 ======

@router.post(
    "/summary",
    summary="设置知识库概要"
)
async def set_kb_summary(
    kb_id: str = Form(...),
    summary: str = Form(""),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    设置知识库概要
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")

    if not summary.strip():
        raise BusinessException(code=400, msg="概要内容不能为空")

    from backend.app.utils.vector_store import vector_store_util
    collection_name = knowledge_service.get_kb_collection_name(kb_id)
    vector_store_util.set_kb_summary(kb_name=collection_name, summary=summary)
    return {"success": True, "message": "知识库概要已更新"}


# ====== 获取知识库概要 ======

@router.get(
    "/summary",
    summary="获取知识库概要"
)
async def get_kb_summary(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """获取指定知识库的概要信息"""
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")

    from backend.app.utils.vector_store import vector_store_util
    collection_name = knowledge_service.get_kb_collection_name(kb_id)
    summary = vector_store_util.get_kb_summary(kb_name=collection_name)
    return {"kb_id": kb_id, "summary": summary or "未设置"}


# ====== 清空知识库 ======

@router.delete(
    "/kb/{kb_id}/clear",
    summary="清空知识库内容"
)
async def clear_knowledge_base(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """清空指定知识库的所有文档"""
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")
    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level != "admin":
            raise BusinessException(code=403, msg="仅管理员协作者可清空知识库")

    success = knowledge_service.clear_knowledge_base(kb_id)
    return {"success": success}


# ====== 知识库详情页 ======

@router.get(
    "/kb/{kb_id}/get_detail",
    summary="获取知识库详情"
)
async def get_kb_detail(
    kb_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取指定知识库的详情信息（含权限校验）
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = result.scalar_one_or_none()

    doc_result = await db.execute(
        select(func.count()).where(
            DocumentInfo.kb_id == kb_id,
            DocumentInfo.status == "completed"
        )
    )
    doc_count = doc_result.scalar()

    collection_name = knowledge_service.get_kb_collection_name(kb_id)
    from backend.app.utils.vector_store import get_vector_store
    vs = get_vector_store(collection_name=collection_name)
    try:
        vs_status = vs.get_collection_status()
        vector_doc_count = vs_status.get("document_count", 0)
    except Exception as e:
        logger.debug(f"[kb_detail] 向量库状态获取失败: {e}")
        vector_doc_count = 0

    return {
        "id": kb.id,
        "name": kb.name,
        "desc": kb.desc,
        "folder_id": kb.folder_id,
        "collaborator_count": kb.collaborator_count or 1,
        "embedding_model": kb.embedding_model,
        "chunk_strategy": kb.chunk_strategy,
        "top_k": kb.top_k,
        "owner_id": kb.owner_id,
        "is_owner": access_level == "owner",
        "access_level": access_level,
        "doc_count": doc_count,
        "vector_doc_count": vector_doc_count,
        "create_time": kb.create_time.strftime("%Y-%m-%d %H:%M:%S") if kb.create_time else None,
        "update_time": kb.update_time.strftime("%Y-%m-%d %H:%M:%S") if kb.update_time else None,
    }


@router.put(
    "/kb/{kb_id}",
    summary="更新知识库信息"
)
async def update_knowledge_base(
    kb_id: str,
    req: UpdateKBReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    更新知识库的名称和描述
    权限：owner 或 admin 协作者
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在")
    if access_level not in ("owner", "admin", "write"):
        raise BusinessException(code=403, msg="无权限修改此知识库")

    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise BusinessException(code=404, msg="知识库不存在")

    if req.name is not None:
        kb.name = req.name
    if req.desc is not None:
        kb.desc = req.desc

    await db.commit()
    return {
        "id": kb.id,
        "name": kb.name,
        "desc": kb.desc,
        "message": "更新成功"
    }


@router.get(
    "/kb/{kb_id}/get_documents",
    summary="获取知识库文档列表"
)
async def get_kb_documents(
    kb_id: str,
    status: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取指定知识库的文档列表（含权限校验）
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    query = select(DocumentInfo).where(
        DocumentInfo.kb_id == kb_id,
        DocumentInfo.status != "deleted"
    )
    if status:
        query = query.where(DocumentInfo.status == status)

    result = await db.execute(query.order_by(DocumentInfo.create_time.desc()))
    docs = result.scalars().all()

    return {
        "documents": [
            {
                "id": doc.id,
                "file_name": doc.file_name,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "chunk_count": doc.chunk_count,
                "status": doc.status,
                "error_msg": doc.error_msg,
                "create_time": doc.create_time.strftime("%Y-%m-%d %H:%M:%S") if doc.create_time else None,
                "update_time": doc.update_time.strftime("%Y-%m-%d %H:%M:%S") if doc.update_time else None,
            }
            for doc in docs
        ]
    }


@router.get(
    "/kb/{kb_id}/documents/{doc_id}/chunks",
    summary="获取文档分块数据"
)
async def get_document_chunks(
    kb_id: str,
    doc_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取指定文档的分块数据（从向量库查询）
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    # 查询文档信息
    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    logger.info(f"获取文档分块: doc_id={doc_id}, file_name={doc.file_name}, status={doc.status}, chunk_count={doc.chunk_count}")

    # 从向量库获取分块数据
    chunks = []
    try:
        # 获取向量存储实例
        from backend.app.utils.vector_store import get_vector_store
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        vector_store = get_vector_store(collection_name=collection_name)
        
        # 获取所有文档
        client = vector_store.get_client
        all_data = client.get()
        
        total_docs = len(all_data['ids']) if all_data and all_data.get('ids') else 0
        logger.info(f"向量库文档总数: {total_docs}")
        
        if all_data and all_data['ids']:
            # 收集所有唯一的 source 值用于调试
            all_sources = set()
            for metadata in all_data['metadatas']:
                if metadata:
                    all_sources.add(metadata.get('source', ''))
            logger.info(f"向量库中的所有 source 值: {all_sources}")
            
            # 查找匹配的分块
            target_file_name = doc.file_name
            # 提取文件名（去除路径）
            import os
            target_base_name = os.path.basename(target_file_name) if target_file_name else ''
            
            for chunk_id, content, metadata in zip(
                all_data['ids'], all_data['documents'], all_data['metadatas']
            ):
                metadata = metadata or {}
                source = metadata.get('source', '')
                
                # 多种匹配方式：完整文件名、基础文件名
                is_match = (
                    source == target_file_name or
                    source == target_base_name or
                    source.lower() == target_file_name.lower() or
                    source.lower() == target_base_name.lower()
                )
                
                if is_match:
                    chunk_index = metadata.get('chunk_index', len(chunks))
                    chunks.append({
                        "id": chunk_id,
                        "index": chunk_index,
                        "content": content,
                        "length": len(content) if content else 0,
                        "metadata": metadata
                    })
            
            logger.info(f"通过 source 匹配到 {len(chunks)} 个分块")
            
            # 如果 source 匹配不到，尝试使用 id 前缀匹配
            if len(chunks) == 0 and total_docs > 0:
                logger.info("source 匹配无结果，尝试其他匹配方式...")
                for chunk_id, content, metadata in zip(
                    all_data['ids'], all_data['documents'], all_data['metadatas']
                ):
                    metadata = metadata or {}
                    if chunk_id.startswith(f"{doc_id}_") or doc_id in chunk_id:
                        chunk_index = metadata.get('chunk_index', len(chunks))
                        chunks.append({
                            "id": chunk_id,
                            "index": chunk_index,
                            "content": content,
                            "length": len(content) if content else 0,
                            "metadata": metadata
                        })
                logger.info(f"通过 id 匹配到 {len(chunks)} 个分块")
                
    except Exception as e:
        logger.warning(f"获取分块数据失败: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    logger.info(f"最终匹配到的分块数量: {len(chunks)}")

    # 检查分片数量是否与数据库记录一致
    actual_chunk_count = len(chunks)
    if actual_chunk_count != doc.chunk_count:
        logger.warning(f"分片数量不一致: 数据库记录={doc.chunk_count}, 实际匹配={actual_chunk_count}, 自动更新数据库")
        doc.chunk_count = actual_chunk_count
        await db.commit()
        await db.refresh(doc)

    # 按索引排序
    chunks.sort(key=lambda x: x.get('index', 0))

    return {
        "doc_id": doc_id,
        "file_name": doc.file_name,
        "chunk_count": actual_chunk_count,
        "db_chunk_count": doc.chunk_count,  # 数据库原记录数量
        "chunks": chunks
    }


@router.delete(
    "/kb/{kb_id}/document/{doc_id}",
    summary="删除知识库中的文档"
)
async def delete_kb_document(
    kb_id: str,
    doc_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除指定知识库中的文档（权限校验：owner 或 write/admin 协作者）
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    # 协作者需要 write 或 admin 权限
    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level == "read":
            raise BusinessException(code=403, msg="只读协作者无权删除文档")

    # 查询文档
    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    # 删除向量库中的文档
    try:
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        vs.delete_by_filter({"source": doc.file_name})
    except Exception as e:
        logger.warning(f"[doc_delete] 向量删除失败（不影响文档记录删除）: kb={kb_id}, file={doc.file_name}, error={e}")

    # 软删除文档记录
    doc.status = "deleted"
    await db.commit()

    return {"success": True, "message": "文档已删除"}


@router.post(
    "/kb/{kb_id}/documents/{doc_id}/reprocess",
    summary="重新处理文档分片"
)
async def reprocess_document_chunks(
    kb_id: str,
    doc_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    重新处理文档分片：删除旧分片，重新解析和分块
    """
    import asyncio
    import traceback
    
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    # 查询文档
    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    logger.info(f"[reprocess] 开始重新处理: doc_id={doc_id}, file={doc.file_name}")

    # 检查文件是否存在
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise BusinessException(code=400, msg="文件不存在，无法重新处理")

    try:
        # 1. 删除旧的向量数据
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        vs.delete_by_filter({"source": doc.file_name})
        logger.info(f"[reprocess] 已删除旧向量数据")

        # 2. 重新解析和分块
        result = await asyncio.to_thread(
            knowledge_service.upload_document,
            file_path=doc.file_path,
            kb_id=kb_id,
            user_id=current_user,
        )
        logger.info(f"[reprocess] 分块完成: chunks={result['chunks']}")

        # 3. 更新数据库记录
        doc.chunk_count = result["chunks"]
        doc.status = "completed"
        doc.update_time = datetime.now()
        await db.commit()
        await db.refresh(doc)

        return {
            "success": True,
            "doc_id": doc_id,
            "file_name": doc.file_name,
            "chunk_count": result["chunks"],
            "message": f"重新处理成功，共生成 {result['chunks']} 个分片"
        }

    except BusinessException:
        raise
    except Exception as e:
        logger.error(f"[reprocess] 重新处理失败: {e}\n{traceback.format_exc()}")
        doc.status = "failed"
        doc.error_msg = str(e)
        await db.commit()
        raise BusinessException(code=500, msg=f"重新处理失败: {e}")


@router.get(
    "/kb/{kb_id}/documents/{doc_id}/config",
    summary="获取文档当前配置参数"
)
async def get_document_config(
    kb_id: str,
    doc_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    # 从 chunk_config 字段获取配置，或返回默认配置
    config = {}
    if doc.chunk_config:
        try:
            import json
            config = json.loads(doc.chunk_config) if isinstance(doc.chunk_config, str) else doc.chunk_config
        except Exception:
            pass

    # 默认配置
    default_config = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 6,
        "score_threshold": 0.0,
        "embedding_model": "bge",
        "chunkParamMode": "default"
    }

    # 合并：如果 config 中有自定义参数，使用自定义
    if config.get("chunk_size") or config.get("chunk_overlap"):
        default_config.update(config)
        default_config["chunkParamMode"] = "custom"
    else:
        default_config.update(config)

    return {"success": True, "config": default_config}


class ReprocessReq(BaseModel):
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50
    top_k: Optional[int] = 6
    score_threshold: Optional[float] = 0.0
    embedding_model: Optional[str] = "bge"


@router.post(
    "/kb/{kb_id}/documents/{doc_id}/preview",
    summary="调整参数-数据预览"
)
async def document_preview(
    kb_id: str,
    doc_id: str,
    req: AdjustPreviewReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    import traceback

    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    if not doc.file_path or not os.path.exists(doc.file_path):
        raise BusinessException(code=400, msg="文件不存在，无法生成预览")

    try:
        result = await knowledge_service.preview_document(
            doc_id=doc_id,
            file_path=doc.file_path,
            user_id=current_user,
            chunk_size=req.chunk_size or 500,
            chunk_overlap=req.chunk_overlap or 50,
            separators=req.separators,
            top_k=req.top_k or 6,
            score_threshold=req.score_threshold or 0.0,
            embedding_model=req.embedding_model or "bge",
        )
        return {"success": True, **result}
    except BusinessException:
        raise
    except Exception as e:
        logger.error(f"[document_preview] 失败: {e}\n{traceback.format_exc()}")
        raise BusinessException(code=500, msg=f"预览生成失败: {str(e)}")


@router.post(
    "/kb/{kb_id}/documents/{doc_id}/reprocess_with_config",
    summary="使用新配置重新处理文档（覆盖旧分片）"
)
async def reprocess_with_config(
    kb_id: str,
    doc_id: str,
    req: ReprocessReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    import asyncio
    import json
    import traceback

    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    if not doc.file_path or not os.path.exists(doc.file_path):
        raise BusinessException(code=400, msg="文件不存在，无法重新处理")

    logger.info(f"[reprocess_with_config] 开始: doc_id={doc_id}, chunk_size={req.chunk_size}")

    try:
        # 1. 删除旧的向量数据
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        vs.delete_by_filter({"source": doc.file_name})
        logger.info(f"[reprocess_with_config] 已删除旧向量数据")

        # 2. 使用新配置重新解析和分块
        old_chunk_size = rag_split_config.chunk_size
        old_chunk_overlap = rag_split_config.chunk_overlap
        try:
            rag_split_config.chunk_size = req.chunk_size or 500
            rag_split_config.chunk_overlap = req.chunk_overlap or 50

            result = await asyncio.to_thread(
                knowledge_service.upload_document,
                file_path=doc.file_path,
                kb_id=kb_id,
                user_id=current_user,
            )
        finally:
            rag_split_config.chunk_size = old_chunk_size
            rag_split_config.chunk_overlap = old_chunk_overlap

        # 3. 保存配置到数据库
        config_data = {
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
            "top_k": req.top_k,
            "score_threshold": req.score_threshold,
            "embedding_model": req.embedding_model,
        }
        doc.chunk_config = json.dumps(config_data, ensure_ascii=False)
        doc.chunk_count = result["chunks"]
        doc.status = "completed"
        doc.update_time = datetime.now()
        await db.commit()
        await db.refresh(doc)

        return {
            "success": True,
            "doc_id": doc_id,
            "file_name": doc.file_name,
            "chunk_count": result["chunks"],
            "message": f"重新处理成功，共生成 {result['chunks']} 个分片"
        }

    except BusinessException:
        raise
    except Exception as e:
        logger.error(f"[reprocess_with_config] 失败: {e}\n{traceback.format_exc()}")
        doc.status = "failed"
        doc.error_msg = str(e)
        await db.commit()
        raise BusinessException(code=500, msg=f"重新处理失败: {str(e)}")


@router.post(
    "/kb/{kb_id}/documents/batch-delete",
    summary="批量删除知识库中的文档"
)
async def batch_delete_kb_documents(
    kb_id: str,
    req: BatchDeleteDocsReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    批量删除指定知识库中的文档
    权限：owner 或 write/admin 协作者
    """
    if not req.doc_ids:
        raise BusinessException(code=400, msg="请提供要删除的文档ID列表")

    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level == "read":
            raise BusinessException(code=403, msg="只读协作者无权删除文档")

    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id.in_(req.doc_ids),
            DocumentInfo.kb_id == kb_id
        )
    )
    docs = doc_result.scalars().all()
    if not docs:
        raise BusinessException(code=404, msg="未找到要删除的文档")

    deleted_count = 0
    try:
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        for doc in docs:
            try:
                vs.delete_by_filter({"source": doc.file_name})
            except Exception as e:
                logger.warning(f"[kb_delete] 向量删除失败: kb={kb_id}, file={doc.file_name}, error={e}")
            doc.status = "deleted"
            deleted_count += 1
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise BusinessException(code=500, msg=f"批量删除失败: {str(e)}")

    return {"success": True, "message": f"成功删除 {deleted_count} 个文档", "deleted_count": deleted_count}


@router.delete(
    "/kb/{kb_id}/document/{doc_id}",
    summary="删除知识库中的单个文档"
)
async def delete_kb_document(
    kb_id: str,
    doc_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除指定知识库中的单个文档
    权限：owner 或 write/admin 协作者
    """
    access_level = await _check_kb_access(db, kb_id, current_user)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")

    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == current_user,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level == "read":
            raise BusinessException(code=403, msg="只读协作者无权删除文档")

    doc_result = await db.execute(
        select(DocumentInfo).where(
            DocumentInfo.id == doc_id,
            DocumentInfo.kb_id == kb_id
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise BusinessException(code=404, msg="文档不存在")

    try:
        collection_name = knowledge_service.get_kb_collection_name(kb_id)
        from backend.app.utils.vector_store import get_vector_store
        vs = get_vector_store(collection_name=collection_name)
        try:
            vs.delete_by_filter({"source": doc.file_name})
        except Exception as e:
            logger.warning(f"[kb_delete] 向量删除失败: kb={kb_id}, file={doc.file_name}, error={e}")

        doc.status = "deleted"
        await db.commit()

        # 更新知识库文档数
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        )
        kb = kb_result.scalar_one_or_none()
        if kb and kb.doc_count > 0:
            kb.doc_count -= 1
            await db.commit()

    except Exception as e:
        await db.rollback()
        raise BusinessException(code=500, msg=f"删除失败: {str(e)}")

    return {"success": True, "message": "文档已删除"}


# ====== 四步上传流程（staging / preview / confirm） ======

async def _check_kb_write_access(db: AsyncSession, kb_id: str, user_id: str) -> KnowledgeBase:
    """
    校验用户对知识库的写权限（owner / write / admin）
    返回知识库对象，用于后续读取 folder_id 等信息
    """
    access_level = await _check_kb_access(db, kb_id, user_id)
    if not access_level:
        raise BusinessException(code=404, msg="知识库不存在或无访问权限")
    if access_level == "collaborator":
        member_result = await db.execute(
            select(KbMember).where(
                KbMember.kb_id == kb_id,
                KbMember.member_type == "user",
                KbMember.member_id == user_id,
                KbMember.status == "active"
            )
        )
        member = member_result.scalar_one_or_none()
        if member and member.access_level == "read":
            raise BusinessException(code=403, msg="只读协作者无权操作")

    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.status == "active"
        )
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise BusinessException(code=404, msg="知识库不存在")
    return kb


@router.post(
    "/upload/staging",
    summary="第一步：上传文件暂存"
)
async def upload_staging(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """
    将文件暂存到 ./uploads/user_{user_id}/session_{batch_id}/
    返回 batch_id 供后续步骤使用
    """
    if not file.filename:
        raise BusinessException(code=400, msg="文件名为空")

    # --- 大小校验（防止前端绕过） ---
    # 先检查 Content-Length（如果请求头有带，不等读完整个文件）
    if file.size and file.size > 0:
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file.size > max_bytes:
            raise BusinessException(
                code=400,
                msg=f"文件过大: {file.filename} ({file.size / 1024 / 1024:.1f}MB)，"
                    f"单个文件最大支持 {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

    content = await file.read()
    if not content:
        raise BusinessException(code=400, msg="文件内容为空")

    # 读完再校验一次（兜底，防止 Content-Length 不准）
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise BusinessException(
            code=400,
            msg=f"文件过大: {file.filename} ({len(content) / 1024 / 1024:.1f}MB)，"
                f"单个文件最大支持 {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # 单用户磁盘配额检查
    user_staging_dir = knowledge_service.get_staging_dir(current_user)
    current_size = 0
    if os.path.exists(user_staging_dir):
        for dirpath, dirnames, filenames in os.walk(user_staging_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    current_size += os.path.getsize(fp)
    new_total = current_size + len(content)
    if new_total > settings.MAX_STORAGE_PER_USER_MB * 1024 * 1024:
        raise BusinessException(
            code=400,
            msg=f"暂存空间不足：当前 {current_size / 1024 / 1024:.1f}MB，"
                f"本次 {len(content) / 1024 / 1024:.1f}MB，"
                f"上限 {settings.MAX_STORAGE_PER_USER_MB}MB。"
                f"请先清理之前的暂存文件（等待兜底清理或联系管理员）"
        )

    try:
        result = knowledge_service.staging_upload(
            file_content=content,
            filename=file.filename,
            user_id=current_user,
        )
        return {
            "success": True,
            "batch_id": result["batch_id"],
            "filename": result["filename"],
            "file_size": result["file_size"],
            "message": "文件暂存成功，请进入参数设置",
        }
    except BusinessException:
        raise
    except Exception as e:
        raise BusinessException(code=500, msg=f"文件暂存失败: {str(e)}")


@router.post(
    "/upload/preview",
    summary="第三步：数据预览"
)
async def upload_preview(
    req: PreviewReq,
    current_user: str = Depends(get_current_user),
):
    """
    按用户设置的参数对暂存文件进行解析、切分，
    返回前 N 个切片预览，完整切片结果缓存到 Redis（1 小时）
    """
    try:
        result = await knowledge_service.preview_upload(
            batch_id=req.batch_id,
            user_id=current_user,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            separators=req.separators,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            embedding_model=req.embedding_model,
        )
        return {"success": True, **result}
    except BusinessException:
        raise
    except Exception as e:
        import traceback
        from backend.app.common.logger import logger
        logger.error(f"[upload_preview] 数据预览失败: {e}\n{traceback.format_exc()}")
        raise BusinessException(code=500, msg=f"数据预览失败: {str(e)}")


@router.post(
    "/upload/confirm",
    summary="第四步：确认上传"
)
async def upload_confirm(
    req: ConfirmUploadReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    从缓存读取切片 -> 向量化入库 -> 写入 DocumentInfo -> 移动原始文件 -> 清理暂存
    """
    # 校验写权限并获取知识库信息
    kb = await _check_kb_write_access(db, req.kb_id, current_user)

    try:
        result = await knowledge_service.confirm_upload(
            batch_id=req.batch_id,
            kb_id=req.kb_id,
            cache_key=req.cache_key,
            user_id=current_user,
            folder_id=kb.folder_id,
        )
    except BusinessException:
        raise
    except Exception as e:
        raise BusinessException(code=500, msg=f"确认上传失败: {str(e)}")

    # service 已构造 doc_info 但未入库，由 router 写入数据库
    doc_info = result["doc_info"]
    try:
        db.add(doc_info)
        await db.commit()
        await db.refresh(doc_info)
    except Exception as e:
        import traceback
        from backend.app.common.logger import logger
        logger.error(f"[upload_confirm] 文档信息入库失败: {e}\n{traceback.format_exc()}")
        await db.rollback()
        raise BusinessException(code=500, msg=f"文档信息入库失败: {str(e)}")

    return {
        "success": True,
        "doc_id": doc_info.id,
        "filename": result["filename"],
        "chunks": result["chunks"],
        "message": result["message"],
    }


# ====== 可用模型列表 ======

@router.get(
    "/embedding_models",
    summary="获取可用的嵌入模型列表"
)
async def get_embedding_models(
    current_user: str = Depends(get_current_user),
):
    """
    返回当前服务实际可用的嵌入模型列表（根据 env 配置动态判断，
    比如 openai 未配置 API key 时不会出现在列表中）
    """
    from backend.app.utils.models_factory import model_factory
    models = model_factory.get_available_embeddings()
    default_provider = model_factory.get_default_embedding_provider()
    return {
        "success": True,
        "default": default_provider,
        "models": models,
    }


@router.post(
    "/upload/cleanup",
    summary="清理上传暂存文件（单个或批量）"
)
async def upload_cleanup(
    body: dict,
    current_user: str = Depends(get_current_user),
):
    """
    清理指定 batch_id 列表对应的暂存目录。
    前端在"退出"或"清除"时调用，同步删除磁盘上的暂存文件。
    body: {"batch_ids": ["xxx", "yyy", ...]}
    """
    batch_ids = body.get("batch_ids", [])
    if not batch_ids:
        return {"success": True, "cleaned": 0}

    cleaned_count = 0
    failed = []
    for batch_id in batch_ids:
        try:
            ok = knowledge_service.cleanup_session(current_user, batch_id)
            if ok:
                cleaned_count += 1
            else:
                failed.append(batch_id)
        except Exception as e:
            logger.warning(f"[upload_cleanup] 清理 session 失败: batch={batch_id}, error={e}")
            failed.append(batch_id)

    logger.info(f"[upload_cleanup] 清理完成: user={current_user}, 成功={cleaned_count}, 失败={len(failed)}")
    return {
        "success": True,
        "cleaned": cleaned_count,
        "failed": failed,
    }
