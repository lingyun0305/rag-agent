# backend/app/models/knowledge.py
from sqlalchemy import Column, Integer, String, Text, DATETIME, func, ForeignKey

from backend.db.db_config import Base


class KbFolder(Base):
    __tablename__ = "kb_folder"

    id = Column(String(12), primary_key=True, comment="文件夹ID(12位随机)")
    user_id = Column(String(12), ForeignKey("users.id"), nullable=False, index=True, comment="所属用户ID")
    parent_id = Column(String(12), ForeignKey("kb_folder.id"), nullable=True, comment="父文件夹ID(NULL=根目录)")
    name = Column(String(100), nullable=False, comment="文件夹名称")
    collaborator_count = Column(Integer, default=1, comment="协作者数量(含创建者)")
    sort_order = Column(Integer, default=0, comment="排序序号")
    status = Column(String(20), default="active", comment="状态: active / deleted")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")
    update_time = Column(DATETIME, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(String(12), primary_key=True, comment="知识库ID(12位随机)")
    name = Column(String(100), nullable=False, comment="知识库名称")
    desc = Column(Text, default="", comment="知识库描述")
    icon = Column(String(200), comment="图标")
    owner_id = Column(String(12), ForeignKey("users.id"), nullable=False, index=True, comment="创建者用户ID")
    folder_id = Column(String(12), ForeignKey("kb_folder.id"), nullable=True, comment="所在文件夹(NULL=根目录)")

    embedding_model = Column(String(100), comment="嵌入模型")
    chunk_strategy = Column(String(20), default="fixed", comment="分片策略: fixed/semantic")
    top_k = Column(Integer, default=20, comment="检索返回数量")
    collaborator_count = Column(Integer, default=1, comment="协作者数量(含创建者)")

    status = Column(String(20), default="active", comment="状态: active / deleted")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")
    update_time = Column(DATETIME, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间")


class KbMember(Base):
    __tablename__ = "kb_member"

    id = Column(String(12), primary_key=True, comment="记录ID(12位随机)")
    kb_id = Column(String(12), ForeignKey("knowledge_base.id"), nullable=True, index=True, comment="知识库ID")
    folder_id = Column(String(12), ForeignKey("kb_folder.id"), nullable=True, index=True, comment="文件夹ID")
    member_type = Column(String(20), nullable=False, comment="主体类型: user / department / team / group")
    member_id = Column(String(50), nullable=False, comment="主体ID")
    access_level = Column(String(20), default="read", comment="权限等级: read / write / admin")
    created_by = Column(String(12), ForeignKey("users.id"), comment="操作人用户ID")
    status = Column(String(20), default="active", comment="状态: active / removed")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")


class DocumentInfo(Base):
    __tablename__ = "document_info"

    id = Column(String(12), primary_key=True, comment="文档ID(12位随机)")
    kb_id = Column(String(12), ForeignKey("knowledge_base.id"), comment="所属知识库")
    file_name = Column(String(200), nullable=False, comment="文件名")
    file_type = Column(String(20), comment="文件类型: pdf/txt/md/docx/csv")
    file_size = Column(Integer, default=0, comment="文件大小(字节)")
    file_path = Column(String(500), comment="原始文件存储路径")
    chunk_count = Column(Integer, default=0, comment="分片数量")
    status = Column(String(20), default="pending", comment="状态: pending / processing / completed / failed / deleted")
    error_msg = Column(Text, comment="失败原因")
    chunk_config = Column(Text, comment="分块配置参数(JSON)")
    create_time = Column(DATETIME, default=func.now(), comment="创建时间")
    update_time = Column(DATETIME, default=func.now(), onupdate=func.now(), comment="更新时间")


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id = Column(String(12), primary_key=True, comment="分片ID(12位随机)")
    doc_id = Column(String(12), ForeignKey("document_info.id"), nullable=False, comment="所属文档ID")
    kb_id = Column(String(12), ForeignKey("knowledge_base.id"), index=True, comment="所属知识库ID")
    chunk_content = Column(Text, comment="分片内容")
    chunk_index = Column(Integer, default=0, comment="分片序号")
    vector_id = Column(String(100), comment="ChromaDB向量ID")
    status = Column(String(20), default="active", comment="状态: active / deleted")
    create_time = Column(DATETIME, default=func.now(), comment="创建时间")