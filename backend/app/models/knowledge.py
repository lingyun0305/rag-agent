from sqlalchemy import Column,Integer,String,Text,DATETIME,func,ForeignKey

from backend.db.db_config import Base

# 知识库表
class Knowledges(Base):
    __tablename__= "knowledge_base"
    id = Column(Integer,primary_key=True,index=True,autoincrement=True)
    name = Column(String(100),nullable=False,comment="知识库名称")
    desc = Column(Text,default="",comment="知识库描述")
    create_time = Column(DATETIME,nullable=False,default=func.now(),comment="创建时间")
    update_time = Column(DATETIME,nullable=False,default=func.now(),onupdate=func.now(),comment="更新时间")

# 原始文档表(用于管理上传的文档)
class DocumentInfo(Base):
    __tablename__ = "document_info"
    id = Column(Integer,primary_key=True,autoincrement=True,index=True)
    kb_id = Column(Integer,ForeignKey("knowledge_base.id"),comment="所属知识库")
    file_name = Column(String(200),nullable=False,comment="文件名")
    file_type = Column(String(20),comment="文件类型")
    file_size = Column(Integer,comment="文件大小",default=0)
    status = Column(String(20),default="pending",comment="文件状态：pending/success/fail")
    create_time = Column(DATETIME,default=func.now())

# 文档分片表（用于管理分块后的文档）
class DocumentChunk(Base):
    __tablename__ = "document_chunk"
    id = Column(Integer,autoincrement=True,index=True,primary_key=True)
    doc_id = Column(Integer,ForeignKey("document_info.id"),nullable=False,comment="所属文档id")
    kb_id = Column(Integer,comment="所属知识库id")
    chunk_content = Column(Text,comment="分片内容",default="")
    chunk_index = Column(Integer,comment="分片序号",default=0)
    vector_id = Column(String(100),comment="向量库id")
    create_time = Column(DATETIME,default=func.now())