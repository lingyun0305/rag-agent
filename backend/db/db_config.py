# backend/db/db_config.py
from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession,async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from backend.app.core.config import settings

mysql_user = settings.MYSQL_USER
mysql_password = settings.MYSQL_PASSWORD
mysql_host = settings.MYSQL_HOST
mysql_port = settings.MYSQL_PORT
mysql_database = settings.MYSQL_DATABASE

sql_url = f"mysql+aiomysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}?charset=utf8mb4"

#创建异步引擎
async_engine = create_async_engine(
    url= sql_url,
    echo=True,
    pool_recycle = 300,
    pool_pre_ping = True,
    pool_size = 10,
    max_overflow = 20
)
#创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind = async_engine,
    class_ = AsyncSession,
    expire_on_commit = False,
    autoflush = False,
)
#创建依赖项
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
#创建基类
Base = declarative_base()

#创建所有表
async def create_tables():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 确保 document_info 表结构完整（兼容旧表）
    await _ensure_document_info_columns()

    # 确保 kb_member 表结构完整（kb_id 允许 NULL）
    await _ensure_kb_member_columns()

async def _ensure_document_info_columns():
    """
    检查并补齐 document_info 表缺失的列。
    兼容旧版本数据库（之前可能没有 chunk_count / status / error_msg 等字段）。
    """
    required_columns = {
        "file_type": "VARCHAR(20) DEFAULT NULL COMMENT '文件类型'",
        "file_size": "INT DEFAULT 0 COMMENT '文件大小(字节)'",
        "file_path": "VARCHAR(500) DEFAULT NULL COMMENT '原始文件存储路径'",
        "chunk_count": "INT DEFAULT 0 COMMENT '分片数量'",
        "status": "VARCHAR(20) DEFAULT 'pending' COMMENT '状态'",
        "error_msg": "TEXT COMMENT '失败原因'",
        "chunk_config": "TEXT COMMENT '分块配置参数(JSON)'",
        "create_time": "DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'",
        "update_time": "DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'",
    }
    try:
        async with async_engine.begin() as conn:
            # 检查表是否存在
            result = await conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = 'document_info'"
            ))
            exists = result.scalar()
            if not exists:
                # 表不存在则创建
                await conn.execute(text("""
                    CREATE TABLE document_info (
                        id VARCHAR(12) PRIMARY KEY COMMENT '文档ID(12位随机)',
                        kb_id VARCHAR(12) COMMENT '所属知识库',
                        file_name VARCHAR(200) NOT NULL COMMENT '文件名',
                        file_type VARCHAR(20) DEFAULT NULL COMMENT '文件类型',
                        file_size INT DEFAULT 0 COMMENT '文件大小(字节)',
                        file_path VARCHAR(500) DEFAULT NULL COMMENT '原始文件存储路径',
                        chunk_count INT DEFAULT 0 COMMENT '分片数量',
                        status VARCHAR(20) DEFAULT 'pending' COMMENT '状态',
                        error_msg TEXT COMMENT '失败原因',
                        create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                        INDEX idx_kb_id (kb_id),
                        CONSTRAINT fk_doc_kb FOREIGN KEY (kb_id) REFERENCES knowledge_base(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档信息'
                """))
                print("[DB] document_info 表已创建")
                return

            # 表存在则补齐缺失的列
            result = await conn.execute(text(
                "SELECT COLUMN_NAME FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = 'document_info'"
            ))
            existing = {row[0] for row in result}
            for col_name, col_def in required_columns.items():
                if col_name not in existing:
                    await conn.execute(text(
                        f"ALTER TABLE document_info ADD COLUMN {col_name} {col_def}"
                    ))
                    print(f"[DB] document_info 表补齐列: {col_name}")
    except Exception as e:
        print(f"[DB] document_info 表结构检查失败（可忽略）: {e}")

async def _ensure_kb_member_columns():
    """
    检查并修复 kb_member 表结构。
    确保 kb_id 和 folder_id 字段允许 NULL（文件夹场景不需要 kb_id）。
    """
    try:
        async with async_engine.begin() as conn:
            # 检查 kb_member 表是否存在
            result = await conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = 'kb_member'"
            ))
            exists = result.scalar()
            if not exists:
                return

            # 修复 kb_id 字段允许 NULL
            await conn.execute(text(
                "ALTER TABLE kb_member MODIFY COLUMN kb_id VARCHAR(12) NULL"
            ))
            print("[DB] kb_member.kb_id 已设为允许 NULL")

            # 修复 folder_id 字段允许 NULL
            try:
                await conn.execute(text(
                    "ALTER TABLE kb_member MODIFY COLUMN folder_id VARCHAR(12) NULL"
                ))
                print("[DB] kb_member.folder_id 已设为允许 NULL")
            except Exception as e:
                print(f"[DB] kb_member.folder_id 修改跳过（可能不存在）: {e}")
    except Exception as e:
        print(f"[DB] kb_member 表结构检查失败（可忽略）: {e}")