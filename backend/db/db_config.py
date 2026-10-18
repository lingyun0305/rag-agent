# backend/db/db_config.py
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from backend.app.core.config import settings

mysql_user = settings.MYSQL_USER
mysql_password = settings.MYSQL_PASSWORD
mysql_host = settings.MYSQL_HOST
mysql_port = settings.MYSQL_PORT
mysql_database = settings.MYSQL_DATABASE

sql_url = f"mysql+aiomysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}?charset=utf8mb4"

# 创建异步引擎
async_engine = create_async_engine(
    url=sql_url, echo=True, pool_recycle=300, pool_pre_ping=True, pool_size=10, max_overflow=20
)
# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# 创建依赖项
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# 创建基类
Base = declarative_base()


async def _ensure_migration_log_table():
    """确保迁移日志表存在"""
    async with async_engine.begin() as conn:
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS _migration_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                migration_name VARCHAR(100) NOT NULL UNIQUE,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        )


async def _migration_applied(conn, name: str) -> bool:
    """检查迁移是否已应用"""
    result = await conn.execute(
        text("SELECT COUNT(*) FROM _migration_log WHERE migration_name = :name"), {"name": name}
    )
    return result.scalar() > 0


async def _mark_migration(conn, name: str):
    """标记迁移为已应用"""
    await conn.execute(
        text("INSERT INTO _migration_log (migration_name) VALUES (:name)"), {"name": name}
    )


# 创建所有表
async def create_tables():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 确保迁移日志表存在
    await _ensure_migration_log_table()

    # 一次性迁移（带日志跟踪，只执行一次）
    migrations = [
        ("document_info_columns", _ensure_document_info_columns),
        ("kb_member_columns_nullable", _ensure_kb_member_columns),
        ("agent_columns_add_fields", _ensure_agent_columns),
        ("agent_tool_configs_column", _ensure_agent_tool_configs),
        ("tool_columns_add_fields", _ensure_tool_columns),
        ("chat_session_fk_removal", _ensure_chat_session_columns),
        ("chat_message_thinking_steps", _ensure_chat_message_thinking_columns),
        ("chat_message_drop_unused_columns", _ensure_chat_message_drop_unused_columns),
        ("chat_message_drop_react_columns", _ensure_chat_message_drop_react_columns),
        ("task_message_create_table", _ensure_task_message_table),
    ]

    for name, fn in migrations:
        async with async_engine.begin() as conn:
            if not await _migration_applied(conn, name):
                print(f"[DB] 执行迁移: {name}")
                try:
                    await fn(conn)
                    await _mark_migration(conn, name)
                    print(f"[DB] 迁移完成: {name}")
                except Exception as e:
                    print(f"[DB] 迁移失败: {name} - {e}")


async def _ensure_document_info_columns(conn):
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
    # 检查表是否存在
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'document_info'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    # 表存在则补齐缺失的列
    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'document_info'"
        )
    )
    existing = {row[0] for row in result}
    for col_name, col_def in required_columns.items():
        if col_name not in existing:
            await conn.execute(text(f"ALTER TABLE document_info ADD COLUMN {col_name} {col_def}"))


async def _ensure_kb_member_columns(conn):
    """
    检查并修复 kb_member 表结构。
    确保 kb_id 和 folder_id 字段允许 NULL。
    """
    # 检查表是否存在
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'kb_member'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    # 修复 kb_id 字段允许 NULL
    await conn.execute(text("ALTER TABLE kb_member MODIFY COLUMN kb_id VARCHAR(12) NULL"))

    # 修复 folder_id 字段允许 NULL
    try:
        await conn.execute(text("ALTER TABLE kb_member MODIFY COLUMN folder_id VARCHAR(12) NULL"))
    except Exception:
        pass


async def _ensure_agent_columns(conn):
    """
    检查并修复 agent 表结构。
    添加缺失的 system_tool_names 字段。
    """
    # 检查 agent 表是否存在
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'agent'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    # 检查并添加缺失的列
    required_columns = {
        "system_tool_names": "TEXT DEFAULT '[]' COMMENT '系统工具名称列表JSON'",
        "greeting": "TEXT DEFAULT '' COMMENT '对话开场白'",
    }

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'agent'"
        )
    )
    existing = {row[0] for row in result}

    for col_name, col_def in required_columns.items():
        if col_name not in existing:
            await conn.execute(text(f"ALTER TABLE agent ADD COLUMN {col_name} {col_def}"))

    # 修正 tool 表的 owner_id 类型
    tool_exists = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'tool'"
        )
    )
    if tool_exists.scalar():
        await conn.execute(
            text(
                "ALTER TABLE tool MODIFY COLUMN owner_id VARCHAR(12) NULL COMMENT '创建者用户ID(系统工具为NULL)'"
            )
        )


async def _ensure_agent_tool_configs(conn):
    """
    检查并补齐 agent 表缺失的列。
    添加 tool_configs 字段, 用于存储每个 Agent 绑定工具的参数配置
    (例如 knowledge_search 工具独立的 top_k / strategy / use_rerank)。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'agent'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    required_columns = {
        "tool_configs": "TEXT NULL COMMENT '工具级配置JSON, key=工具名'",
    }

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'agent'"
        )
    )
    existing = {row[0] for row in result}

    for col_name, col_def in required_columns.items():
        if col_name not in existing:
            await conn.execute(text(f"ALTER TABLE agent ADD COLUMN {col_name} {col_def}"))


async def _ensure_tool_columns(conn):
    """
    检查并补齐 tool 表缺失的列。
    添加缺失的 icon, code_path, parameters_schema 字段。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'tool'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    required_columns = {
        "icon": "VARCHAR(200) DEFAULT NULL COMMENT '图标'",
        "code_path": "VARCHAR(500) DEFAULT NULL COMMENT '代码路径或API endpoint'",
        "parameters_schema": "TEXT COMMENT '参数JSON Schema'",
        "visibility": "VARCHAR(20) DEFAULT 'public' COMMENT '可见性: public=全员可见 / private=仅创建者'",
    }

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'tool'"
        )
    )
    existing = {row[0] for row in result}

    for col_name, col_def in required_columns.items():
        if col_name not in existing:
            await conn.execute(text(f"ALTER TABLE tool ADD COLUMN {col_name} {col_def}"))


async def _ensure_chat_session_columns(conn):
    """
    检查并修改 chat_session 表结构。
    去掉外键约束，允许 user_id 和 agent_id 为空。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_session'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    # 查找并删除外键约束
    result = await conn.execute(
        text(
            "SELECT CONSTRAINT_NAME FROM information_schema.TABLE_CONSTRAINTS "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_session' "
            "AND CONSTRAINT_TYPE = 'FOREIGN KEY'"
        )
    )
    fk_constraints = result.fetchall()

    for fk in fk_constraints:
        constraint_name = fk[0]
        await conn.execute(text(f"ALTER TABLE chat_session DROP FOREIGN KEY `{constraint_name}`"))

    # 修改字段长度和允许 NULL
    await conn.execute(
        text("ALTER TABLE chat_session MODIFY COLUMN id VARCHAR(50) NOT NULL COMMENT '会话ID'")
    )
    await conn.execute(
        text("ALTER TABLE chat_session MODIFY COLUMN user_id VARCHAR(50) NULL COMMENT '用户ID'")
    )
    await conn.execute(
        text("ALTER TABLE chat_session MODIFY COLUMN agent_id VARCHAR(50) NULL COMMENT 'Agent ID'")
    )

    # 修改 chat_message 表的字段长度
    result2 = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    if result2.scalar():
        await conn.execute(
            text("ALTER TABLE chat_message MODIFY COLUMN id VARCHAR(50) NOT NULL COMMENT '消息ID'")
        )
        await conn.execute(
            text(
                "ALTER TABLE chat_message MODIFY COLUMN session_id VARCHAR(50) NOT NULL COMMENT '会话ID'"
            )
        )


async def _ensure_chat_message_thinking_columns(conn):
    """
    检查并补齐 chat_message 表的思考过程相关列。
    添加 thinking_steps 和 tool_calls 字段，用于持久化Agent的ReAct过程。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    exists = result.scalar()
    if not exists:
        return

    required_columns = {
        "thinking_steps": "TEXT NULL COMMENT '思考过程步骤JSON（含thinking/tool_result事件列表）'",
        "tool_calls": "TEXT NULL COMMENT '本次调用的工具列表JSON'",
    }

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    existing = {row[0] for row in result}

    for col_name, col_def in required_columns.items():
        if col_name not in existing:
            await conn.execute(text(f"ALTER TABLE chat_message ADD COLUMN {col_name} {col_def}"))
            print(f"[DB] chat_message 表添加列: {col_name}")


async def _ensure_chat_message_drop_unused_columns(conn):
    """
    清理 chat_message 表的冗余列。
    tool_name / tool_input / tool_output 历史上用于单独存储单次工具调用的
    入参/出参，但实际从未被填充（调用方一律不传），且完整的工具调用信息
    已经统一存放在 thinking_steps(JSON) 与 tool_calls(JSON) 两个字段里。
    删除这三个冗余列，避免歧义与存储浪费。幂等：列不存在则跳过。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    if not result.scalar():
        return

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    existing = {row[0] for row in result}

    unused = ["tool_name", "tool_input", "tool_output"]
    for col in unused:
        if col in existing:
            try:
                await conn.execute(text(f"ALTER TABLE chat_message DROP COLUMN `{col}`"))
                print(f"[DB] chat_message 表删除冗余列: {col}")
            except Exception as e:
                print(f"[DB] 删除列 {col} 失败（可忽略）: {e}")


async def _ensure_chat_message_drop_react_columns(conn):
    """
    从 chat_message 表删除 ReAct 相关列（step_type / thinking_steps / tool_calls）。
    这些字段已迁移到独立的 task_message 表，此处清理旧列。
    幂等：列不存在则跳过。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    if not result.scalar():
        return

    result = await conn.execute(
        text(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'chat_message'"
        )
    )
    existing = {row[0] for row in result}

    react_columns = ["step_type", "thinking_steps", "tool_calls"]
    for col in react_columns:
        if col in existing:
            try:
                await conn.execute(text(f"ALTER TABLE chat_message DROP COLUMN `{col}`"))
                print(f"[DB] chat_message 表删除 ReAct 列: {col}")
            except Exception as e:
                print(f"[DB] 删除 ReAct 列 {col} 失败（可忽略）: {e}")


async def _ensure_task_message_table(conn):
    """
    创建 task_message 表（ReAct 逐步推理轨迹表）。
    每条记录对应一个 thought/action/observation/answer/error 步骤。
    幂等：表已存在则跳过。
    """
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'task_message'"
        )
    )
    if result.scalar():
        return  # 表已存在

    await conn.execute(
        text("""
        CREATE TABLE task_message (
            id VARCHAR(50) NOT NULL COMMENT '主键',
            session_id VARCHAR(50) NOT NULL COMMENT '所属会话ID，与消息表对齐',
            message_id VARCHAR(50) DEFAULT NULL COMMENT '关联的助手消息ID',
            step INT NOT NULL COMMENT '当前步数，从1开始递增',
            step_type VARCHAR(20) NOT NULL COMMENT '轨迹类型：thought/action/observation/answer/error',
            content TEXT DEFAULT NULL COMMENT '具体内容',
            tool_name VARCHAR(50) DEFAULT NULL COMMENT '工具名称，仅action类型有值',
            tool_input TEXT DEFAULT NULL COMMENT '工具入参，仅action类型有值',
            token_count INT DEFAULT 0 COMMENT '本条内容的token数',
            latency_ms INT DEFAULT 0 COMMENT '本步骤耗时(毫秒)',
            model_name VARCHAR(50) DEFAULT NULL COMMENT '所用模型',
            create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            PRIMARY KEY (id),
            INDEX idx_task_session_msg (session_id, message_id),
            INDEX idx_task_session_step (session_id, step)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    )
    print("[DB] task_message 表创建成功")
