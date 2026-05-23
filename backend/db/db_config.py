from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession,async_sessionmaker
from sqlalchemy.orm import declarative_base

from backend.app.core.config import MYSQL_USER,MYSQL_PASSWORD,MYSQL_HOST,MYSQL_PORT,MYSQL_DATABASE

sql_url = f"mysql+aiomysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

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