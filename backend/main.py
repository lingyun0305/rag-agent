from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.db.db_config import create_tables 
from backend.app.common.exceptions import register_exceptions_handlers
from backend.app.common.middleware import loggerMiddleware
from backend.app.router.chat import router as chat_router
from backend.app.router.auth import router as auth_router
from backend.app.router.knowledge import router as knowledge_router
from backend.app.services.knowledge_service import knowledge_service
from backend.app.common.logger import logger

app = FastAPI()

# 挂载页面
app.mount("/front",StaticFiles(directory="front",html=True),name="font")
# # 挂载静态资源
# app.mount("/static",StaticFiles(directory="static",html=True),name="static")
# 挂载路由
app.include_router(chat_router)
app.include_router(auth_router, prefix="/api/auth")
app.include_router(knowledge_router, prefix="/api/knowledge")
# 注册全局异常处理器
register_exceptions_handlers(app)
# 挂载全局日志中间件
app.add_middleware(loggerMiddleware)

# 创建所有数据库表
@app.on_event("startup")
async def startup_event():
    await create_tables()
    # 兜底清理：删除超过 24 小时的上传暂存目录
    try:
        cleaned = knowledge_service.cleanup_expired(max_age_hours=24)
        if cleaned > 0:
            logger.info(f"启动清理：删除 {cleaned} 个过期暂存目录")
    except Exception as e:
        logger.warning(f"启动清理暂存目录失败: {e}")

# 根路径重定向
@app.get("/")
async def root():
    return RedirectResponse("/front/index.html")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)