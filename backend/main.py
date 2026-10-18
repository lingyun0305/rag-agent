from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.app.agent.tools.registry import init_default_tools
from backend.app.common.exceptions import register_exceptions_handlers
from backend.app.common.logger import logger
from backend.app.common.middleware import TraceMiddleware
from backend.app.core.config import settings
from backend.app.router.agent import router as agent_router
from backend.app.router.auth import router as auth_router
from backend.app.router.chat import router as chat_router
from backend.app.router.knowledge import router as knowledge_router
from backend.app.router.upload import router as upload_router
from backend.app.services.knowledge_service import knowledge_service
from backend.db.db_config import create_tables

# ====== 应用实例 ======
app = FastAPI(
    title="RAG-AI 智能对话助手",
    description="基于检索增强生成（RAG）的企业级 AI 对话系统，支持知识库问答、Agent 工具调用和智能路由",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ====== 限流配置 ======
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_exception_handler(
    429,
    lambda request, exc: logger.error(
        "[限流触发]",
        event="rate_limit_exceeded",
        path=request.url.path,
        remote=get_remote_address(request),
    ),
)

# ====== 中间件（按注册顺序倒序执行） ======
# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 限流中间件
app.add_middleware(SlowAPIMiddleware)

# 3. 链路追踪中间件（最内层，最后执行）
app.add_middleware(TraceMiddleware)

# ====== 路由挂载 ======
app.mount("/front", StaticFiles(directory="front", html=True), name="font")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_STAGING_DIR), name="uploads")

# API 路由
app.include_router(chat_router)
app.include_router(auth_router, prefix="/api/auth")
app.include_router(knowledge_router, prefix="/api/knowledge")
app.include_router(agent_router, prefix="/api/agent")
app.include_router(upload_router, prefix="/api/upload")

# ====== 异常处理 ======
register_exceptions_handlers(app)


# ====== 启动事件 ======
@app.on_event("startup")
async def startup_event():
    logger.info("[服务启动]", event="app_startup")
    await create_tables()

    # 初始化默认工具（内存注册表）
    init_default_tools()

    # 确保系统工具存在于数据库中（前端展示 + Agent 绑定需要）
    try:
        from backend.app.router.agent import ensure_system_tools
        from backend.db.db_config import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await ensure_system_tools(db)
        logger.info("系统工具数据库初始化完成")
    except Exception as e:
        logger.warning(f"系统工具数据库初始化失败: {e}")

    # 兜底清理：删除超过 24 小时的上传暂存目录
    try:
        cleaned = knowledge_service.cleanup_expired(max_age_hours=24)
        if cleaned > 0:
            logger.kb_operation("cleanup", file_count=cleaned)
    except Exception as e:
        logger.warning(f"启动清理暂存目录失败: {e}")

    logger.info("[服务启动完成]", event="app_startup_done")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[服务关闭]", event="app_shutdown")


# ====== 根路径 ======
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/front/index.html")


# ====== 健康检查 ======
@app.get("/health", tags=["系统"], summary="健康检查")
@limiter.exempt
async def health_check():
    """健康检查接口，用于 Docker/负载均衡器探活"""
    return {
        "status": "healthy",
        "service": "rag-ai-chat",
        "version": "2.0.0",
    }


@app.get("/metrics", tags=["系统"], summary="系统指标")
@limiter.exempt
async def metrics():
    """系统指标接口"""
    from backend.app.core.cache import cache_service

    cache_stats = await cache_service.get_stats()
    return {
        "cache": cache_stats,
    }
