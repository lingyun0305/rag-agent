from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.db.db_config import create_tables 
from backend.app.common.exceptions import register_exceptions_handlers
from backend.app.common.middleware import loggerMiddleware
from backend.app.router.chat import router as chat_router
from backend.app.router.auth import router as auth_router

app = FastAPI()

# 挂载页面
app.mount("/front",StaticFiles(directory="front",html=True),name="font")
# 挂载静态资源
app.mount("/static",StaticFiles(directory="static",html=True),name="static")
# 挂载路由
app.include_router(chat_router)
app.include_router(auth_router)
# 注册全局异常处理器
register_exceptions_handlers(app)
# 挂载全局日志中间件
app.add_middleware(loggerMiddleware)

# 创建所有数据库表
@app.on_event("startup")
async def startup_event():
    await create_tables()

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