# backend/app/common/middleware.py
# 中间件
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time

from backend.app.common.logger import logger

# 日志中间件
class loggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request:Request, call_next):
        # 开始时间
        start_time = time.time()
        # 记录请求
        logger.info(f"【请求】{request.method} {request.url.path}")
        # 执行接口
        response = await call_next(request)
        # 计算耗时
        duration = round((time.time()-start_time)*1000,2)
        # 记录响应状态码和过程的耗时
        logger.info(f"【响应】{response.status_code} | 【耗时】{duration}ms")
        #返回响应
        return response