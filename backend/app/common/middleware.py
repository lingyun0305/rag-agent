# backend/app/common/middleware.py
"""
全局中间件
- 请求链路追踪 ID 自动生成
- 结构化 HTTP 请求日志
- 响应耗时统计
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.app.common.logger import (
    logger,
    new_trace_id,
    reset_trace_id,
)


class TraceMiddleware(BaseHTTPMiddleware):
    """
    链路追踪中间件
    为每个 HTTP 请求自动生成 trace_id，贯穿整个请求生命周期
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 生成请求级 trace_id
        trace_id = new_trace_id()

        # 记录请求开始时间
        start_time = time.time()

        # 尝试从请求头获取用户 ID
        user_id = ""
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    import jwt

                    from backend.app.core.config import settings

                    payload = jwt.decode(
                        token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
                    )
                    user_id = str(payload.get("sub", ""))
                except Exception:
                    user_id = "invalid_token"
            elif auth_header:
                user_id = "no_bearer"
        except Exception:
            pass

        # 记录请求入站
        logger.info(
            "[请求入站]",
            event="http_in",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else "",
            client=request.client.host if request.client else "",
            user_id=user_id,
        )

        try:
            # 执行请求
            response = await call_next(request)

            # 计算耗时
            duration_ms = (time.time() - start_time) * 1000

            # 记录响应
            logger.http_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=duration_ms,
                user_id=user_id,
            )

            # 在响应头写入 trace_id，便于前端调试
            response.headers["X-Trace-Id"] = trace_id

            return response

        except Exception as e:
            # 异常日志
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "[请求异常]",
                event="http_error",
                method=request.method,
                path=request.url.path,
                latency_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
                user_id=user_id,
            )
            raise
        finally:
            # 清理 trace_id（防止 asyncio 上下文污染）
            reset_trace_id()


# 兼容旧代码的别名
loggerMiddleware = TraceMiddleware
