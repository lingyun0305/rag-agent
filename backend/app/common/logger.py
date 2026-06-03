# backend/app/common/logger.py
# 日志
from loguru import logger
import sys,os

# 创建logs 文件夹
if not os.path.exists("logs"):
    os.mkdir("logs")

# 移除默认输出
logger.remove()

# 控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# 文件输出
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    encoding="utf-8",
    rotation="00:00",
    retention=7,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# 导出给全局使用
__all__ = ["logger"]