# backend/app/core/cache.py
"""
统一 Redis 缓存服务

功能：
- 统一 Redis 连接管理
- 基础缓存操作（get/set/delete）
- 带过期时间的缓存
- 缓存不存在时自动计算并缓存
- 对话历史缓存（已存在，复用）
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import redis.asyncio as redis

from backend.app.common.logger import logger
from backend.app.core.config import settings


class CacheService:
    """
    统一缓存服务

    使用方式：
    ```python
    from backend.app.core.cache import cache_service

    # 基础操作
    await cache_service.set("key", "value", expire=3600)
    value = await cache_service.get("key")

    # 自动缓存（不存在时计算）
    result = await cache_service.get_or_compute(
        key="search:xxx",
        ttl=3600,
        compute_fn=lambda: expensive_computation()
    )
    ```
    """

    def __init__(self):
        self._redis = None
        self._initialized = False
        self._fallback_cache = {}

    async def _init(self):
        """初始化 Redis 连接"""
        if self._initialized:
            return

        try:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=1,  # 连接超时1秒
                socket_timeout=1,  # 读写超时1秒
                decode_responses=False,
            )
            await asyncio.wait_for(self._redis.ping(), timeout=2)
            self._initialized = True
            logger.info("Redis 连接成功")
        except Exception as e:
            logger.warning(f"Redis 连接失败，使用内存缓存: {e}")
            self._redis = None
            self._initialized = True

    async def get(self, key: str) -> Any:
        """获取缓存"""
        await self._init()

        if self._redis is None:
            return self._fallback_cache.get(key)

        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.warning(f"Redis get 失败: {e}")
            return self._fallback_cache.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        expire: int | None = None,
        expire_time: timedelta | None = None,
    ) -> None:
        """设置缓存"""
        await self._init()

        try:
            serialized = json.dumps(value, ensure_ascii=False)

            if self._redis is not None:
                if expire_time:
                    await self._redis.set(key, serialized, expire=expire_time)
                elif expire:
                    await self._redis.setex(key, expire, serialized)
                else:
                    await self._redis.set(key, serialized)
            else:
                self._fallback_cache[key] = value
                if expire:
                    import asyncio

                    asyncio.create_task(self._clear_fallback_after(key, expire))

        except Exception as e:
            logger.warning(f"Redis set 失败: {e}")

    async def delete(self, key: str) -> None:
        """删除缓存"""
        await self._init()

        try:
            if self._redis is not None:
                await self._redis.delete(key)
            else:
                self._fallback_cache.pop(key, None)
        except Exception as e:
            logger.warning(f"Redis delete 失败: {e}")

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        await self._init()

        try:
            if self._redis is not None:
                return await self._redis.exists(key) > 0
            else:
                return key in self._fallback_cache
        except Exception as e:
            logger.warning(f"Redis exists 失败: {e}")
            return False

    async def get_or_compute(
        self,
        key: str,
        ttl: int,
        compute_fn: Callable[[], Any],
        skip_cache: bool = False,
    ) -> Any:
        """
        获取缓存，不存在时计算并缓存

        Args:
            key: 缓存键
            ttl: 过期时间（秒）
            compute_fn: 计算函数（同步或异步）
            skip_cache: 是否跳过缓存直接计算

        Returns:
            缓存值或计算结果
        """
        if not skip_cache:
            cached = await self.get(key)
            if cached is not None:
                return cached

        result = compute_fn()
        if hasattr(result, "__await__"):
            result = await result

        await self.set(key, result, expire=ttl)
        return result

    async def clear_pattern(self, pattern: str) -> int:
        """删除匹配模式的所有缓存"""
        await self._init()

        try:
            if self._redis is not None:
                keys = await self._redis.keys(pattern)
                if keys:
                    return await self._redis.delete(*keys)
            else:
                count = 0
                for key in list(self._fallback_cache.keys()):
                    if pattern.replace("*", "") in key:
                        del self._fallback_cache[key]
                        count += 1
                return count
        except Exception as e:
            logger.warning(f"Redis clear_pattern 失败: {e}")
            return 0

    async def _clear_fallback_after(self, key: str, seconds: int):
        """内存缓存过期清理"""
        await asyncio.sleep(seconds)
        self._fallback_cache.pop(key, None)

    async def get_stats(self) -> dict:
        """获取缓存统计信息"""
        await self._init()

        if self._redis is not None:
            try:
                info = await self._redis.info("stats")
                return {
                    "redis_connected": True,
                    "keys_count": info.get("keyspace_hits", 0),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                }
            except Exception as e:
                logger.debug(f"[cache] Redis INFO 命令失败: {e}")

        return {
            "redis_connected": False,
            "keys_count": len(self._fallback_cache),
            "hits": 0,
            "misses": 0,
        }

    async def lpush(self, key: str, *values: Any) -> int:
        """列表左侧插入"""
        await self._init()
        try:
            if self._redis is not None:
                serialized = [json.dumps(v, ensure_ascii=False) for v in values]
                return await self._redis.lpush(key, *serialized)
            else:
                if key not in self._fallback_cache:
                    self._fallback_cache[key] = []
                for v in values:
                    self._fallback_cache[key].insert(0, v)
                return len(self._fallback_cache[key])
        except Exception as e:
            logger.warning(f"Redis lpush 失败: {e}")
            return 0

    async def rpush(self, key: str, *values: Any) -> int:
        """列表右侧插入"""
        await self._init()
        try:
            if self._redis is not None:
                serialized = [json.dumps(v, ensure_ascii=False) for v in values]
                return await self._redis.rpush(key, *serialized)
            else:
                if key not in self._fallback_cache:
                    self._fallback_cache[key] = []
                for v in values:
                    self._fallback_cache[key].append(v)
                return len(self._fallback_cache[key])
        except Exception as e:
            logger.warning(f"Redis rpush 失败: {e}")
            return 0

    async def lrange(self, key: str, start: int, end: int) -> list[Any]:
        """获取列表指定范围元素"""
        await self._init()
        try:
            if self._redis is not None:
                raw = await self._redis.lrange(key, start, end)
                return [json.loads(v) for v in raw] if raw else []
            else:
                lst = self._fallback_cache.get(key, [])
                return lst[start : end + 1] if end >= 0 else lst[start:]
        except Exception as e:
            logger.warning(f"Redis lrange 失败: {e}")
            return []

    async def ltrim(self, key: str, start: int, end: int) -> None:
        """裁剪列表"""
        await self._init()
        try:
            if self._redis is not None:
                await self._redis.ltrim(key, start, end)
            else:
                lst = self._fallback_cache.get(key, [])
                self._fallback_cache[key] = lst[start : end + 1] if end >= 0 else lst[start:]
        except Exception as e:
            logger.warning(f"Redis ltrim 失败: {e}")

    async def llen(self, key: str) -> int:
        """获取列表长度"""
        await self._init()
        try:
            if self._redis is not None:
                return await self._redis.llen(key)
            else:
                return len(self._fallback_cache.get(key, []))
        except Exception as e:
            logger.warning(f"Redis llen 失败: {e}")
            return 0

    async def keys(self, pattern: str) -> list[str]:
        """获取匹配模式的键"""
        await self._init()
        try:
            if self._redis is not None:
                return await self._redis.keys(pattern)
            else:
                return [k for k in self._fallback_cache.keys() if pattern.replace("*", "") in k]
        except Exception as e:
            logger.warning(f"Redis keys 失败: {e}")
            return []

    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        await self._init()
        try:
            if self._redis is not None:
                return await self._redis.expire(key, seconds)
            else:
                if key in self._fallback_cache:
                    asyncio.create_task(self._clear_fallback_after(key, seconds))
                    return True
                return False
        except Exception as e:
            logger.warning(f"Redis expire 失败: {e}")
            return False

    def generate_key_hash(self, *args: Any) -> str:
        """生成参数的哈希值作为键的一部分"""
        content = json.dumps(args, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]


# ====== 全局实例 ======

cache_service = CacheService()

# ====== 缓存键命名规范 ======

# chat:history:{session_id}     → 对话历史
# search:{query_hash}            → 检索结果缓存
# rag:prompt:{hash}              → Prompt 缓存
# llm:response:{hash}            → LLM 响应缓存
# agent:state:{session_id}       → Agent 状态缓存
