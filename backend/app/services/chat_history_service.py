# backend/app/services/chat_history_service.py
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.logger import logger
from backend.app.core.cache import cache_service
from backend.app.models.chat_history import ChatMessage, ChatSession, TaskMessage
from backend.app.utils.roles import canonical_role, normalize_history


class ChatHistoryService:
    """
    对话历史管理服务
    Redis: 快速读写，做缓存
    MySQL: 持久化存储，做备份
    """

    def __init__(
        self,
        max_history_limit: int = 20,
        redis_expire: int = 60 * 60 * 24 * 7,
    ):
        self.max_history_limit = max_history_limit
        self.redis_expire = redis_expire

    def _key(self, session_id: str) -> str:
        """生成Redis Key"""
        return f"chat:history:{session_id}"

    @staticmethod
    def _map_role(role: str) -> str:
        """
        【写入层】兼容旧的 role 命名: user/assistant → human/ai

        注意：这是**存储层**的命名约定（历史遗留）。读取侧请统一走
        `get_history()`（出口已归一化为 user/assistant），消费端判定角色时
        请使用 `backend.app.utils.roles.is_user_role / is_assistant_role`，
        不要再自己写 `role == "user"`，否则会重演"多轮历史静默失效"。
        """
        mapping = {
            "user": "human",
            "assistant": "ai",
            "human": "human",
            "ai": "ai",
            "tool": "tool",
        }
        return mapping.get(canonical_role(role), role)

    @staticmethod
    def _map_event_to_step_type(event: dict[str, Any]) -> str | None:
        """把流式事件类型映射到 task_message.step_type"""
        event_type = event.get("type", "")
        phase = event.get("phase", "")

        if event_type == "tool_result":
            return "observation"
        if event_type == "answer":
            return "answer"
        if event_type == "error":
            return "error"
        if event_type == "thinking":
            if phase == "thought":
                return "thought"
            if phase == "action":
                return "action"
            if phase == "final":
                return "answer"
        return None

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        title: str | None = None,
        db: AsyncSession | None = None,
    ) -> None:
        """
        创建或获取会话
        如果会话已存在则直接返回
        """
        if db is None:
            return

        try:
            result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                return

            session = ChatSession(
                id=session_id,
                user_id=user_id,
                agent_id=agent_id or "",
                title=title or "",
                status="active",
                message_count=0,
                total_token_usage=0,
            )
            db.add(session)
            await db.commit()
            logger.info(f"创建会话: {session_id}, agent_id={agent_id}")
        except Exception as e:
            await db.rollback()
            logger.error(f"创建会话失败: {e}")

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_name: str | None = None,
        token_usage: float | None = None,
        db: AsyncSession | None = None,
    ) -> None:
        """
        保存一条基础消息到 Redis 和 MySQL。
        注意：此方法不会自动 commit，需要调用方负责事务提交。
        """
        mapped_role = self._map_role(role)
        message = {
            "role": mapped_role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        await self._save_to_redis(session_id, message)

        if db is not None:
            try:
                from backend.app.services.auth import generate_id

                record = ChatMessage(
                    id=generate_id(),
                    session_id=session_id,
                    role=mapped_role,
                    content=content,
                    model_name=model_name,
                    token_usage=token_usage or 0,
                )
                db.add(record)
            except Exception as e:
                logger.error(f"保存消息失败: {e}")

    async def save_user_message_only(
        self,
        session_id: str,
        user_message: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        model_name: str | None = None,
        db: AsyncSession | None = None,
    ) -> None:
        """
        提前保存用户问题（流式生成开始前调用）。
        作用：即使客户端在流式生成中途断开连接 / 刷新页面，
        历史记录中至少已保留用户问题，便于回来后恢复或重新生成。
        注意：本方法只写用户消息，助手消息由末尾的 save_exchange(skip_user=True) 写入。
        """
        try:
            await self.save_message(
                session_id, "human", user_message, model_name=model_name, token_usage=0, db=db
            )
            if db is not None:
                await db.commit()
            logger.info(f"提前保存用户消息: session_id={session_id}")
        except Exception as e:
            if db is not None:
                await db.rollback()
            logger.error(f"提前保存用户消息失败: {e}")

    async def _update_session_stats(
        self,
        session_id: str,
        token_usage: float | None = None,
        db: AsyncSession | None = None,
    ) -> None:
        """更新会话统计信息"""
        if db is not None:
            try:
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(
                        message_count=ChatSession.message_count + 1,
                        total_token_usage=ChatSession.total_token_usage + (token_usage or 0),
                        update_time=datetime.now(),
                    )
                )
            except Exception as e:
                logger.error(f"更新会话统计失败: {e}")

    async def _save_to_redis(self, session_id: str, message: dict) -> None:
        """写入 Redis"""
        key = self._key(session_id)
        await cache_service.rpush(key, message)
        await cache_service.expire(key, self.redis_expire)

        current_len = await cache_service.llen(key)
        if current_len > self.max_history_limit:
            await cache_service.ltrim(key, -self.max_history_limit, -1)

    async def _save_task_messages(
        self,
        session_id: str,
        message_id: str,
        thinking_steps: str | list[dict[str, Any]],
        model_name: str | None,
        db: AsyncSession,
    ) -> None:
        """把思考过程事件拆分保存到 task_message 表"""
        if isinstance(thinking_steps, str):
            try:
                steps = json.loads(thinking_steps)
            except json.JSONDecodeError:
                logger.warning("thinking_steps JSON 解析失败，跳过保存 task_message")
                return
        else:
            steps = thinking_steps

        if not isinstance(steps, list):
            return

        from backend.app.services.auth import generate_id

        for event in steps:
            if not isinstance(event, dict):
                continue

            step_type = self._map_event_to_step_type(event)
            if step_type is None:
                continue

            step = event.get("step", 0) or 0
            content = event.get("content", "") or ""
            latency_ms = event.get("latency_ms", 0) or 0
            token_count = event.get("token_count", 0) or 0

            if step_type == "action":
                tools = event.get("tools") or []
                if not tools:
                    task_msg = TaskMessage(
                        id=generate_id(),
                        session_id=session_id,
                        message_id=message_id,
                        step=step,
                        step_type="action",
                        content=content,
                        model_name=model_name,
                        token_count=token_count,
                        latency_ms=latency_ms,
                    )
                    db.add(task_msg)
                else:
                    for i, tool in enumerate(tools):
                        name = tool.get("name", "") if isinstance(tool, dict) else str(tool)
                        args = tool.get("args", {}) if isinstance(tool, dict) else {}
                        tool_input = json.dumps(args, ensure_ascii=False) if args else None
                        task_msg = TaskMessage(
                            id=generate_id(),
                            session_id=session_id,
                            message_id=message_id,
                            step=step + i,
                            step_type="action",
                            content=f"调用 {name}" if len(tools) > 1 else content,
                            tool_name=name,
                            tool_input=tool_input,
                            model_name=model_name,
                            token_count=token_count,
                            latency_ms=latency_ms,
                        )
                        db.add(task_msg)
            elif step_type == "observation":
                task_msg = TaskMessage(
                    id=generate_id(),
                    session_id=session_id,
                    message_id=message_id,
                    step=step,
                    step_type="observation",
                    content=content,
                    tool_name=event.get("tool", ""),
                    model_name=model_name,
                    token_count=token_count,
                    latency_ms=latency_ms,
                )
                db.add(task_msg)
            else:
                task_msg = TaskMessage(
                    id=generate_id(),
                    session_id=session_id,
                    message_id=message_id,
                    step=step,
                    step_type=step_type,
                    content=content,
                    model_name=model_name,
                    token_count=token_count,
                    latency_ms=latency_ms,
                )
                db.add(task_msg)

    async def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        model_name: str | None = None,
        token_usage: float | None = None,
        thinking_steps: str | list[dict[str, Any]] | None = None,
        db: AsyncSession | None = None,
        skip_user: bool = False,
    ) -> None:
        """
        保存一轮对话（在同一个事务中提交）

        Args:
            thinking_steps: AI回复关联的思考过程，支持 JSON 字符串或事件列表。
                            会被拆分保存到 task_message 表。
            skip_user: 为 True 时只保存助手消息（用户消息已由 save_user_message_only 提前保存），
                       避免重复写入用户消息
        """
        try:
            if not skip_user:
                await self.save_message(
                    session_id, "human", user_message, model_name=model_name, token_usage=0, db=db
                )

            from backend.app.services.auth import generate_id

            assistant_id = generate_id()
            if db is not None:
                assistant_record = ChatMessage(
                    id=assistant_id,
                    session_id=session_id,
                    role="ai",
                    content=assistant_message,
                    model_name=model_name,
                    token_usage=token_usage or 0,
                )
                db.add(assistant_record)

                if thinking_steps:
                    await self._save_task_messages(
                        session_id=session_id,
                        message_id=assistant_id,
                        thinking_steps=thinking_steps,
                        model_name=model_name,
                        db=db,
                    )

            # 同步写入 Redis（包含 thinking_steps，便于历史快速恢复）
            assistant_redis_msg = {
                "role": "ai",
                "content": assistant_message,
                "timestamp": datetime.now().isoformat(),
            }
            if thinking_steps:
                if isinstance(thinking_steps, str):
                    assistant_redis_msg["thinking_steps"] = thinking_steps
                else:
                    assistant_redis_msg["thinking_steps"] = json.dumps(
                        thinking_steps, ensure_ascii=False
                    )
            await self._save_to_redis(session_id, assistant_redis_msg)

            await self._update_session_stats(session_id, token_usage=token_usage, db=db)

            if db is not None:
                await db.commit()
                logger.info(
                    f"保存对话成功: session_id={session_id}, token_usage={token_usage}, has_thinking={bool(thinking_steps)}"
                )
        except Exception as e:
            if db is not None:
                await db.rollback()
            logger.error(f"保存对话失败: {e}")

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
        db: AsyncSession | None = None,
    ) -> list[dict]:
        """
        查询历史：Redis优先，Redis没有则查MySQL并回填

        【读取层出口契约】返回的每条消息 role 恒为 `user` / `assistant`（规范名），
        与写入层的 `human` / `ai` 解耦。所有拼 prompt / 渲染历史 / 路由摘要的
        消费端都应基于本契约判断角色，避免再次出现"角色名对不上导致历史被丢弃"。
        """
        messages = await self._get_from_redis(session_id, limit)
        if not messages:
            if db is not None:
                messages = await self._get_from_mysql(session_id, limit, db)
                if messages:
                    await self._backfill_redis(session_id, messages)

        # 出口归一化：human/ai → user/assistant（未知角色原样保留）
        return normalize_history(messages)

    async def _get_from_redis(self, session_id: str, limit: int | None) -> list[dict]:
        """从 Redis 读取"""
        key = self._key(session_id)
        if not await cache_service.exists(key):
            return []

        if limit:
            raw = await cache_service.lrange(key, -limit, -1)
        else:
            raw = await cache_service.lrange(key, 0, -1)

        return raw

    @staticmethod
    def build_thinking_steps(task_records: list[TaskMessage]) -> list[dict[str, Any]]:
        """把 task_message 记录组装回前端可识别的 thinking_steps 事件列表"""
        steps = []
        # 连续相同 step 的 action 记录需要合并成一个 action 事件
        i = 0
        while i < len(task_records):
            tr = task_records[i]
            if tr.step_type == "action":
                tools = []
                j = i
                while j < len(task_records) and task_records[j].step_type == "action":
                    tool_name = task_records[j].tool_name or ""
                    tool_input = {}
                    if task_records[j].tool_input:
                        try:
                            tool_input = json.loads(task_records[j].tool_input)
                        except json.JSONDecodeError:
                            tool_input = {"raw": task_records[j].tool_input}
                    if tool_name:
                        tools.append({"name": tool_name, "args": tool_input})
                    j += 1

                content = tr.content
                if len(tools) > 1 and not content.startswith("调用 "):
                    content = "；".join(f"调用 {t['name']}" for t in tools)

                steps.append(
                    {
                        "type": "thinking",
                        "step": tr.step,
                        "content": content,
                        "tools": tools,
                        "phase": "action",
                    }
                )
                i = j
            elif tr.step_type == "observation":
                steps.append(
                    {
                        "type": "tool_result",
                        "step": tr.step,
                        "tool": tr.tool_name or "",
                        "content": tr.content or "",
                        "phase": "observation",
                    }
                )
                i += 1
            elif tr.step_type == "thought":
                steps.append(
                    {
                        "type": "thinking",
                        "step": tr.step,
                        "content": tr.content or "",
                        "tools": [],
                        "phase": "thought",
                    }
                )
                i += 1
            elif tr.step_type == "answer":
                steps.append(
                    {
                        "type": "thinking",
                        "step": tr.step,
                        "content": tr.content or "",
                        "tools": [],
                        "phase": "final",
                    }
                )
                i += 1
            elif tr.step_type == "error":
                steps.append(
                    {
                        "type": "error",
                        "step": tr.step,
                        "content": tr.content or "",
                    }
                )
                i += 1
            else:
                i += 1
        return steps

    async def _get_from_mysql(
        self, session_id: str, limit: int | None, db: AsyncSession
    ) -> list[dict]:
        """从 MySQL 读取"""
        stmt = (
            select(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.create_time.asc())
        )

        if limit:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        records = result.scalars().all()

        if not records:
            return []

        # 批量查询关联的 task_message（仅助手消息需要）
        message_ids = [r.id for r in records if r.role == "ai"]
        task_map: dict[str, list[TaskMessage]] = {}
        if message_ids:
            task_stmt = (
                select(TaskMessage)
                .filter(TaskMessage.message_id.in_(message_ids))
                .order_by(TaskMessage.step.asc())
            )
            task_result = await db.execute(task_stmt)
            for tr in task_result.scalars().all():
                task_map.setdefault(tr.message_id, []).append(tr)

        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "model_name": r.model_name,
                "timestamp": str(r.create_time),
                "thinking_steps": (
                    json.dumps(
                        self.build_thinking_steps(task_map.get(r.id, [])),
                        ensure_ascii=False,
                    )
                    if r.id in task_map
                    else None
                ),
            }
            for r in records
        ]

    async def _backfill_redis(self, session_id: str, messages: list[dict]) -> None:
        """MySQL数据回填到Redis"""
        key = self._key(session_id)
        for msg in messages:
            await cache_service.rpush(key, msg)
        await cache_service.expire(key, self.redis_expire)

    async def get_last_n_exchanges(
        self,
        session_id: str,
        n: int = 5,
        db: AsyncSession | None = None,
    ) -> list[dict]:
        """获取最近N轮对话"""
        return await self.get_history(session_id, limit=n * 2, db=db)

    async def evict_session_cache(self, session_id: str) -> None:
        """仅清除会话的 Redis 热缓存（不动 MySQL 记录）。

        用于会话被软删除（如所属 Agent 被删除）等场景：MySQL 中保留记录但状态标记为
        deleted，历史接口不再放行；Redis 热缓存若不清除，软删除会话仍可能被命中返回。
        """
        await cache_service.delete(self._key(session_id))

    async def clear_history(self, session_id: str, db: AsyncSession | None = None) -> bool:
        """清空指定会话"""
        await cache_service.delete(self._key(session_id))
        if db is not None:
            try:
                # 先删子表 task_message，再删父表 chat_message
                stmt_task = select(TaskMessage).filter(TaskMessage.session_id == session_id)
                result_task = await db.execute(stmt_task)
                for record in result_task.scalars().all():
                    await db.delete(record)

                stmt = select(ChatMessage).filter(ChatMessage.session_id == session_id)
                result = await db.execute(stmt)
                for record in result.scalars().all():
                    await db.delete(record)
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"MySQL删除失败: {e}", exc_info=True)
        return True

    async def session_exists(self, session_id: str) -> bool:
        """判断会话是否存在"""
        return await cache_service.exists(self._key(session_id))

    async def get_session_count(self) -> int:
        """获取会话总数"""
        keys = await cache_service.keys("chat:history:*")
        return len(keys)

    async def get_all_session_ids(self) -> list[str]:
        """获取所有会话ID"""
        keys = await cache_service.keys("chat:history:*")
        return [key.replace("chat:history:", "") for key in keys]

    async def get_history_length(self, session_id: str) -> int:
        """获取某会话的消息条数"""
        return await cache_service.llen(self._key(session_id))


chat_history_service = ChatHistoryService()
