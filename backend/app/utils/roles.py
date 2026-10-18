# backend/app/utils/roles.py
"""
角色名归一工具（单一判定入口）

背景（真实踩坑）：
- 存储层 `chat_history_service._map_role()` 把 `user/assistant` 映射成 `human/ai` 落库；
- 但消费层（拼 LangChain messages / 渲染历史对话 / 路由历史摘要）多处写成
  `role == "user"` / `role == "assistant"`，与存储层的 `human/ai` 不匹配，
  结果**整段多轮历史被静默丢弃**（不报错、日志也看不出来）。

修法：
1. 所有消费端不再各写字符串比较，一律调用本模块的判定函数；
2. `ChatHistoryService.get_history()` 在出口处统一归一化为 `user/assistant`，
   形成"存储层一套、读取层一套"的显式边界。

这样即使将来有人再改存储命名，也不会重演静默失效。
"""

from typing import Any

# 用户侧角色的所有历史写法
USER_ROLES: frozenset[str] = frozenset({"user", "human", "客户", "用户"})
# 助手侧角色的所有历史写法
ASSISTANT_ROLES: frozenset[str] = frozenset({"assistant", "ai", "bot", "助手"})
# 工具/系统等其它角色
OTHER_ROLES: frozenset[str] = frozenset({"tool", "system", "function"})


def _norm(role: Any) -> str:
    """把任意 role 归一为小写去空白的字符串，None → ''。"""
    return str(role or "").strip().lower()


def is_user_role(role: Any) -> bool:
    """是否用户侧消息（兼容 user / human，忽略大小写与空白）。"""
    return _norm(role) in USER_ROLES


def is_assistant_role(role: Any) -> bool:
    """是否助手侧消息（兼容 assistant / ai / bot，忽略大小写与空白）。"""
    return _norm(role) in ASSISTANT_ROLES


def canonical_role(role: Any) -> str:
    """
    归一化为规范角色名：user / assistant / tool / system。

    未知角色原样返回（小写去空白），便于排查。
    """
    low = _norm(role)
    if low in USER_ROLES:
        return "user"
    if low in ASSISTANT_ROLES:
        return "assistant"
    if low in OTHER_ROLES:
        return low
    return low


def normalize_history(messages: list[dict] | None) -> list[dict]:
    """
    把历史消息列表的角色字段就地归一化为 user/assistant。

    只处理已知角色，未知角色保留原值；不改动 content / timestamp 等其它字段。
    """
    if not messages:
        return []
    normalized: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        item = dict(msg)
        item["role"] = canonical_role(msg.get("role"))
        normalized.append(item)
    return normalized


def role_label_cn(role: Any) -> str:
    """渲染中文角色前缀（用于拼提示词时的历史对话展示）。"""
    if is_user_role(role):
        return "用户"
    if is_assistant_role(role):
        return "助手"
    return str(role or "未知")
