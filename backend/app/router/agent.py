# backend/app/router/agent.py
import json
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.exceptions import BusinessException
from backend.app.core.security import get_current_user
from backend.app.models.agent import Agent, AgentKnowledgeBase, AgentTool
from backend.app.models.chat_history import ChatSession
from backend.app.models.tool import Tool
from backend.app.services.chat_history_service import ChatHistoryService
from backend.app.templates import AGENT_TEMPLATES
from backend.db.db_config import get_session

router = APIRouter()


# ====== 请求模型 ======


class CreateAgentReq(BaseModel):
    name: str
    avatar: str | None = None
    desc: str | None = ""
    system_prompt: str | None = ""
    temperature: float | None = 0.7
    tool_ids: list[str] | None = []  # 所有工具ID（系统工具和自定义工具统一管理）
    kb_ids: list[str] | None = []
    greeting: str | None = ""
    tool_configs: dict[str, Any] | None = None  # 工具级配置, key=工具名(如 knowledge_search)


class UpdateAgentReq(BaseModel):
    name: str | None = None
    avatar: str | None = None
    desc: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    tool_ids: list[str] | None = None
    kb_ids: list[str] | None = None
    greeting: str | None = None
    tool_configs: dict[str, Any] | None = None


class CreateToolReq(BaseModel):
    name: str
    desc: str | None = ""
    icon: str | None = "🧩"
    tool_type: str | None = "custom"
    code_path: str | None = ""
    parameters_schema: str | None = "{}"
    visibility: str | None = "public"  # public=全员可见 / private=仅创建者


# ====== 辅助函数 ======


async def _get_agent_tools(agent_id: str, db: AsyncSession) -> list:
    """获取Agent关联的所有工具ID列表"""
    result = await db.execute(select(AgentTool).where(AgentTool.agent_id == agent_id))
    agent_tools = result.scalars().all()
    return [t.tool_id for t in agent_tools]


async def _get_agent_tools_detail(agent_id: str, db: AsyncSession) -> list:
    """获取Agent关联的所有工具详情（系统工具+自定义工具）"""
    # 获取Agent对象
    agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = agent_result.scalar_one_or_none()
    if not agent:
        return []

    # 获取自定义工具详情（从 AgentTool 关联表）
    result = await db.execute(select(AgentTool).where(AgentTool.agent_id == agent_id))
    agent_tools = result.scalars().all()

    custom_tool_details = []
    if agent_tools:
        tool_ids = [t.tool_id for t in agent_tools]
        tools_result = await db.execute(select(Tool).where(Tool.id.in_(tool_ids)))
        for t in tools_result.scalars().all():
            custom_tool_details.append({"id": t.id, "name": t.name, "desc": t.desc, "icon": t.icon})

    # 获取系统工具详情（从 system_tool_names 字段的 code_path）
    system_tool_details = []
    system_tool_code_paths = agent.get_system_tools()
    if system_tool_code_paths:
        tools_result = await db.execute(
            select(Tool).where(
                Tool.code_path.in_(system_tool_code_paths), Tool.tool_type == "system"
            )
        )
        for t in tools_result.scalars().all():
            system_tool_details.append({"id": t.id, "name": t.name, "desc": t.desc, "icon": t.icon})

    # 合并返回
    return system_tool_details + custom_tool_details


async def _get_agent_kbs(agent_id: str, db: AsyncSession) -> list:
    """获取Agent关联的知识库ID列表"""
    result = await db.execute(
        select(AgentKnowledgeBase).where(AgentKnowledgeBase.agent_id == agent_id)
    )
    agent_kbs = result.scalars().all()
    return [k.kb_id for k in agent_kbs]


async def _get_agent_kbs_detail(agent_id: str, db: AsyncSession) -> list:
    """获取Agent关联的知识库详情"""
    result = await db.execute(
        select(AgentKnowledgeBase).where(AgentKnowledgeBase.agent_id == agent_id)
    )
    agent_kbs = result.scalars().all()
    if not agent_kbs:
        return []

    kb_ids = [k.kb_id for k in agent_kbs]
    from backend.app.models.knowledge import KnowledgeBase

    kbs_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids)))
    kbs = kbs_result.scalars().all()
    return [{"id": kb.id, "name": kb.name} for kb in kbs]


# ====== 系统工具初始化 ======


def _generate_tool_id() -> str:
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(12))


SYSTEM_TOOLS = [
    {
        "name": "计算器",
        "desc": "用于计算各种数学表达式",
        "icon": "🧮",
        "tool_type": "system",
        "code_path": "calculator",
        "parameters_schema": '{"type":"object","properties":{"expression":{"type":"string","description":"数学表达式"}},"required":["expression"]}',
    },
    {
        "name": "文件读取",
        "desc": "读取文件内容，支持txt、md等格式",
        "icon": "📄",
        "tool_type": "system",
        "code_path": "file_reader",
        "parameters_schema": '{"type":"object","properties":{"file_path":{"type":"string","description":"文件路径"}},"required":["file_path"]}',
    },
    {
        "name": "知识库查询",
        "desc": "从知识库中检索相关信息",
        "icon": "🔍",
        "tool_type": "system",
        "code_path": "knowledge_search",
        "parameters_schema": '{"type":"object","properties":{"query":{"type":"string","description":"查询内容"},"required":["query"]}',
    },
    {
        "name": "报销OA连接器",
        "desc": "对接企业 OA 系统：提交报销单、查询审批状态、查询我的待办报销单（Mock 版，部署时替换为真实连接器）",
        "icon": "💼",
        "tool_type": "system",
        "code_path": "oa_reimbursement",
        "parameters_schema": '{"type":"object","properties":{"action":{"type":"string","description":"操作类型：submit=提交报销单；query_status=查询审批状态；list_mine=查询我的待办报销单"},"params":{"type":"string","description":"JSON 字符串参数。submit 示例：{\\"applicant\\":\\"张三\\",\\"items\\":[{\\"category\\":\\"差旅\\",\\"amount\\":1200}],\\"total\\":1200}；query_status 示例：{\\"ticket_no\\":\\"BX20260001\\"}；list_mine 示例：{\\"applicant\\":\\"张三\\"}"}},"required":["action"]}',
    },
]


async def ensure_system_tools(db: AsyncSession):
    """确保系统工具存在于tool表中"""
    for tool_data in SYSTEM_TOOLS:
        result = await db.execute(
            select(Tool).where(Tool.name == tool_data["name"], Tool.tool_type == "system")
        )
        existing = result.scalar_one_or_none()
        if existing:
            # 如果被软删除，重新激活
            if existing.status != "active":
                existing.status = "active"
                db.add(existing)
        else:
            tool = Tool(
                id=_generate_tool_id(),
                name=tool_data["name"],
                desc=tool_data["desc"],
                icon=tool_data["icon"],
                tool_type=tool_data["tool_type"],
                code_path=tool_data["code_path"],
                parameters_schema=tool_data["parameters_schema"],
                owner_id=None,
            )
            db.add(tool)
    await db.commit()


# ====== Agent 管理接口 ======


@router.get("/agents", summary="获取我的Agent列表")
async def get_my_agents(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前用户的Agent列表
    """
    query = select(Agent).where(Agent.owner_id == current_user, Agent.status == "active")
    result = await db.execute(query.order_by(Agent.create_time.desc()))
    agents = result.scalars().all()

    agent_list = []
    for agent in agents:
        tools = await _get_agent_tools_detail(agent.id, db)
        kbs = await _get_agent_kbs_detail(agent.id, db)

        agent_list.append(
            {
                "id": agent.id,
                "name": agent.name,
                "avatar": agent.avatar,
                "desc": agent.desc,
                "system_prompt": agent.system_prompt,
                "temperature": agent.temperature,
                "tools": tools,
                "knowledge_bases": kbs,
                "greeting": agent.greeting or "",
                "tool_configs": agent.get_tool_configs(),
                "create_time": (
                    agent.create_time.strftime("%Y-%m-%d %H:%M:%S") if agent.create_time else None
                ),
            }
        )

    return {"agents": agent_list}


@router.post("/agents", summary="创建Agent")
async def create_agent(
    req: CreateAgentReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    创建Agent并关联工具和知识库

    工具处理逻辑：
    - 系统工具（tool_type=system）：将 code_path 保存到 Agent.system_tool_names 字段
    - 自定义工具（tool_type=custom）：将 tool_id 保存到 AgentTool 关联表
    """
    import json

    from backend.app.services.auth import generate_id

    # 检查同名Agent
    existing = await db.execute(
        select(Agent).where(
            Agent.name == req.name, Agent.owner_id == current_user, Agent.status == "active"
        )
    )
    if existing.scalar_one_or_none():
        raise BusinessException(code=400, msg="已存在同名Agent")

    # 区分系统工具和自定义工具
    system_tool_code_paths = []  # 系统工具的 code_path 列表
    custom_tool_ids = []  # 自定义工具的 ID 列表

    if req.tool_ids:
        # 查询工具类型
        tools_result = await db.execute(select(Tool).where(Tool.id.in_(req.tool_ids)))
        tools = tools_result.scalars().all()

        for tool in tools:
            if tool.tool_type == "system":
                system_tool_code_paths.append(tool.code_path or tool.name)
            else:
                custom_tool_ids.append(tool.id)

    # 创建Agent
    agent = Agent(
        id=generate_id(),
        name=req.name,
        avatar=req.avatar,
        desc=req.desc,
        owner_id=current_user,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        greeting=req.greeting or "",
        system_tool_names=json.dumps(system_tool_code_paths),  # 保存系统工具的 code_path
        tool_configs=json.dumps(req.tool_configs, ensure_ascii=False) if req.tool_configs else "{}",
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    # 关联自定义工具（系统工具已通过 system_tool_names 保存）
    for tool_id in custom_tool_ids:
        agent_tool = AgentTool(
            id=generate_id(),
            agent_id=agent.id,
            tool_id=tool_id,
        )
        db.add(agent_tool)

    # 关联知识库
    for kb_id in req.kb_ids or []:
        agent_kb = AgentKnowledgeBase(
            id=generate_id(),
            agent_id=agent.id,
            kb_id=kb_id,
        )
        db.add(agent_kb)

    await db.commit()

    return {"id": agent.id, "name": agent.name, "message": "Agent创建成功"}


@router.get("/agents/{agent_id}", summary="获取Agent详情")
async def get_agent_detail(
    agent_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取指定Agent的详情

    返回的 tool_ids 包含系统工具和自定义工具的ID
    """
    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.owner_id == current_user)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise BusinessException(code=404, msg="Agent不存在")

    # 获取自定义工具ID（从 AgentTool 关联表）
    custom_tool_ids = await _get_agent_tools(agent.id, db)

    # 获取系统工具ID（从 system_tool_names 字段获取 code_path，再查找 Tool 记录）
    system_tool_ids = []
    system_tool_code_paths = agent.get_system_tools()
    if system_tool_code_paths:
        # 查找 code_path 对应的 Tool 记录
        tools_result = await db.execute(
            select(Tool).where(
                Tool.code_path.in_(system_tool_code_paths), Tool.tool_type == "system"
            )
        )
        for tool in tools_result.scalars().all():
            system_tool_ids.append(tool.id)

    # 合并所有工具ID
    all_tool_ids = system_tool_ids + custom_tool_ids

    kb_ids = await _get_agent_kbs(agent_id, db)

    return {
        "id": agent.id,
        "name": agent.name,
        "avatar": agent.avatar,
        "desc": agent.desc,
        "system_prompt": agent.system_prompt,
        "temperature": agent.temperature,
        "tool_ids": all_tool_ids,
        "kb_ids": kb_ids,
        "greeting": agent.greeting or "",
        "tool_configs": agent.get_tool_configs(),
        "create_time": (
            agent.create_time.strftime("%Y-%m-%d %H:%M:%S") if agent.create_time else None
        ),
    }


@router.put("/agents/{agent_id}", summary="更新Agent")
async def update_agent(
    agent_id: str,
    req: UpdateAgentReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    更新Agent信息
    """
    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.owner_id == current_user)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise BusinessException(code=404, msg="Agent不存在")

    # 更新基本信息
    if req.name is not None:
        agent.name = req.name
    if req.avatar is not None:
        agent.avatar = req.avatar
    if req.desc is not None:
        agent.desc = req.desc
    if req.system_prompt is not None:
        agent.system_prompt = req.system_prompt
    if req.temperature is not None:
        agent.temperature = req.temperature
    if req.greeting is not None:
        agent.greeting = req.greeting

    # 更新工具级配置（如果传了tool_configs）
    if req.tool_configs is not None:
        agent.tool_configs = json.dumps(req.tool_configs, ensure_ascii=False)

    # 更新工具关联（如果传了tool_ids）
    if req.tool_ids is not None:
        # 查询所有工具类型
        tools_result = await db.execute(select(Tool).where(Tool.id.in_(req.tool_ids)))
        tools = tools_result.scalars().all()

        # 区分系统工具和自定义工具
        system_tool_code_paths = []
        custom_tool_ids = []

        for tool in tools:
            if tool.tool_type == "system":
                system_tool_code_paths.append(tool.code_path or tool.name)
            else:
                custom_tool_ids.append(tool.id)

        # 更新系统工具（保存 code_path 到 system_tool_names）
        agent.system_tool_names = json.dumps(system_tool_code_paths)

        # 删除旧的自定义工具关联
        await db.execute(delete(AgentTool).where(AgentTool.agent_id == agent_id))
        # 添加新的自定义工具关联
        from backend.app.services.auth import generate_id

        for tool_id in custom_tool_ids:
            agent_tool = AgentTool(
                id=generate_id(),
                agent_id=agent_id,
                tool_id=tool_id,
            )
            db.add(agent_tool)

    # 更新知识库关联（如果传了kb_ids）
    if req.kb_ids is not None:
        await db.execute(delete(AgentKnowledgeBase).where(AgentKnowledgeBase.agent_id == agent_id))
        from backend.app.services.auth import generate_id

        for kb_id in req.kb_ids:
            agent_kb = AgentKnowledgeBase(
                id=generate_id(),
                agent_id=agent_id,
                kb_id=kb_id,
            )
            db.add(agent_kb)

    await db.commit()
    return {"message": "更新成功"}


@router.delete("/agents/{agent_id}", summary="删除Agent")
async def delete_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除Agent（软删除），并级联软删除该 Agent 的所有关联会话：

    - agent.status -> "deleted"：Agent 不再出现在列表/详情，也无法被选中对话；
    - chat_session.status -> "deleted"（仅 soft，消息与轨迹行保留，可审计/可恢复）：
      /history/{session_id} 对 deleted 会话一律返回空，软删除会话对用户不可见；
    - 同步清除这些会话的 Redis 热缓存，避免缓存命中绕过软删除标记。
    """
    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.owner_id == current_user)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise BusinessException(code=404, msg="Agent不存在")

    agent.status = "deleted"

    # 级联软删除关联会话（含未 active 的历史会话），并清掉 Redis 热缓存
    sessions_result = await db.execute(
        select(ChatSession).where(ChatSession.agent_id == agent_id, ChatSession.status != "deleted")
    )
    sessions = sessions_result.scalars().all()
    history_service = ChatHistoryService()
    for session in sessions:
        session.status = "deleted"
        await history_service.evict_session_cache(session.id)

    await db.commit()
    return {
        "message": "删除成功",
        "deleted_sessions": len(sessions),
    }


@router.get("/templates", summary="获取场景模板列表")
async def get_agent_templates(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    返回可用的场景模板（报销助手、请假助手等）。
    模板 = 可克隆的 Agent 配置包（system_prompt / 工具集 / 检索参数 / 模型参数）。
    先确保系统工具（含 mock OA 连接器）已种入，避免前端按 code_path 解析不到工具。
    """
    await ensure_system_tools(db)
    return {"templates": AGENT_TEMPLATES}


# ====== 工具管理接口 ======


@router.get("/tools", summary="获取可用工具列表")
async def get_tools(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    tool_type: str | None = None,
):
    """
    获取可用工具列表（系统工具 + 用户自定义工具）
    tool_type: system / custom / None(全部)
    自定义工具根据 visibility 过滤：public=全员可见，private=仅创建者
    """
    await ensure_system_tools(db)

    # 自定义工具的可见性条件：public 全员可见，private 仅创建者
    _custom_visible = (Tool.tool_type == "custom") & (
        (Tool.visibility == "public") | (Tool.owner_id == current_user)
    )

    query = select(Tool).where(Tool.status == "active")

    if tool_type == "system":
        query = query.where(Tool.tool_type == "system")
    elif tool_type == "custom":
        query = query.where(_custom_visible)
    else:
        # 返回系统工具 + 当前用户可见的自定义工具
        query = query.where((Tool.tool_type == "system") | _custom_visible)

    result = await db.execute(query.order_by(Tool.tool_type, Tool.create_time.desc()))
    tools = result.scalars().all()

    return {
        "tools": [
            {
                "id": t.id,
                "name": t.name,
                "desc": t.desc,
                "icon": t.icon,
                "tool_type": t.tool_type,
                "code_path": t.code_path,
                "parameters_schema": t.parameters_schema,
                "visibility": t.visibility or "public",
            }
            for t in tools
        ]
    }


@router.post("/tools", summary="创建自定义工具")
async def create_tool(
    req: CreateToolReq,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    创建系统工具（仅管理员可操作）
    """
    # 权限校验：仅 admin / super_admin 可创建工具
    from backend.app.models.user import User as UserModel

    result = await db.execute(select(UserModel).where(UserModel.id == current_user))
    user_obj = result.scalar_one_or_none()
    if not user_obj or user_obj.role not in ("admin", "super_admin"):
        raise BusinessException(code=403, msg="仅管理员可创建工具")

    tool = Tool(
        name=req.name,
        desc=req.desc,
        icon=req.icon,
        tool_type=req.tool_type or "custom",
        code_path=req.code_path,
        parameters_schema=req.parameters_schema,
        owner_id=current_user,
        visibility=req.visibility or "public",
    )
    db.add(tool)
    await db.commit()
    await db.refresh(tool)

    return {"id": tool.id, "name": tool.name, "message": "工具创建成功"}


@router.get("/tools/my", summary="获取工具市场（用户可见的自定义工具）")
async def get_my_tools(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前用户可见的自定义工具列表（public 全员可见 + 自己创建的 private）
    用于"工具市场"展示
    """
    query = select(Tool).where(
        Tool.tool_type == "custom",
        Tool.status == "active",
        (Tool.visibility == "public") | (Tool.owner_id == current_user),
    )
    result = await db.execute(query.order_by(Tool.create_time.desc()))
    tools = result.scalars().all()

    return {
        "tools": [
            {
                "id": t.id,
                "name": t.name,
                "desc": t.desc,
                "icon": t.icon,
                "tool_type": t.tool_type,
                "code_path": t.code_path,
                "parameters_schema": t.parameters_schema,
                "visibility": t.visibility or "public",
                "owner_id": t.owner_id,
            }
            for t in tools
        ]
    }


@router.get("/tools/system", summary="获取系统工具列表")
async def get_system_tools(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取系统预置工具列表（所有用户可用）
    """
    await ensure_system_tools(db)

    query = select(Tool).where(Tool.tool_type == "system", Tool.status == "active")
    result = await db.execute(query.order_by(Tool.create_time.desc()))
    tools = result.scalars().all()

    return {
        "tools": [
            {
                "id": t.id,
                "name": t.name,
                "desc": t.desc,
                "icon": t.icon,
                "tool_type": t.tool_type,
                "code_path": t.code_path,
                "parameters_schema": t.parameters_schema,
            }
            for t in tools
        ]
    }


@router.delete("/tools/{tool_id}", summary="删除自定义工具")
async def delete_tool(
    tool_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除用户自定义工具
    """
    result = await db.execute(select(Tool).where(Tool.id == tool_id, Tool.owner_id == current_user))
    tool = result.scalar_one_or_none()
    if not tool:
        raise BusinessException(code=404, msg="工具不存在")

    tool.status = "deleted"
    await db.commit()
    return {"message": "删除成功"}


# ====== 知识库管理接口 ======


@router.get("/knowledge-bases", summary="获取我的知识库列表")
async def get_my_knowledge_bases(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前用户的知识库列表（包括"我的知识库"中的文件夹里的知识库）
    """
    from backend.app.models.knowledge import KbFolder, KbMember, KnowledgeBase

    # 获取用户直接关联的知识库
    member_query = select(KbMember.kb_id).where(
        KbMember.member_id == current_user,
        KbMember.member_type == "user",
        KbMember.status == "active",
    )
    member_result = await db.execute(member_query)
    member_kb_ids = [row[0] for row in member_result.fetchall()]

    # 获取用户作为所有者的知识库
    owner_query = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == current_user)
    owner_result = await db.execute(owner_query)
    owner_kb_ids = [row[0] for row in owner_result.fetchall()]

    # 获取"我的知识库"文件夹中的知识库
    kb_folder_query = select(KbFolder.id).where(
        KbFolder.user_id == current_user, KbFolder.name == "我的知识库"
    )
    kb_folder_result = await db.execute(kb_folder_query)
    my_kb_folder = kb_folder_result.scalar_one_or_none()

    folder_kb_ids = []
    if my_kb_folder:
        folder_kb_query = select(KnowledgeBase.id).where(KnowledgeBase.folder_id == my_kb_folder.id)
        folder_kb_result = await db.execute(folder_kb_query)
        folder_kb_ids = [row[0] for row in folder_kb_result.fetchall()]

    # 合并所有知识库ID并去重
    all_kb_ids = list(set(member_kb_ids + owner_kb_ids + folder_kb_ids))

    if not all_kb_ids:
        return {"knowledge_bases": []}

    # 获取知识库详情
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id.in_(all_kb_ids))
    kb_result = await db.execute(kb_query)
    knowledge_bases = kb_result.scalars().all()

    # 获取每个知识库的文档数量
    from sqlalchemy import func

    from backend.app.models.knowledge import DocumentInfo

    doc_counts = {}
    if knowledge_bases:
        kb_id_list = [kb.id for kb in knowledge_bases]
        doc_count_query = (
            select(DocumentInfo.kb_id, func.count(DocumentInfo.id))
            .where(DocumentInfo.kb_id.in_(kb_id_list), DocumentInfo.status == "completed")
            .group_by(DocumentInfo.kb_id)
        )
        doc_count_result = await db.execute(doc_count_query)
        for row in doc_count_result.fetchall():
            doc_counts[row[0]] = row[1]

    return {
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
                "desc": kb.desc or "",
                "doc_count": doc_counts.get(kb.id, 0),
                "folder_id": kb.folder_id,
            }
            for kb in knowledge_bases
        ]
    }


@router.get("/kb-tree", summary="获取知识库文件夹树结构")
async def get_kb_tree(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    folder_id: str | None = None,
):
    """
    获取指定文件夹下的子文件夹和知识库列表
    folder_id 为 None 时返回根目录内容
    """
    from sqlalchemy import func

    from backend.app.models.knowledge import DocumentInfo, KbFolder, KbMember, KnowledgeBase

    target_folder_id = folder_id

    # 获取当前文件夹信息（用于返回路径导航）
    current_folder = None
    path_parts = []

    if target_folder_id:
        folder_query = select(KbFolder).where(
            KbFolder.id == target_folder_id, KbFolder.user_id == current_user
        )
        folder_result = await db.execute(folder_query)
        current_folder = folder_result.scalar_one_or_none()

        if not current_folder:
            return {"error": "文件夹不存在"}, 404

        # 构建路径导航
        visited = set()
        temp = current_folder
        while temp:
            if temp.id in visited:
                break
            visited.add(temp.id)
            path_parts.insert(0, {"id": temp.id, "name": temp.name})
            if temp.parent_id:
                parent_query = select(KbFolder).where(KbFolder.id == temp.parent_id)
                parent_result = await db.execute(parent_query)
                temp = parent_result.scalar_one_or_none()
            else:
                break

    # 获取当前文件夹下的子文件夹
    if target_folder_id:
        subfolder_query = (
            select(KbFolder)
            .where(
                KbFolder.user_id == current_user,
                KbFolder.parent_id == target_folder_id,
                KbFolder.status == "active",
            )
            .order_by(KbFolder.sort_order)
        )
    else:
        # 根目录：parent_id 为 NULL 的子文件夹
        subfolder_query = (
            select(KbFolder)
            .where(
                KbFolder.user_id == current_user,
                KbFolder.parent_id.is_(None),
                KbFolder.status == "active",
            )
            .order_by(KbFolder.sort_order)
        )
    subfolder_result = await db.execute(subfolder_query)
    subfolders = subfolder_result.scalars().all()

    # 获取用户有权限的知识库ID列表
    member_query = select(KbMember.kb_id).where(
        KbMember.member_id == current_user,
        KbMember.member_type == "user",
        KbMember.status == "active",
    )
    member_result = await db.execute(member_query)
    member_kb_ids = set(row[0] for row in member_result.fetchall())

    owner_query = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == current_user)
    owner_result = await db.execute(owner_query)
    owner_kb_ids = set(row[0] for row in owner_result.fetchall())

    # 获取当前文件夹下的知识库
    if target_folder_id:
        kb_query = select(KnowledgeBase).where(
            KnowledgeBase.folder_id == target_folder_id, KnowledgeBase.status == "active"
        )
    else:
        # 根目录：folder_id 为 NULL 的知识库
        kb_query = select(KnowledgeBase).where(
            KnowledgeBase.folder_id.is_(None), KnowledgeBase.status == "active"
        )

    # 权限过滤：只有用户是 owner 或 member 的才可见
    kb_result = await db.execute(kb_query)
    all_kbs = kb_result.scalars().all()

    authorized_kbs = [
        kb
        for kb in all_kbs
        if kb.owner_id == current_user or kb.id in member_kb_ids or kb.id in owner_kb_ids
    ]

    # 获取文档数量
    doc_counts = {}
    if authorized_kbs:
        kb_id_list = [kb.id for kb in authorized_kbs]
        doc_count_query = (
            select(DocumentInfo.kb_id, func.count(DocumentInfo.id))
            .where(DocumentInfo.kb_id.in_(kb_id_list), DocumentInfo.status == "completed")
            .group_by(DocumentInfo.kb_id)
        )
        doc_count_result = await db.execute(doc_count_query)
        for row in doc_count_result.fetchall():
            doc_counts[row[0]] = row[1]

    # 获取子文件夹中的知识库数量（用于显示）
    folder_kb_counts = {}
    if subfolders:
        subfolder_ids = [f.id for f in subfolders]
        folder_kb_count_query = (
            select(KnowledgeBase.folder_id, func.count(KnowledgeBase.id))
            .where(KnowledgeBase.folder_id.in_(subfolder_ids), KnowledgeBase.status == "active")
            .group_by(KnowledgeBase.folder_id)
        )
        folder_kb_count_result = await db.execute(folder_kb_count_query)
        for row in folder_kb_count_result.fetchall():
            folder_kb_counts[row[0]] = row[1]

    return {
        "current_folder_id": target_folder_id,
        "current_folder_name": current_folder.name if current_folder else "根目录",
        "path": path_parts,
        "subfolders": [
            {
                "id": f.id,
                "name": f.name,
                "kb_count": folder_kb_counts.get(f.id, 0),
            }
            for f in subfolders
        ],
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
                "desc": kb.desc or "",
                "doc_count": doc_counts.get(kb.id, 0),
                "embedding_model": kb.embedding_model or "",
                "folder_id": kb.folder_id,
            }
            for kb in authorized_kbs
        ],
    }


@router.put("/knowledge-bases/{kb_id}/folder", summary="更新知识库所在文件夹")
async def update_kb_folder(
    kb_id: str,
    body: dict,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    更新知识库所在的文件夹
    body: {"folder_id": "xxx"} 或 {"folder_id": None} (移到根目录)
    """
    from backend.app.models.knowledge import KbFolder, KnowledgeBase

    folder_id = body.get("folder_id")

    # 验证知识库存在且属于当前用户
    kb_query = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == current_user
    )
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        return {"error": "知识库不存在或无权限"}, 404

    # 如果指定了文件夹，验证文件夹存在且属于当前用户
    if folder_id:
        folder_query = select(KbFolder).where(
            KbFolder.id == folder_id, KbFolder.user_id == current_user
        )
        folder_result = await db.execute(folder_query)
        folder = folder_result.scalar_one_or_none()

        if not folder:
            return {"error": "文件夹不存在"}, 404

    # 更新文件夹
    kb.folder_id = folder_id
    await db.commit()

    return {"message": "更新成功", "folder_id": folder_id}


@router.post("/kb-folders", summary="创建文件夹")
async def create_kb_folder(
    body: dict,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    创建知识库文件夹
    body: {"name": "文件夹名", "parent_id": "可选的父文件夹ID"}
    """
    import uuid

    from backend.app.models.knowledge import KbFolder

    name = body.get("name", "").strip()
    parent_id = body.get("parent_id")

    if not name:
        return {"error": "文件夹名称不能为空"}, 400

    folder_id = uuid.uuid4().hex[:12]

    new_folder = KbFolder(
        id=folder_id,
        user_id=current_user,
        parent_id=parent_id,
        name=name,
    )

    db.add(new_folder)
    await db.commit()

    return {
        "id": folder_id,
        "name": name,
        "parent_id": parent_id,
    }


@router.delete("/kb-folders/{folder_id}", summary="删除文件夹")
async def delete_kb_folder(
    folder_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    删除知识库文件夹（只能删除空文件夹或把里面的知识库移到根目录）
    """
    from backend.app.models.knowledge import KbFolder, KnowledgeBase

    # 验证文件夹存在且属于当前用户
    folder_query = select(KbFolder).where(
        KbFolder.id == folder_id, KbFolder.user_id == current_user
    )
    folder_result = await db.execute(folder_query)
    folder = folder_result.scalar_one_or_none()

    if not folder:
        return {"error": "文件夹不存在或无权限"}, 404

    # 检查文件夹下是否有子文件夹
    subfolder_query = select(KbFolder).where(
        KbFolder.parent_id == folder_id, KbFolder.status == "active"
    )
    subfolder_result = await db.execute(subfolder_query)
    subfolders = subfolder_result.scalars().all()

    if subfolders:
        return {"error": "文件夹下还有子文件夹，请先删除子文件夹"}, 400

    # 检查文件夹下是否有知识库
    kb_query = select(KnowledgeBase).where(
        KnowledgeBase.folder_id == folder_id, KnowledgeBase.status == "active"
    )
    kb_result = await db.execute(kb_query)
    kbs = kb_result.scalars().all()

    if kbs:
        # 将知识库移到根目录
        for kb in kbs:
            kb.folder_id = None

    # 删除文件夹（软删除）
    folder.status = "deleted"
    await db.commit()

    return {"message": "删除成功"}
