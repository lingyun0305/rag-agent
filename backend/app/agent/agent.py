# backend/app/agent/agent.py
"""
Agentic RAG 智能体模块
基于 ReAct 范式的 Agent：Think(Reason) → Act(Act) → Observe(Observation) 循环

事件流（run_stream 产出 JSON 字符串）：
- {"type": "thinking", "step": N, "content": "...", "tools": [...], "phase": "thought|action|final"}
- {"type": "tool_result", "step": N, "tool": "...", "content": "...", "phase": "observation"}
- {"type": "answer", "content": "最终答案"}
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent.agent_state import AgentState
from backend.app.agent.tools.base import ToolResult
from backend.app.agent.tools.registry import (
    init_default_tools,
    load_agent_base_tools,
    load_agent_tools,
    tool_registry,
)
from backend.app.common.logger import logger
from backend.app.core.config import settings
from backend.app.rag.query_rewrite_service import query_rewrite_service
from backend.app.utils.models_factory import model_factory
from backend.app.utils.roles import role_label_cn
from backend.app.utils.token_estimator import (
    cosine_similarity,
    estimate_tokens,
    similar_args,
)

# ====== 常量 ======
MAX_TOOL_RETRIES = 3  # 单次工具执行失败重试上限
MAX_REACT_ROUNDS = 8  # ReAct 循环硬性步数上限（达到后强制总结）
OBS_TRUNCATE_SOFT = 2000  # Observation 软截断阈值（>2000 字符保留头尾/关键词窗口）
OBS_EXTRACT_HARD = 3000  # Observation 硬提取阈值（>3000 字符调用 LLM 抽取核心信息）
LLM_EXTRACT_MAX_INPUT = 6000  # 送交 LLM 提取观测值时的最大输入长度


def _usage_get(usage: Any, key: str, default: Any = 0) -> Any:
    """从 usage_metadata 取值，兼容 dict 与对象两种形态（缺失返回 default）。"""
    if usage is None:
        return default
    if isinstance(usage, dict):
        value = usage.get(key, default)
    else:
        value = getattr(usage, key, default)
    return default if value is None else value


# ReAct 主循环生成时的停止序列：
# - 正常轮次：模型只应输出「一步」工具调用，遇到 </input> 即停止，防止它把多个
#   步骤（含伪造的 <observation>）一次性吐出导致循环；<observation> 同理拦截。
REACT_STOP_SEQUENCES = ["</input>", "<observation>"]

# 知识库检索无结果/无关时的统一拒绝模板（防幻觉，避免模型凭空作答）
KB_REJECT_TEMPLATE = "抱歉，我在知识库中未找到相关信息，无法提供答案。"

# 拒答/知识库未覆盖类关键词（用于软结论标注、历史对话降权，避免"前轮拒答"被当成硬事实锚定）
_REFUSAL_KW = (
    "无法提供",
    "无法回答",
    "未找到",
    "没有找到",
    "没有相关信息",
    "知识库中未",
    "不在企业知识库",
    "暂无相关信息",
    "不知道",
    "不存在",
    "未提及",
    "未查询到",
    "没有收录",
    "未收录",
    "暂不提供",
)


def _contains_refusal(text: str) -> bool:
    """判断文本是否含拒答/知识库未覆盖措辞。"""
    return any(kw in (text or "") for kw in _REFUSAL_KW)


# 「知识库未覆盖 + 常识推测」混合答案识别：模型一边承认知识库没有，一边用常识推测补全
# （如「公司食堂通常会提供午餐和晚餐，但具体菜单未在知识库中提及」→ 应整体替换为拒答模板）
_KB_GAP_ADMISSION_RE = re.compile(
    r"未在知识库|知识库中未|知识库未|未在文档|文档中未|未找到相关|未检索到|检索不到|未直接提及|未提及"
)
_SPECULATION_RE = re.compile(
    r"推测|猜测|通常是|通常会|一般是|一般会|可能会|应该是|大概是|估计|想来"
)


# "无需工具 / 停止"信号的归一化集合：模型输出这些 action 值时，直接视为强制总结（Final Answer）。
_NONE_ACTIONS = {
    "none",
    "无",
    "no",
    "n/a",
    "na",
    "finish",
    "done",
    "final",
    "stop",
    "结束",
    "停止",
    "不使用工具",
    "不需要工具",
    "无需调用工具",
    "不调用工具",
    "无需工具",
    "no_tool",
}
# rerank 相关度低于此值视为“无关”（仅作兜底；主依据是检索结果为空）。
KB_RELEVANCE_FLOOR = 0.15

# ====== Observation 关键词高亮（检索结果侧引导） ======
# 抽取 query 中的中文 2~3 字 n-gram 与拉丁词（过滤停用词），在 Observation 文本里
_HL_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_HL_LATIN_RE = re.compile(r"[A-Za-z0-9_./\-]+")
_HL_STOPWORDS = set(
    list(
        "的了一是在不和与或我你他她它这那们有个为人中也下对等到及其实该各些"
        "要会能可以如果因为所以通过进行如何怎么什么哪些是否问题系统公司"
        "使用需要提供相关信息数据请问一个没有我们你们他们你"
    )
    + [
        "可以",
        "如果",
        "因为",
        "所以",
        "通过",
        "进行",
        "如何",
        "怎么",
        "什么",
        "哪些",
        "是否",
        "问题",
        "系统",
        "公司",
        "使用",
        "需要",
        "提供",
        "相关",
        "信息",
        "数据",
        "请问",
        "一个",
        "没有",
        "我们",
        "你们",
        "他们",
    ]
)
# 单次 Observation 最多高亮的关键片段数
_HL_MAX_HIGHLIGHTS = 3


# ====== 解析结果 ======
@dataclass
class ReactParseResult:
    is_final: bool
    answer: str | None
    thought: str
    action: str | None
    tool_input: str | None
    raw: str
    need_reminder: bool = False


class SmartAgent:
    """
    智能 Agent（Smart Agent）—— 纯 Prompt 驱动的 ReAct 实现。

    - LLM 在每轮只输出 <thought>/<action>/<input>。
    - 系统解析标签、执行工具、生成 <observation> 并拼回历史推理轨迹。
    - 支持工具调用重复检测、空/无效结果检测、步数硬上限、嵌入式自审、
      独立 LLM 复盘、Observation 预处理、上下文滑动窗口 + 摘要压缩。
    """

    def __init__(
        self,
        llm_provider: str = None,
        temperature: float = 0.7,
        permission_level: str = "safe",
        agent_id: str | None = None,
        db=None,
        pre_loaded_tools: list | None = None,
        pre_loaded_base_tools: list | None = None,
        agent_system_prompt: str | None = None,
    ):
        """
        初始化 Agent

        Args:
            llm_provider: LLM 提供商（qwen/doubao）
            temperature: 生成温度
            permission_level: 工具权限级别（safe/read/write/admin）
            agent_id: Agent ID，用于加载关联的自定义工具
            db: 数据库会话，用于查询 Agent-工具关联
            pre_loaded_tools: 预加载的 LangChain 工具列表（兼容旧调用，新实现主要用 base_tools）
            pre_loaded_base_tools: 预加载的 BaseTool 实例列表（供直接 run() 执行）
            agent_system_prompt: Agent 自定义系统提示词（从数据库加载）
        """
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER

        self.default_provider = llm_provider
        self.temperature = temperature
        self.permission_level = permission_level
        self.agent_id = agent_id
        self._db = db
        self._agent_custom_prompt = agent_system_prompt

        # 初始化默认工具（如果还没初始化）
        if len(tool_registry) == 0:
            init_default_tools()

        # 获取 LLM（默认 provider）
        self.llm = model_factory.get_llm_model(
            provider=llm_provider,
            temperature=temperature,
        )
        self._llm_cache: dict[str, Any] = {}

        # 解析 BaseTool 实例列表（核心：直接 run() 执行）
        if pre_loaded_base_tools is not None:
            self.base_tools = list(pre_loaded_base_tools)
            logger.info(f"使用预加载 BaseTool: {len(self.base_tools)} 个")
        elif agent_id and db is not None:
            # 同步兜底（异步上下文应由调用方传入 pre_loaded_base_tools）
            self.base_tools = self._resolve_base_tools_sync(agent_id, db, permission_level)
            logger.info(f"同步解析 Agent '{agent_id}' BaseTool: {len(self.base_tools)} 个")
        else:
            # 无 agent_id（系统级默认 Agent）：返回全部已注册 BaseTool
            self.base_tools = tool_registry.get_all()
            logger.info(f"无 agent_id，返回默认 BaseTool 集: {len(self.base_tools)} 个")

        # 工具名称索引
        self._tool_map = {t.name: t for t in self.base_tools}
        self._tool_map_lower = {t.name.lower(): t for t in self.base_tools}

        # 工具列表文本（供 system prompt 注入）
        self.tool_list_text = self._build_tool_list_text()

        # 系统提示词（用户已写好的内容，保持原样）
        self.system_prompt = self._load_system_prompt()

        # 上下文压缩配置
        self._context_cfg = settings.get_react_context_config(llm_provider)

    # ====== 工具解析 ======

    @staticmethod
    async def async_resolve_base_tools(
        agent_id: str,
        db,
        permission_level: str = "safe",
    ) -> list:
        """异步解析 Agent 关联的 BaseTool 实例列表（异步上下文用）"""
        return await load_agent_base_tools(agent_id, db, permission_level)

    @staticmethod
    async def async_resolve_tools(
        agent_id: str,
        db,
        permission_level: str = "safe",
    ) -> list:
        """异步解析 Agent 关联的 LangChain 工具列表（兼容旧调用/无 Agent 模式）"""
        return await load_agent_tools(agent_id, db, permission_level)

    def _resolve_base_tools_sync(
        self,
        agent_id: str | None,
        db,
        permission_level: str,
    ) -> list:
        """同步兜底解析（仅在非异步上下文且无预加载时使用）"""
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                logger.warning(
                    "在异步上下文中调用 _resolve_base_tools_sync，返回空列表，请使用 pre_loaded_base_tools"
                )
                return []
            except RuntimeError:
                pass
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(load_agent_base_tools(agent_id, db, permission_level))
        except Exception as e:
            logger.error(f"同步解析 Agent '{agent_id}' BaseTool 失败: {e}")
            return []

    # ====== 系统提示词 ======

    def _load_system_prompt(self) -> str:
        """
        加载系统提示词（CoT Prompt + Few-shot 模式）。

        若存在 agent 自定义系统提示词，则追加到末尾。
        """
        # 构建精简工具列表（名称 + 一句话描述）
        tool_descriptions = []
        for tool in self.base_tools:
            tool_desc = (tool.description or "").split("\n")[0].strip()
            tool_descriptions.append(f"- {tool.name}: {tool_desc}")

        if self.base_tools:
            tools_section = "\n".join(tool_descriptions)
            prompt = f"""你是企业研发智能体，擅长自主调用可用工具帮助用户解决问题，严格遵循以下规则执行：

1.你可以使用以下工具解决问题：
{tools_section}

2.输出必须严格遵循以下格式，每次只能输出一个步骤：
<thought>分析当前问题，说明下一步要做什么、为什么选这个工具</thought>
<action>工具名称，必须是上面列出的工具之一</action>
<input>工具的参数，严格按照工具要求的格式传入</input>

3. 当你认为信息足够回答问题时，使用以下格式输出最终答案：
<thought>说明已经收集到足够信息，可以给出答案</thought>
最终答案内容

3.1 【停止条件（何时必须结束）】出现以下任一情况，立即停止调用工具并输出最终答案：
  - 已通过工具取得回答所需全部事实/数值，无需再检索或计算；
  - knowledge_search 已连续检索 2 次仍无相关结果（视为知识库未覆盖 → 走统一拒答模板）；
  - 用户问题本身不需要任何工具即可回答（已知数据齐全 / 纯问候闲聊等）。
  停止表达方式（二选一，必须明确选择其一，禁止只发 <thought> 而不输出答案）：
  - 方式A：<thought>说明信息已足够</thought> 然后直接写最终答案（纯自然语言，不带任何标签）；
  - 方式B：<action>none</action> 然后直接写最终答案（表示"无需调用工具，立即作答"）。
  禁止在"想结束"时编造不存在的工具名或反复调用；若确实无需工具，请用以上方式明确停止。

4. 重要规则：
- 【最高优先级】接收到用户的问题时必须先做俩件事：1.提取用户问题内全部已知指标与事实、2.根据上下文及用户提问中的内容判断求解答案是否还需要调用外部工具并输出判读结论。如果历史已经返回本次问题需要的全部字段数据，禁止调用任何工具，禁止再次检索，直接进行推理。
- 【最高优先级】纯计算题要调用计算工具。
- 不要生成 <observation> 标签，工具执行结果会由系统自动追加到对话中
- 每次只能调用一个工具，不要并行输出多个动作
- 【最高优先级-输出格式铁律】需要调用工具时，必须同时写出 <action> 和 <input> 两个标签（例如 <action>calculator</action><input>9200 / 20</input>）。只写工具名、或写 "Action: 工具名" 而不带标签，系统无法识别为工具调用，会被当作最终答案直接返回，导致计算/检索根本没有真正执行。
- 不要重复调用参数完全相同的同一工具
- 若需要计算多个结构相似的表达式，请使用 calculator 的 expressions 参数一次性传入多个表达式，避免逐条调用
- 【最高优先级】每次都要根据上文判断是否需要使用工具，如果是必要的，才调用
- 调用 calculator 时必须同时输出 <input>；input 中必须是合法数学表达式或 expressions 数组，不能为空
- 生成计算表达式前后，注意保证数据的单位一致；单位一致时直接计算，不要进行单位换算
- 【最高优先级】计算结果默认保留两位小数（例如 20.31%）；若题目明确指定了其他精度（如“保留一位小数”“取整”），则遵循题目要求。
- 【最高优先级·数值计算铁律】凡是需要计算任何数值（含乘方、百分比、增长率、金额、求和、同比/环比、倍数、平均值等），无论问题是否简单，都必须调用 calculator 完成；最终答案中出现的所有数字、比例、百分比、计算类结论，必须有对应 calculator 工具返回的结果作为依据，禁止自行心算、口算、估算或在答案中改写计算结果。
- 【最高优先级·纯算术题】纯数学计算、不涉及企业业务数据的算术题（如"100除以4加20""2的10次方""365乘以24"），直接调用 calculator 完成，禁止调用 knowledge_search 检索。
- 【最高优先级·文件/检索结果直接引用】通过 file_reader 或 knowledge_search 返回的内容中，若已直接包含问题所求的具体数值、结论、原文内容（如违约金金额、缺陷数量、具体日期、明确条文、已算好的结果），必须直接引用该数值作答；【绝对禁止】再调用 calculator 或 knowledge_search 重复验证相同数据。仅当工具返回只给出比例/原料而未给出最终答案时（如只说"总房款的20%"但未给金额），才调用 calculator 补全计算。
- 提问业务、技术相关时先检索知识库再推理：凡是涉及公司业务、技术、制度、流程、API、权限、业务数据、产品/人员/项目等企业内部主题的问题，必须先调用 knowledge_search 取得依据后再推理作答；禁止凭记忆或预训练知识直接回答，未检索到才走拒答模板。
- 用户在问题中给出业务数据时先进行知识库检索验证后再计算：若用户在问题中直接提供了企业内部业务数据（如营收、销量、成本、利润、人数等），不要直接用这些数字作答或计算，必须先调用 knowledge_search 检索验证其真实性/准确性；确认与知识库一致后再计算，若不一致则以知识库为准并在答案中说明差异。
- 多轮对话中的代词/省略要继承前文主题：若当前问题包含代词（如"那病假呢""这个接口""它"）或省略表述，必须结合对话历史中的前文主题确定真实实体后再决定检索/回答，不得把省略问题孤立当作全新未知问题直接拒答。
- 多轮累积后仍以工具事实为准：即使对话已进入多轮、前面已有检索/计算结果，只要当前问题涉及企业内部主题（业务/技术/制度/流程/数据/API/权限/人员/项目等），作答仍须基于工具最新返回的事实，**禁止仅凭历史摘要或预训练记忆直接作答**；若当前问题与历史强相关且历史已有可靠工具结果，可复用该结果，但不得用记忆替代工具调用。
- 多轮对话复用规则：多轮对话中，前文已经通过工具获取过的数据、结论，可以直接复用；**禁止重复调用相同工具查询相同内容**；回答后续问题必须结合上下文历史信息，缺失的数据再调用工具补充。
- 识别问候/致谢等社交场景直接回应：当问题属于问候（如"你好""你是谁"）、致谢（如"谢谢""感谢"）、闲聊等非业务/技术查询时，不调用 knowledge_search，直接用自然语言友好回应，不要套用拒答模板；其中致谢类问题必须回应"不客气/不用谢/很高兴能帮到你"，问候类问题才回应"您好/很高兴为您服务"；当被问到"你是谁/你是什么"等身份类问题时，应明确自我介绍，例如"我是企业的智能助手，可以帮你查询公司制度、技术文档、业务数据等内容。"
- 【统一拒答模板】当判定无法回答时（knowledge_search 返回为空/无关、问题超出企业知识库范围、涉及外部信息/时事/预测、或涉及隐私/安全/敏感信息），必须直接拒绝，并**原样使用以下固定句式**，禁止强行作答、编造、推测，或补充"通常/一般/可能"之类的通用常识与外部知识；**尤其禁止以"历史对话中已明确指出/前文已说明/之前已告知"等表述作为拒答理由（拒答只依据本轮检索结果，不引用历史）**：
  - 知识库未覆盖："抱歉，我在知识库中未找到相关信息，无法提供答案。"
  - 外部/时事/预测类："抱歉，该问题涉及外部信息/预测，不在企业知识库范围内，无法提供答案。"
  - 隐私/安全/敏感类（如身份证号、手机号、薪资明细、员工个人信息、账号密码等个人数据）："抱歉，该问题涉及隐私/安全信息，无法提供答案。"
- 【答案须接地】给出最终答案时，必须基于检索/工具返回的具体事实作答，并尽量点出关键信息来源或原文事实（如"根据知识库《XXX》""检索结果显示…""文件载明…"）；答案中出现的具体数字、专名、条款、日期等，须能在对应工具的 Observation 中找到依据，禁止脱离检索结果凭空得出结论或补充外部常识。
- 【禁止常识推测填补】知识库/文档未覆盖的部分，禁止基于常识或经验推测补全（如"通常会…""应该是…""一般会…"）；只要最终答案里含推测性内容，必须整体改用「知识库未覆盖」拒答模板，禁止输出「承认未找到 + 推测内容」的混合答案。
- 【最终答案格式】最终答案只输出面向用户的自然语言结论，禁止输出工具名、内部字段、调用标签、调试信息、JSON 结构、以及 answer/called_tools/tool_results 等字段名。

--- 示例 ---
示例一 - 复合链式推理（「单检索 + 多步计算」+「二次检索补数 + 继续计算」两类混合场景）
用户问题：家居家纺品类的总退款损失是多少？五个品类中哪个品类退款损失最高？
<thought>计算退款损失需要GMV和退款率两类数据，先检索各品类退款率</thought>
<action>knowledge_search</action>
<input>各品类退款率 家居家纺 厨房用品 个护清洁 小家电 服饰配件</input>
<observation>五个品类退款率：家居家纺 4.2%、厨房用品 5.8%、个护清洁 3.1%、小家电 8.5%、服饰配件 12.3%。</observation>
<thought>已拿到退款率，但缺少各品类GMV，无法计算损失，需补充检索GMV数据</thought>
<action>knowledge_search</action>
<input>各品类总GMV 家居家纺 厨房用品 个护清洁 小家电 服饰配件</input>
<observation>五个品类总GMV：家居家纺 1280万元、厨房用品 2160万元、个护清洁 950万元、小家电 1820万元、服饰配件 3200万元。</observation>
<thought>两类数据已齐全，退款损失=总GMV×退款率，先计算家居家纺</thought>
<action>calculator</action>
<input>1280 * 0.042</input>
<observation>53.76</observation>
<thought>家居家纺退款损失53.76万元，继续计算服饰配件（退款率最高）</thought>
<action>calculator</action>
<input>3200 * 0.123</input>
<observation>393.6</observation>
<thought>已算出核心数据：家居家纺53.76万元，服饰配件393.6万元为最高，信息充足可以作答</thought>
家居家纺品类总退款损失为 53.76 万元。五个品类中退款损失金额最高的是服饰配件品类，为 393.6 万元。

示例二 - 边界行为复合（「无答案拒答」+「工具决策判断」两类边界场景）
用户问题：2025年7月平台新增注册用户总数是多少？
<thought>用户询问的7月新增用户数据不在已知常识范围内，需要检索知识库确认是否有收录</thought>
<action>knowledge_search</action>
<input>2025年7月 新增注册用户数</input>
<observation>未检索到与“2025年7月新增注册用户”相关的内容，知识库暂无该维度数据。</observation>
<thought>知识库无对应数据，无法回答，必须如实告知用户，禁止编造数据</thought>
目前知识库中暂无 2025 年 7 月平台新增注册用户总数的相关数据，无法为你解答该问题。

示例三 - 中文算式歧义（纯算术题直接调用 calculator，不要自加括号）
用户问题：100除以4加20等于多少？
<thought>这是纯数学计算，不涉及企业业务数据，直接调用 calculator，按从左到右先除后加</thought>
<action>calculator</action>
<input>(100/4)+20</input>
<observation>(100/4)+20 = 45.0</observation>
<thought>计算器已返回结果 45.0，信息充足可以作答</thought>
100除以4加20等于 45。

示例四 - 混合场景公式推导（先明确公式，再代入数值计算）
用户问题：公司总成本是净利润的多少倍？
<thought>用户问"总成本是净利润的多少倍"，需要先明确公式：倍数 = 总成本 / 净利润；然后从知识库获取总成本和净利润</thought>
<action>knowledge_search</action>
<input>公司总成本 净利润</input>
<observation>公司总成本为 2930 万元，净利润为 920 万元。</observation>
<thought>已获得数值，公式为 倍数 = 总成本 / 净利润 = 2930 / 920，调用 calculator 计算</thought>
<action>calculator</action>
<input>2930/920</input>
<observation>2930/920 = 3.18</observation>
<thought>计算完成，倍数约为 3.18 倍，信息充足可以作答</thought>
公司总成本约为净利润的 3.18 倍。

示例五 - 不用调用工具的情况
用户问题：已知五大区人均营收：华东 8.2万，华南9.5万，华北7.1万，西南11.3万，西北6.8万。哪个大区人均营收最高？
<thought>
五大区人均营收数据全部给出：华东8.2万，华南9.5万，华北7.1万，西南11.3万，西北6.8万。求解答案不需要调用外部工具，数据完整，直接对比数值完成排序。
</thought>
<action>none</action>
<observation>无</observation>
<thought>五大区人均营收对比：西南11.3万 > 华南9.5万 > 华东8.2万 > 华北7.1万 > 西北6.8万。西南大区人均营收最高。信息充足可以作答</thought>
西南大区人均营收最高。

--- 示例结束 ---
"""
        else:
            prompt = """你是企业研发智能体。当前无可用工具，请直接对话。
- 用自然友好的方式回答
- 需要外部信息时，诚实告知没有相关工具"""

        # 如果 agent 有自定义系统提示词，追加到末尾
        if self._agent_custom_prompt:
            prompt += f"\n\n【补充指令】\n{self._agent_custom_prompt}"

        return prompt

    # ====== 工具列表文本 ======

    def _build_concise_tools_text(self) -> str:
        """精简工具列表（名称 + 一句话描述），用于填充 {tools_section} 占位符"""
        lines = []
        for t in self.base_tools:
            desc = (t.description or "").split("\n")[0].strip()
            lines.append(f"- {t.name}: {desc}")
        return "\n".join(lines) if lines else "（无可用工具）"

    def _build_tool_list_text(self) -> str:
        """详细工具列表（含参数 JSON Schema），作为独立 section 注入 system prompt"""
        parts = []
        for t in self.base_tools:
            schema = t.parameters_schema or {}
            try:
                schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            except Exception:
                schema_str = str(schema)
            parts.append(
                f"### {t.name}\n{t.description or ''}\n参数（JSON Schema）：\n```json\n{schema_str}\n```"
            )
        return "\n\n".join(parts)

    # ====== system message 构建 ======

    def _build_system_message(
        self,
        query: str,
        embed_review: bool = False,
        force_final: bool = False,
    ) -> str:
        """
        拼接完整 system message：
        固定 system prompt（含 {tools_section} 填充） + 可用工具参数 section
        + 当前问题 + 最新一轮推理轨迹 + "请继续推理" 指令（+ 自审/强制终止提示）。

        单臂 ReAct 改造（2026-10-12）：轨迹块只渲染「最新一轮」原文，
        历史事实的连续性由上方「轮间状态归集（AgentState 确定性摘要）」块承载，
        因此每轮 prefill 从 O(轮数) 降为 O(1)，不再重发历史全量 Observation。
        """
        base = self.system_prompt
        if "{tools_section}" in base:
            base = base.replace("{tools_section}", self._build_concise_tools_text())

        system = base
        if self.base_tools:
            system += "\n\n## 可用工具（参数格式）\n" + self.tool_list_text

        if force_final:
            system += (
                "\n\n【强制终止】你已达到最大推理步数上限，请立即停止调用任何工具，"
                "基于历史推理轨迹中已有的信息，直接给出最终完整答案。"
                "格式：<thought>说明已经收集到足够信息，可以给出答案</thought> 然后紧跟最终答案。"
            )
            # 防幻觉兜底：若过程中曾调用工具但均失败/未返回有效数据，
            # 必须直接按拒答模板回应，禁止编造任何具体事实或数字。
            if not self._has_successful_retrieval:
                system += (
                    "\n注意：你在本轮推理中调用的工具均未成功返回有效数据，"
                    '因此必须直接按拒答模板（包含"无法提供"）回应，'
                    "禁止输出任何具体事实、数字或猜测性结论。"
                )

        conv_hist = getattr(self, "_conversation_history", []) or []
        if conv_hist:
            hist_lines = []
            for m in conv_hist:
                role = m.get("role")
                content = m.get("content", "")
                line = f"{role_label_cn(role)}：{content}"
                # 拒答类历史结论降权为软结论：标注"可被本轮新证据推翻"，避免模型锚定旧拒答
                if role == "assistant" and _contains_refusal(content):
                    line += (
                        "（⚠️ 前轮拒答结论，可被本轮新检索证据推翻；"
                        "若本轮 knowledge_search 检索到相关内容，请以新证据为准，勿直接沿用或引用此拒答）"
                    )
                hist_lines.append(line)
            system += (
                "\n\n## 历史对话（之前的交流，可据此复用前文事实）\n" + "\n".join(hist_lines) + "\n"
            )

        # 轮间状态归集块（代码确定性抽取，非 LLM 维护；仅作上下文提醒，不替代历史轨迹）
        state_obj = getattr(self, "_state", None)
        if state_obj is not None:
            state_text = state_obj.render_for_prompt()
            if state_text:
                system += "\n\n" + state_text + "\n"

        kb_block = self._render_kb_sources()
        if kb_block:
            system += "\n\n" + kb_block + "\n"

        trace_block = self._render_trace_latest_only()

        system += f"\n\n当前问题：{query}\n\n历史推理轨迹：\n{trace_block}\n\n"
        if force_final:
            system += (
                "请直接给出最终答案，不要输出任何 <action> / <input> 标签，"
                "也不要重复之前已经生成过的工具调用步骤。"
            )
        else:
            system += "请继续推理："

        if embed_review:
            system += (
                "\n\n【自审提示】在输出下一步之前，请先回顾之前的推理过程："
                "是否仍围绕用户的原始问题？是否存在无效步骤"
                "（重复以相似参数调用同一工具、工具返回空或无关信息、在已掌握足够信息时仍反复检索）？"
                "若存在，请立即纠正或基于已有信息给出答案。"
            )
        return system

    # ====== ReAct 解析 ======

    def _loose_parse_react(self, raw: str) -> ReactParseResult | None:
        """
        弱格式容错：当 LLM 未严格输出 <action>/<input> XML 标签时，
        尝试从常见退化格式中还原出一步工具调用，避免把「工具名+参数」误当最终答案。

        覆盖的退化格式：
          1) 经典 ReAct：「Action: 工具名」+「Action Input: 参数」
          2) 裸「工具名\\n参数」（首行恰好是某个可用工具名，其后各行作为 <input>）
          3) 首行「工具名 参数」/「工具名：参数」（工具名后以空格/冒号直接跟参数）

        仅当原文不含 <action> 标签时才启用，避免与正常解析冲突。
        若还原出的工具名不在可用列表中，返回 None（交给正常逻辑处理）。
        """
        if not raw:
            return None
        s = raw.strip()
        if "<action>" in s or "</action>" in s:
            return None

        # 1) 经典 ReAct：Action: / Action Input:
        m_act = re.search(r"Action\s*:\s*([^\n]+)", s, re.IGNORECASE)
        if m_act:
            tool_name = m_act.group(1).strip().rstrip(":").strip()
            tool = self._find_tool(tool_name)
            if tool is not None:
                m_in = re.search(r"Action\s*Input\s*:\s*(.+)", s, re.IGNORECASE | re.DOTALL)
                params = m_in.group(1).strip() if m_in else ""
                return ReactParseResult(False, None, "", tool.name, params, raw)

        # 提取 <thought>（若有），剩余部分用于判定工具调用
        thought = ""
        tm = re.search(r"<thought>(.*?)</thought>", s, re.DOTALL)
        if tm:
            thought = tm.group(1).strip()
            body = s[tm.end() :].strip()
        else:
            body = s

        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not lines:
            return None

        # 2) 首行恰好是工具名 → 其余行作为 <input>
        first = lines[0]
        tool = self._find_tool(first)
        if tool is not None:
            params = "\n".join(lines[1:]).strip()
            return ReactParseResult(False, None, thought, tool.name, params, raw)

        # 3) 首行「工具名 参数」/「工具名：参数」
        stripped = first.rstrip(":")
        for name in self._tool_map:
            low = stripped.lower()
            if (
                low == name.lower()
                or low.startswith(name.lower() + " ")
                or low.startswith(name.lower() + "：")
            ):
                tool = self._tool_map[name]
                rest = first[len(name) :].lstrip(": ：").strip()
                params = (rest + "\n" + "\n".join(lines[1:])).strip()
                return ReactParseResult(False, None, thought, tool.name, params, raw)

        return None

    def _parse_react(self, raw: str) -> ReactParseResult:
        """
        解析 LLM 输出的 ReAct 文本。

        - 含 <action> → 视为一步工具调用（非最终）。
        - 无 <action> 但 <thought> 含「已经收集到足够信息 / 可以给出答案」 → 最终答案。
        - 无 <action> 且 <thought> 之后有实质性答案文本（无新标签） → 视为最终答案。
        - 完全无标签的纯文本 → 视为最终答案。
        - 有 <thought> 但无 <action> 也无答案 → 视为卡住（need_reminder）。
        """
        if not raw:
            return ReactParseResult(False, None, "", None, None, raw, need_reminder=True)

        # 弱格式容错：还原退化的工具调用写法（如「工具名\\n参数」「Action: 工具名」），
        # 优先于"最终答案"回退，避免计算/检索被当答案直接返回而未真正执行。
        loose = self._loose_parse_react(raw)
        if loose is not None:
            return loose

        thought_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action_match = re.search(r"<action>(.*?)</action>", raw, re.DOTALL)

        # 有 action → 工具调用步骤（非最终）
        if action_match:
            action = action_match.group(1).strip()
            # 归一化"无需工具/停止"信号：action=none/无/等 → 直接视为强制总结（Final Answer），
            # 不再进入工具校验（避免触发"工具 none 不在列表"的无效重试，省 1~2 轮）。
            if action.lower() in _NONE_ACTIONS:
                rest = raw.split("</action>", 1)[-1]
                # 去掉可能紧跟的 <observation>无</observation> 占位，取后续最终答案
                rest = re.sub(
                    r"<observation>.*?</observation>", "", rest, flags=re.DOTALL | re.IGNORECASE
                ).strip()
                rem_tm = re.search(r"<thought>(.*?)</thought>", rest, re.DOTALL)
                after_text = rest[rem_tm.end() :].strip() if rem_tm else rest.strip()
                if not after_text.strip():
                    # 模型表达了"停止"(none)但没写答案：不能把 thought（流程元话语）当答案，
                    # 走格式提醒让下一轮直接输出最终答案（否则 Final Answer=思考文本）。
                    return ReactParseResult(
                        False, None, thought, None, None, raw, need_reminder=True
                    )
                return ReactParseResult(True, after_text, thought, None, None, raw)
            # 单步格式校验：一条输出里只允许出现一个 <action>；
            # 若模型一次吐出多个工具调用，或自行伪造 <observation> 标签，
            # 判定为格式退化，返回格式提醒让其重新只输出「一步」。
            action_count = len(re.findall(r"<action>.*?</action>", raw, re.DOTALL))
            if action_count > 1 or re.search(r"<observation>", raw, re.IGNORECASE):
                return ReactParseResult(False, None, thought, None, None, raw, need_reminder=True)
            input_match = re.search(r"<input>(.*?)</input>", raw, re.DOTALL)
            if input_match:
                tool_input = input_match.group(1).strip()
            else:
                # 兼容 stop sequence 在 </input> 之前截断导致的未闭合 input：
                # 提取 <input> 之后到文本末尾/下一个标签之前的内容作为参数。
                input_start_match = re.search(r"<input>(.*)", raw, re.DOTALL)
                if input_start_match:
                    tool_input = input_start_match.group(1).strip()
                    next_tag = re.search(r"<[^/]", tool_input)
                    if next_tag:
                        tool_input = tool_input[: next_tag.start()].strip()
                else:
                    tool_input = ""
            return ReactParseResult(False, None, thought, action, tool_input, raw)

        # 无 action：判断是否最终
        if thought and re.search(r"已经收集到足够信息|可以给出答案|信息(已|足够|充足)", thought):
            after = raw.split("</thought>")[-1].strip()
            return ReactParseResult(True, after, thought, None, None, raw)

        # 无 action：看 <thought> 之后是否有答案文本
        remaining = raw
        if thought_match:
            remaining = raw[thought_match.end() :].strip()
        if remaining and not re.search(r"<action>|<thought>", remaining):
            return ReactParseResult(True, remaining, thought, None, None, raw)

        # 无任何标签：纯文本答案
        if not thought_match and not action_match:
            return ReactParseResult(True, raw.strip(), "", None, None, raw)

        # 卡住：有 thought 但无 action 也无答案
        return ReactParseResult(False, None, thought, None, None, raw, need_reminder=True)

    # ====== 工具参数解析与执行 ======

    def _parse_tool_args(self, tool, raw_input: str) -> dict | None:
        """
        将 <input> 文本解析为工具参数字典。
        - 若可解析为 JSON 对象 → 直接使用。
        - 否则若工具仅有一个（必填）参数 → 整段原文作为该参数值（兼容 calculator 的 "9200 / 20" / knowledge_search 的自然语言 query）。
        - 多参数工具且无法 JSON 解析 → 返回 None（失败）。
        """
        if raw_input is None:
            raw_input = ""
        raw_input = raw_input.strip()
        # v2（2026-09-18）：剥离 Observation 高亮遗留的【】括号。模型常把高亮片段
        # （如【星辰】科技【Q2】销售运营报告.txt）原样抄进工具参数，导致 file_reader
        # 找不到文件、knowledge_search 检索串变脏。统一在此清洗，无论 JSON 还是纯文本参数。
        if "【" in raw_input or "】" in raw_input:
            raw_input = raw_input.replace("【", "").replace("】", "")
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        schema = tool.parameters_schema or {}
        required = schema.get("required", [])
        props = list(schema.get("properties", {}).keys())

        if len(required) == 1:
            return {required[0]: raw_input}
        if len(props) == 1:
            return {props[0]: raw_input}

        # 兜底：当 schema 无 required 字段但存在字符串类型主参数时，把纯文本映射过去。
        # 典型场景：calculator 的 required 为了兼容 expressions 改为 [] 后，单表达式
        # "100/4+20" 仍应落到 expression 参数，而不是因无法匹配而参数解析失败。
        if not required and raw_input:
            string_props = [
                p for p in props if schema.get("properties", {}).get(p, {}).get("type") == "string"
            ]
            for preferred in ("expression", "query", "content", "input", "reason", "text"):
                if preferred in string_props:
                    return {preferred: raw_input}
            if string_props:
                return {string_props[0]: raw_input}

        # 多参数且非 JSON：尝试把 "k=v, k2=v2" 形式解析
        if props and ("," in raw_input or "=" in raw_input):
            result = {}
            for pair in raw_input.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip().strip('"').strip("'")
                    v = v.strip().strip('"').strip("'")
                    if k in props:
                        result[k] = v
            if result:
                return result
        return None

    def _find_tool(self, action: str):
        """按名称（含大小写不敏感）查找工具"""
        if not action:
            return None
        if action in self._tool_map:
            return self._tool_map[action]
        return self._tool_map_lower.get(action.lower())

    async def _execute_tool(self, tool, raw_input: str) -> tuple[Any, str, dict]:
        """
        执行工具，带重试（≤ MAX_TOOL_RETRIES）。

        Returns:
            (ToolResult, observation_text, parsed_args)
        """
        args = self._parse_tool_args(tool, raw_input)
        if args is None:
            return (
                None,
                f"工具 '{tool.name}' 参数解析失败：<input> 应为 JSON 或单个参数。",
                {},
            )

        # —— Pydantic JSON Schema 强校验（类型 / 必填 / 枚举）——
        # 在 tool.run() 之前拦截参数格式错误，错误信息作为 Observation 回传，
        # 让模型自行修正后重试（不消耗执行重试次数 MAX_TOOL_RETRIES）。
        ok, err = tool.validate_args(**args)
        if not ok:
            return (
                None,
                f"工具 '{tool.name}' 参数格式校验未通过：{err}。"
                "请严格按工具描述的参数格式（类型 / 必填 / 枚举）重新调用。",
                args,
            )

        last_obs = ""
        for attempt in range(MAX_TOOL_RETRIES):
            try:
                result = await tool.run(**args)
                if isinstance(result, ToolResult) and result.success:
                    return result, result.to_string(), args
                # 执行返回但 success=False
                obs = result.to_string() if isinstance(result, ToolResult) else str(result)
                last_obs = obs
                if attempt < MAX_TOOL_RETRIES - 1:
                    logger.warning(f"工具 '{tool.name}' 第{attempt + 1}次返回非成功，重试")
                    continue
                return result, obs, args
            except Exception as e:
                logger.error(f"工具 '{tool.name}' 执行异常 (第{attempt + 1}次): {e}")
                last_obs = f"工具 '{tool.name}' 执行异常：{str(e)[:200]}"
                if attempt < MAX_TOOL_RETRIES - 1:
                    continue
                return None, last_obs, args
        return None, last_obs, args

    # ====== Observation 预处理 ======

    @staticmethod
    def _highlight_terms(text: str, query: str) -> str:
        """
        轻量关键词高亮（v2）：整词优先 + 优先级排序 + 上限 3，抑制【】噪声。

        改造动机：旧版对每个 2~3 字 n-gram 都打【】，密度过高。模型常把高亮片段
        （如【星辰】科技【Q2】销售运营报告.txt）原样抄进工具参数，导致 file_reader
        找不到文件 / knowledge_search 检索串变脏。v2 收敛为：
        - 候选仍是 query 中的 2~3 字中文 n-gram 与拉丁词（过滤停用词）；
        - 优先级：按长度降序（越长越具体，越可能是实体/专名，越该高亮）；
        - cap 3：单次 Observation 最多高亮 3 个最关键片段，剩余不包【】，大幅降噪。
        - 仍仅高亮正文真实存在的片段（k in text 过滤）。
        """
        if not text or not query:
            return text
        cands: set[str] = set()
        for run in _HL_CJK_RE.findall(query):
            run = run.strip()
            if len(run) < 2:
                continue
            for n in (3, 2):
                for i in range(len(run) - n + 1):
                    cands.add(run[i : i + n])
        for tok in _HL_LATIN_RE.findall(query):
            if len(tok) >= 2:
                cands.add(tok)
        kws = [k for k in cands if k not in _HL_STOPWORDS and k in text]
        if not kws:
            return text
        # 优先级：长度降序（更具体者优先），再取前 _HL_MAX_HIGHLIGHTS 个，抑制噪声
        kws.sort(key=len, reverse=True)
        kws = kws[:_HL_MAX_HIGHLIGHTS]
        pat = re.compile("|".join(re.escape(k) for k in kws))
        return pat.sub(lambda m: f"【{m.group(0)}】", text)

    async def _preprocess_observation(self, text: str, query: str) -> str:
        """对工具返回的 Observation 做长度/相关性预处理，并做轻量关键词高亮引导注意力。"""
        if not text:
            return "（无返回结果）"
        if len(text) <= OBS_TRUNCATE_SOFT:
            return self._highlight_terms(text, query)

        if len(text) <= OBS_EXTRACT_HARD:
            # 软截断：保留头尾 + 关键词窗口
            return self._highlight_terms(self._soft_truncate(text, query), query)

        # 硬提取：调用 LLM 抽取核心信息
        extracted = await self._llm_extract_observation(query, text)
        final = extracted or self._soft_truncate(text[:LLM_EXTRACT_MAX_INPUT], query)
        return self._highlight_terms(final, query)

    @staticmethod
    def _soft_truncate(text: str, query: str, head: int = 800, tail: int = 800) -> str:
        """保留头部 head 字符 + 尾部 tail 字符，并在中间插入提示。"""
        if len(text) <= head + tail + 50:
            return text
        head_part = text[:head]
        tail_part = text[-tail:]
        # 关键词窗口：在头尾之外寻找 query 关键词附近内容
        keywords = [
            kw
            for kw in re.findall(r"[\u4e00-\u9fff]{2,}", query)
            if len(kw) >= 2 and kw not in _HL_STOPWORDS
        ][:5]
        mid = text[head:-tail]
        window = ""
        for kw in keywords:
            idx = mid.find(kw)
            if idx >= 0:
                start = max(0, idx - 100)
                end = min(len(mid), idx + 200)
                window = mid[start:end]
                break
        if window:
            return f"{head_part}\n...\n（与问题相关的片段）{window}\n...\n{tail_part}"
        return f"{head_part}\n...\n（内容过长已截断）\n...\n{tail_part}"

    async def _llm_extract_observation(self, query: str, text: str) -> str | None:
        """调用 LLM 从工具返回结果中提取与用户问题相关的核心信息（≤500 字）。"""
        try:
            src = text[:LLM_EXTRACT_MAX_INPUT]
            system = "你是信息抽取助手。请基于用户问题，从工具返回结果中提取所有相关核心信息，忽略无关内容，输出不超过 500 字。"
            user = f"用户问题：{query}\n\n工具返回结果：\n{src}\n\n请提取核心信息："
            resp = await self.llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ]
            )
            extracted = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            if extracted:
                return extracted[:600]
        except Exception as e:
            logger.warning(f"LLM 抽取 Observation 失败，使用软截断兜底: {e}")
        return None

    # ====== 校验机制 ======

    def _is_repeat_loop(self, action: str, raw_input: str) -> bool:
        """
        重复调用检测（三维判定，需同时满足：连续同工具 + 参数相似 + 结果停滞/无新增）。

        1. 连续 2~3 轮调用同一工具（只看该工具的最近调用记录，不受中间其他工具影响）；
        2. 参数相似度超过阈值（编辑距离 或 字符级余弦相似度，取较强信号）；
        3. 返回结果无新增有效信息：同工具近两次观测高度相似（结果已停滞），
           或上一次该工具调用已触发空/无效结果干预。
        兜底：即便结果不同，若连续 3 次参数高度相似也判为循环，防死循环。
        """
        # 只看当前工具的调用历史
        same_actions = [(a, i) for (a, i) in self._recent_actions if a == action]
        if len(same_actions) < 2:
            return False

        same_tool_obs = [o for (t, o) in self._recent_observations if t == action]

        last_input = same_actions[-1][1]

        # 维度2：当前参数与最近一次同工具参数相似
        param_sim = (
            similar_args(last_input, raw_input, 0.85)
            or cosine_similarity(last_input, raw_input) >= 0.85
        )
        if not param_sim:
            return False

        # 维度3：结果停滞 / 上一次该工具已触发空/无效干预
        no_new_info = False
        if same_tool_obs:
            last_obs = same_tool_obs[-1]
            # 如果上一次同工具调用返回的是空/无效干预，且当前参数仍相似，
            # 说明模型在重复同样的错误调用，直接判为循环。
            if "[干预]" in last_obs and (
                "返回结果为空或无效" in last_obs or "参数解析失败" in last_obs
            ):
                return True
            # 兜底：连续 3 次同工具参数高度相似（即便结果不同）也判为循环
            if len(same_actions) >= 3:
                if (
                    similar_args(same_actions[-2][1], raw_input, 0.85)
                    or cosine_similarity(same_actions[-2][1], raw_input) >= 0.85
                ) and (
                    similar_args(same_actions[-3][1], raw_input, 0.85)
                    or cosine_similarity(same_actions[-3][1], raw_input) >= 0.85
                ):
                    return True
            # 正常 stagnation：同工具近两次观测高度相似
            if len(same_tool_obs) >= 2:
                prev_a, prev_b = same_tool_obs[-2], same_tool_obs[-1]
                no_new_info = (
                    similar_args(prev_a, prev_b, 0.9) or cosine_similarity(prev_a, prev_b) >= 0.9
                )

        return no_new_info

    @staticmethod
    def _is_empty_result(result, obs_text: str) -> bool:
        """无效结果检测：执行失败 / 空数据 / 返回空或无关指示。"""
        if result is None:
            # 参数解析失败等场景，视为无效结果（触发干预提示）
            return True
        if not isinstance(result, ToolResult):
            return False
        if not result.success:
            return True
        data = result.data
        if data is None:
            return True
        if isinstance(data, list) and len(data) == 0:
            return True
        indicators = [
            "未找到",
            "未检索到",
            "没有找到",
            "暂无相关",
            "无数据",
            "no results",
            "empty",
            "错误",
            "失败",
        ]
        low = (obs_text or "").lower()
        if any(ind in low for ind in indicators):
            return True
        return False

    def _kb_result_empty_or_irrelevant(self, result, obs_text: str) -> bool:
        """
        判定 knowledge_search 是否“返回空或无关”。

        - 空：执行失败 / data 为空列表 / 含“未找到”等指示（复用 _is_empty_result）。
        - 无关（兜底）：检索成功但返回文档的相关度分数全部低于 KB_RELEVANCE_FLOOR。
          主依据仍是“结果为空”；分数兜底仅用于明显无关但对齐失败的场景。
        """
        if self._is_empty_result(result, obs_text):
            return True
        if isinstance(result, ToolResult) and isinstance(result.data, list) and result.data:
            scores = [float(d.get("score", 0) or 0) for d in result.data if isinstance(d, dict)]
            if scores and max(scores) < KB_RELEVANCE_FLOOR:
                return True
        return False

    async def _kb_retry_with_rewrite(
        self, tool, parsed_args: dict, user_query: str
    ) -> tuple[bool, Any, str, str]:
        """
        knowledge_search 首次为空/无关时，换关键词自动重试（最多 3 次）。

        依次尝试：
          1) 查询扩展（同义/相关术语扩充，缓解词汇不匹配）
          2) HyDE（用假设性文档检索，缓解概念/语义不匹配）
          3) 回退到顶层用户原始问法（LLM 自带的检索词可能是糟糕改写）

        返回 (是否命中, 命中结果, 命中观测文本, 命中查询词)。
        任一候选命中即返回；全部未命中返回 (False, None, "", "")。
        """
        search_q = (parsed_args.get("query") or "").strip()
        candidates: list[str] = []

        # 1) 查询扩展
        try:
            expanded = await query_rewrite_service.expand(search_q)
            if expanded and expanded.strip() and expanded.strip() != search_q:
                candidates.append(expanded.strip())
        except Exception as e:  # 改写失败不影响主流程，继续后续候选
            logger.warning(f"knowledge_search 重试-查询扩展失败: {e}")

        # 2) HyDE 假设文档
        try:
            hyde_doc = await query_rewrite_service.hyde(search_q)
            if hyde_doc and hyde_doc.strip() and hyde_doc.strip() != search_q:
                candidates.append(hyde_doc.strip())
        except Exception as e:
            logger.warning(f"knowledge_search 重试-HyDE 失败: {e}")

        # 3) 回退：用顶层用户原问题再试（仅当与当前检索词不同且未重复）
        if user_query and user_query.strip() and user_query.strip() != search_q:
            if user_query.strip() not in candidates:
                candidates.append(user_query.strip())

        # 最多 3 次重试
        candidates = candidates[:3]

        for cq in candidates:
            try:
                rargs = dict(parsed_args)
                rargs["query"] = cq
                rres = await tool.run(**rargs)
                robs = rres.to_string() if isinstance(rres, ToolResult) else str(rres)
                if not self._kb_result_empty_or_irrelevant(rres, robs):
                    logger.info(f"knowledge_search 关键词重试命中(query={cq!r})")
                    return True, rres, robs, cq
            except Exception as e:
                logger.warning(f"knowledge_search 重试(query={cq!r})执行异常: {e}")

        return False, None, "", ""

    # ====== 新增：工具调用与答案一致性校验（防心算、防重复验证、防重复检索） ======

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        """从文本中提取所有数字（含小数），用于判断计算所需数值是否已在前序工具结果中给出。"""
        return set(re.findall(r"\d+(?:\.\d+)?", text or ""))

    def _should_enforce_calculator(self, query: str, answer: str) -> bool:
        """
        心算校验闸：若答案出现数值/百分比/计算结论，且本轮未调 calculator，且问题疑似计算题，则触发。
        返回 True 时，应拦截最终答案并提示模型调用 calculator。
        """
        # 仅当 calculator 既被调用、又成功返回数值时才豁免；
        # 调用过但失败（如语法错误）仍要强制走 calculator，避免模型心算。
        if self._calculator_returned_number:
            return False
        if not answer:
            return False
        # 答案中的数字、百分比、倍数、比例等计算结论
        if not re.search(r"\d+(?:\.\d+)?(?:%|倍|\.\.\.)?", answer):
            return False
        # 问题中的强计算指示词（避免误伤普通 KB 查询）
        calc_indicators = [
            "计算",
            "等于",
            "占比",
            "百分比",
            "比例",
            "增长率",
            "同比",
            "环比",
            "倍数",
            "总和",
            "合计",
            "平均",
            "均值",
            "乘方",
            "次方",
            "除以",
            "乘以",
            "加上",
            "减去",
        ]
        return any(ind in query for ind in calc_indicators)

    def _is_redundant_file_calc(self, action: str, raw_input: str) -> tuple[bool, str]:
        """
        文件读取后重复验证校验闸：上一轮刚用 file_reader 读过文件，本轮要调用 calculator，
        且 calculator 所需数字全部已在文件结果中出现，则判定为重复验证，应直接引用文件结果。
        返回 (是否冗余, 提示文本)。
        """
        if action != "calculator" or not self._recent_actions:
            return False, ""
        last_action, _ = self._recent_actions[-1]
        if last_action != "file_reader":
            return False, ""
        # 找到最近一次 file_reader 的观测文本
        last_obs = ""
        for a, o in reversed(self._recent_observations):
            if a == "file_reader":
                last_obs = o or ""
                break
        if not last_obs:
            return False, ""
        calc_nums = self._extract_numbers(raw_input)
        file_nums = self._extract_numbers(last_obs)
        if calc_nums and calc_nums.issubset(file_nums):
            return True, (
                "该数据已在刚刚读取的文件结果中明确给出，直接引用即可，禁止用 calculator 重复验证。"
            )
        return False, ""

    def _is_duplicate_tool_call(self, action: str, raw_input: str) -> tuple[bool, str]:
        """
        多轮重复调用校验闸：本轮要调用的工具+参数与历史已调用记录高度重合（且此前已成功返回），
        提示模型直接使用历史结果，避免重复检索/计算。
        返回 (是否重复, 提示文本)。
        """
        from backend.app.utils.token_estimator import similar_args

        for prev_action, prev_input in self._recent_actions:
            if prev_action != action:
                continue
            if similar_args(prev_input, raw_input, 0.92):
                return True, (
                    "该查询已在之前的对话步骤中执行过，请直接使用历史结果，无需重复检索。"
                )
        return False, ""

    @staticmethod
    def _norm_retrieval_query(action: str, raw_input: str) -> str:
        """
        把工具入参归一化为用于比对的「规范查询串」。
        处理三类噪声：① 字符串外层引号 ② knowledge_search 的 JSON 包裹 {"query": "..."}
        ③ 多余空白/大小写。注意：比较的是「模型发起的检索意图」而非Tokenizer 级文本，
        因此即便 round1 与 round3 措辞略有出入，只要归一化后高度相似就视为同一检索。
        """
        if action != "knowledge_search":
            return ""
        s = (raw_input or "").strip().strip('"').strip("'")
        try:
            obj = json.loads(raw_input)
            if isinstance(obj, dict) and "query" in obj:
                s = str(obj["query"])
        except Exception:
            pass
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower()

    def _is_duplicate_retrieval(self, action: str, raw_input: str) -> tuple[bool, str]:
        """
        重复检索闸：若本轮 knowledge_search 的「归一化查询」与此前某次**成功**检索的
        归一化查询近完全一致（>=0.95），判定为重复检索——此前已拿到该数据，重发只会产生
        冗余的大 Observation + 又一轮昂贵 prefill，故拦截并强制复用已有结果作答。

        与 _is_repeat_loop 的区别：
        - _is_repeat_loop 依赖「观测结果停滞」维度，当两次相似检索返回了不同 chunks 时
          会漏判（本次 50.4s 卡死正是此因）；
        - 本闸只认「查询意图 + 此前已成功返回」，与返回内容是否相同无关，故能稳稳拦住
          round3/round4 这类「完全相同的检索被发起两次」的浪费。
        阈值取 0.95（近完全一致）：既能拦住 round4==round3 这种完全相同/仅差一两个字的
        重复检索，又不会误伤「Q2 销售额」与「Q2 总销售额」这类可能返回不同数据、应被允许
        的近义改写。
        """
        if action != "knowledge_search":
            return False, ""
        cur = self._norm_retrieval_query(action, raw_input)
        if not cur:
            return False, ""
        for prev in self._successful_retrieval_queries:
            if similar_args(cur, prev, 0.95) or cosine_similarity(cur, prev) >= 0.95:
                return True, (
                    "该检索已在之前的步骤中以相同查询成功返回过结果，"
                    "请直接复用已有检索结果作答，禁止重复检索同一内容。"
                )
        return False, ""

    def _need_embed_review(self, rounds: int) -> bool:
        """每 3 轮触发一次嵌入式自审提示。"""
        return rounds % 3 == 0

    def _need_llm_review(self) -> bool:
        """独立 LLM 复盘仅在规则检测到异常（连续 2 次及以上异常）时触发一次。"""
        if self._anomaly_streak >= 2 and not self._review_done:
            self._review_done = True
            return True
        return False

    async def _llm_review(self, query: str, trace_text: str) -> str | None:
        """独立 LLM 复盘：针对异常轨迹给出纠偏建议（注入为 Observation 提示）。"""
        try:
            system = "你是推理过程审查助手。请审视下面的 Agent 推理轨迹，判断其是否偏离用户原始问题、是否存在无效/重复步骤，并给出不超过 100 字的中文纠偏建议。"
            user = f"用户原始问题：{query}\n\n推理轨迹：\n{trace_text[-3000:]}\n\n请给出纠偏建议："
            resp = await self.llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ]
            )
            s = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            return s[:300] if s else None
        except Exception as e:
            logger.warning(f"独立 LLM 复盘失败: {e}")
            return None

    # ====== 上下文压缩 ======

    def _render_round(self, r: dict) -> str:
        return (
            f"<thought>{r.get('thought', '')}</thought>\n"
            f"<action>{r.get('action', '')}</action>\n"
            f"<input>{r.get('input', '')}</input>\n"
            f"<observation>{r.get('observation', '')}</observation>"
        )

    def _render_trace(self) -> str:
        parts = []
        if self._trace_summary:
            parts.append(f"（历史推理摘要：{self._trace_summary}）")
        for r in self._trace:
            parts.append(self._render_round(r))
        return "\n\n".join(parts)

    def _render_trace_latest_only(self) -> str:
        """单臂 ReAct 改造专用：每轮仅渲染「最近 2 轮」的 thought/action/observation，
        不重发历史全量轨迹（prefill 从 O(轮数) 降为 O(常数)）。

        历史事实的连续性由 _build_system_message 中的常驻 AgentState 要点块承载，
        因此此处无需保留早期轮次原文；保留最近 2 轮可在「上一步」紧邻当前步时
        （如 calculator 的 expr 与结果、连续两次检索）减少重复调用与事实丢失。
        - 正常（压缩未触发）：_trace_summary 为空，仅渲染最近 2 轮。
        - 极端（压缩曾触发）：保留 _trace_summary 前缀作早期事实保险，
          仍只渲染最近 2 轮（_trace 已被折叠为最近 keep 轮，取最新 2 轮即可）。
        """
        if not self._trace and not self._trace_summary:
            return "（暂无，这是第一轮推理）"
        parts = []
        if self._trace_summary:
            parts.append(f"（历史推理摘要：{self._trace_summary}）")
        if self._trace:
            # [2026-10-13] 由「仅最新 1 轮」改为「最近 2 轮」，缓解多步工具链的上文丢失
            for r in self._trace[-2:]:
                parts.append(self._render_round(r))
        return "\n\n".join(parts)

    def _render_kb_sources(self) -> str:
        """与 FC 臂对齐：把已成功的 knowledge_search 的 observation 完整重渲染，常驻注入 system。

        为什么要单独做：ReAct 走 latest-only trace（_render_trace_latest_only），早期轮次的
        KB 全文在后续轮会被丢弃；AgentState 的 known_facts 现已不再截断（完整结果原样留存），
        但 latest-only trace 仍只保留最近 2 轮，更早的 KB 原文需靠本区块全量重渲染补全。
        这里从 _tool_results 取 *全量* result 原样输出，不做截断。

        成本有界：KB 通常只调用 1~2 次（重复检索闸已拦截），故每轮仅多出约 1~2 段
        检索原文；极端多跳场景用「最近 4 次」上限保护，避免无限膨胀。
        """
        trs = [
            tr
            for tr in getattr(self, "_tool_results", [])
            if tr.get("tool_name") == "knowledge_search" and tr.get("status") == "success"
        ]
        if not trs:
            return ""
        # 多跳极端保护：仅保留最近 4 次检索原文，避免无限膨胀
        trs = trs[-4:]
        lines = ["## 检索原文（完整保留，供推理引用，请勿凭记忆补全或编造）"]
        for i, tr in enumerate(trs, 1):
            q = (tr.get("tool_args") or {}).get("query", "")
            res = (tr.get("result", "") or "").strip()
            lines.append(f"### 检索 {i}：「{q}」")
            lines.append(res)
        return "\n\n".join(lines)

    def _append_trace(self, thought: str, action: str, inp: str, observation: str) -> None:
        self._trace.append(
            {
                "thought": thought,
                "action": action,
                "input": inp,
                "observation": observation,
            }
        )

    async def _maybe_compress_context(self) -> None:
        """长度兜底压缩（fallback）——与「轮间状态归集 / AgentState 确定性摘要」是两套
        **方向相反**的机制，必须解耦：

        - 常驻机制：_build_system_message 每轮都注入 AgentState.render_for_prompt() 要点表，
          与轨迹长度无关，让 LLM 始终知道当前掌握的事实/计算值/待补齐项（memory abstraction）。
        - 兜底机制（本函数）：仅当历史推理轨迹的 token 量真的超过 trigger_threshold 时才折叠，
          把早期轮次原文压成 _trace_summary 以省 prefill，避免超长单轮 Observation 或多轮累积
          撑爆上下文。它不是常驻，不能靠「轮数 > keep」每轮触发。

        触发条件：estimate_tokens(_render_trace()) > trigger_threshold（长度阈值，非轮数）。
        触发后保留最近 keep_full_steps 轮原文；若折叠后仍超阈值（极长单轮 Observation），
        进一步激进裁剪到仅留最后 1 轮。
        摘要源复用 AgentState 确定性要点（零 LLM 开销、数字不丢）；无 AgentState 实例时给占位说明。
        适用范围：ReAct 与 FC 共用此函数，且均经 _render_trace()（会前置 _trace_summary）；
        LG 走 LangGraph 原生消息历史，不在此列。
        """
        cfg = self._context_cfg
        trigger = cfg.get("trigger_threshold", 20000)
        keep = cfg.get("keep_full_steps", 2)

        # 仅当轨迹长度超过阈值才压缩——这是「兜底」，不是「常驻」。
        # 绝不按轮数每轮触发，否则就与上面的 AgentState 常驻机制重叠、失去区分意义。
        if estimate_tokens(self._render_trace()) <= trigger:
            return

        kept = self._trace[-keep:]

        # 确定性摘要：复用 AgentState（已汇总全部 _tool_results 的已知事实/计算值/待补齐），
        # 不调用 LLM，零开销且数字不丢。与常驻要点表内容同源，但此处仅在兜底触发时才出现于
        # 轨迹块内，属「冗余但无害」的事实保险。
        state_obj = getattr(self, "_state", None)
        if state_obj is not None:
            self._trace_summary = state_obj.render_for_prompt()
        else:
            self._trace_summary = "（前文推理过程已折叠为状态摘要）"
        self._trace = kept

        # 折叠后仍超阈值（极长单轮 Observation）：进一步激进裁剪，仅留最后 1 轮
        if estimate_tokens(self._render_trace()) > trigger:
            self._trace = self._trace[-1:]
            if not self._trace_summary:
                self._trace_summary = "（前文推理过程已折叠为状态摘要）"

    async def _llm_summarize(self, text: str) -> str | None:
        try:
            system = "你是专业的推理过程摘要助手。"
            user = (
                "请用一段不超过 200 字的中文，概括以下 Agent 推理过程的关键信息"
                "（已检索到的核心数据、已完成的精确计算、重要结论），忽略重复与无效步骤：\n\n"
                + text[:4000]
            )
            resp = await self.llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ]
            )
            s = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            return s[:600] if s else None
        except Exception as e:
            logger.warning(f"推理轨迹摘要失败: {e}")
            return None

    # ====== LLM 获取 ======

    def _get_llm(self, llm_provider: str | None):
        if not llm_provider or llm_provider == self.default_provider:
            return self.llm
        if llm_provider in self._llm_cache:
            return self._llm_cache[llm_provider]
        llm = model_factory.get_llm_model(provider=llm_provider, temperature=self.temperature)
        self._llm_cache[llm_provider] = llm
        return llm

    def _accumulate_usage(self, resp) -> None:
        """
        累积本轮 LLM 调用的 token 用量。

        兼容两种 usage_metadata 形态：dict（LangChain TypedDict）与对象，
        此前只处理 dict，遇到对象形态会静默统计成 0。
        """
        um = getattr(resp, "usage_metadata", None)
        if um:
            self._usage["input_tokens"] += _usage_get(um, "input_tokens")
            self._usage["output_tokens"] += _usage_get(um, "output_tokens")
            total = _usage_get(um, "total_tokens", None)
            if total is None:
                total = _usage_get(um, "input_tokens") + _usage_get(um, "output_tokens")
            self._usage["total_tokens"] += total or 0
        self._usage["llm_calls"] = self._llm_calls

    # ====== 主循环：流式 ======

    async def run_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        llm_provider: str | None = None,
        usage_collector: dict | None = None,
    ):
        """
        流式输出 Agent 对话，产出结构化事件流（JSON 字符串）。

        事件格式：
        - {"type": "thinking", "step": N, "content": "...", "tools": [...], "phase": "thought|action|final"}
        - {"type": "tool_result", "step": N, "tool": "...", "content": "...", "phase": "observation"}
        - {"type": "answer", "content": "最终答案"}
        """
        import json as json_mod

        llm = self._get_llm(llm_provider)

        # 每轮调用前初始化运行状态
        self._trace: list[dict] = []
        self._trace_summary: str = ""
        self._recent_actions: list[tuple[str, str]] = []
        self._recent_observations: list[tuple[str, str]] = []
        self._llm_calls = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        self._called_tools: list[str] = []
        self._tool_calls_detail: list[dict] = []  # 工具调用明细（含失败/空结果），用于评测可见性
        self._tool_results: list[dict] = []
        self._calculator_returned_number: bool = False  # 心算闸豁免：calculator 是否成功返回数值
        self._has_successful_retrieval: bool = False  # 全程是否有任一工具成功返回有效数据
        self._failed_calls: list[
            tuple[str, str]
        ] = []  # 记录失败过的 (工具, 参数)，用于禁止原样重试
        # 成功检索过的「归一化查询」历史（用于重复检索闸）：仅记录成功返回的检索，
        # 因此同一查询被重复发起时可直接判定为纯浪费并拦截，避免 ReAct 反复重发大 Observation。
        self._successful_retrieval_queries: list[str] = []
        self._anomaly_streak = 0
        self._review_done = False
        self._pending_review: str | None = None
        self._loop_intervention_count: dict[str, int] = {}
        # 多轮对话历史：评测/生产通过 run_stream(history=...) 传入的上一轮 user/assistant，
        # 之前该参数被接收却从未注入 prompt，导致 ld_003/ld_005 等多轮用例看不到前文。
        self._conversation_history: list[dict] = list(history) if history else []

        # 轻量轮间状态（代码确定性抽取，非 LLM 维护；详见 agent_state.py）
        self._state = AgentState()
        self._state.reset(query)

        step_counter = 0
        rounds = 0
        final_answer = ""
        stall_streak = 0  # 连续格式提醒（卡住）计数，达到阈值触发携上下文强制收尾

        # 注入独立 LLM 复盘结论（若有），作为下一轮 Observation 提示
        async def _flush_review_if_any():
            if self._pending_review:
                hint = self._pending_review
                self._pending_review = None
                self._append_trace(
                    thought="（系统复盘）", action="（复盘建议）", inp="", observation=hint
                )
                return hint
            return None

        while True:
            rounds += 1
            if rounds > MAX_REACT_ROUNDS + 1:
                # 安全网：理论上不会到达（MAX_REACT_ROUNDS+1 轮会强制总结）
                final_answer = "抱歉，推理步数超出上限，仍未能得出确切答案。"
                break

            force_final = rounds == MAX_REACT_ROUNDS + 1
            embed_review = self._need_embed_review(rounds)

            # 若有复盘结论，先作为 Observation 提示注入轨迹
            await _flush_review_if_any()

            # 轮间状态归集：基于上一轮工具结果确定性重建 state（无需 reducer），
            # 随后注入本轮回 system message（见 _build_system_message）。
            self._state.update(self._tool_results, query)
            system = self._build_system_message(query, embed_review, force_final)

            self._llm_calls += 1
            try:
                resp = await llm.ainvoke(
                    [SystemMessage(content=system)],
                    stop=REACT_STOP_SEQUENCES,
                )
                raw = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                logger.error(f"LLM 调用失败 (第{rounds}轮): {e}")
                final_answer = "抱歉，模型调用出现异常，暂时无法完成推理。"
                break
            self._accumulate_usage(resp)
            raw = raw or ""

            # 打印 LLM 原始输出，便于排查格式/循环问题
            logger.info(
                f"[ReAct LLM raw output] round={rounds} len={len(raw)}\n{raw}\n[/ReAct LLM raw output]"
            )

            parse = self._parse_react(raw)

            # ---- 答案心算校验闸：计算题若未调 calculator 却输出数值，强制提醒 ----
            if (
                not force_final
                and parse.is_final
                and self._should_enforce_calculator(query, parse.answer or "")
            ):
                self._anomaly_streak += 1
                hint = (
                    "数值计算结果必须通过 calculator 工具得出，禁止心算输出未经验证的数值，"
                    "请调用计算器后再作答。"
                )
                self._append_trace(
                    thought=parse.thought or "（系统校验）",
                    action="（计算校验）",
                    inp="",
                    observation=hint,
                )
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": "system",
                        "content": f"Observation: [干预] {hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # ---- 强制终止轮 ----
            if force_final:
                if (
                    parse.is_final
                    and parse.answer
                    and parse.answer.strip()
                    and not self._is_degenerate_answer(self._normalize_answer(parse.answer))
                ):
                    answer = parse.answer.strip()
                else:
                    # 模型未给出干净的最终答案（思考步骤 / 格式指令复读）→ 额外发起一轮严格收尾
                    clean = await self._request_clean_final(query, llm)
                    answer = (
                        clean or raw.strip() or "根据已有信息仍无法得出确切结论，请补充更多上下文。"
                    )
                # 安全网：全程无任何工具返回有效数据，但模型仍试图输出具体答案 →
                # 视为潜在编造，强制替换为拒答模板（防幻觉硬兜底）。
                if not self._has_successful_retrieval:
                    _refuse_kw = (
                        "无法提供",
                        "无法回答",
                        "未找到",
                        "不知道",
                        "不能提供",
                        "不存在",
                        "未提及",
                        "未查询到",
                        "暂无相关信息",
                        "没有相关信息",
                    )
                    if not any(kw in answer for kw in _refuse_kw):
                        answer = KB_REJECT_TEMPLATE
                # 输出最终 thought（若有）+ Final Answer
                if parse.thought:
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "thinking",
                            "step": step_counter,
                            "content": parse.thought,
                            "tools": [],
                            "phase": "thought",
                        },
                        ensure_ascii=False,
                    )
                final_answer = self._normalize_answer(answer)
                if self._admits_gap_with_speculation(final_answer):
                    final_answer = KB_REJECT_TEMPLATE
                break

            # ---- 正常最终答案 ----
            if parse.is_final:
                # 先清洗再判退化：清洗（去 thought:/最终答案: 泄漏行、标签残留）后若仍有
                # 面向用户的实质内容则直接采用；仅当清洗后为空/仍退化才走格式提醒。
                # 例：「thought: 信息已足够\n星辰科技Q2销售额3850万」→ 清洗后直接得干净答案。
                cleaned = self._normalize_answer(parse.answer or "")
                if not cleaned.strip() or self._is_degenerate_answer(cleaned):
                    self._anomaly_streak += 1
                    hint = (
                        "你输出的不是面向用户的最终答案，而是思考步骤或格式说明。"
                        "请直接输出最终结论：只写自然语言答案，不要包含 thought/action/input 标签或步骤描述。"
                    )
                    self._append_trace(
                        thought=parse.thought, action="（格式提醒）", inp="", observation=hint
                    )
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": "system",
                            "content": f"Observation: [干预] {hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue
                # 拒答模板强制：一边承认知识库未覆盖、一边用常识推测补全的混合答案
                # → 整体替换为拒答模板（禁止「未找到+推测内容」混合输出）。
                if self._admits_gap_with_speculation(cleaned):
                    cleaned = KB_REJECT_TEMPLATE
                if parse.thought:
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "thinking",
                            "step": step_counter,
                            "content": parse.thought,
                            "tools": [],
                            "phase": "thought",
                        },
                        ensure_ascii=False,
                    )
                final_answer = cleaned
                break

            # ---- 输出 Thought 步骤 ----
            if parse.thought:
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "thinking",
                        "step": step_counter,
                        "content": parse.thought,
                        "tools": [],
                        "phase": "thought",
                    },
                    ensure_ascii=False,
                )

            # ---- 格式提醒（卡住） ----
            if parse.need_reminder:
                self._anomaly_streak += 1
                stall_streak += 1
                # 硬终止：连续 2 次格式提醒仍卡住（典型：反复 none+无答案的死循环）→
                # 携完整上下文（KB 原文+轨迹）强制收尾，避免 7B 烧到 MAX_REACT_ROUNDS。
                if stall_streak >= 2:
                    clean = await self._request_clean_final(query, llm)
                    if clean:
                        # 与 force_final 同款防幻觉安全网：全程无有效检索却给出具体答案 → 拒答模板
                        if not self._has_successful_retrieval:
                            _refuse_kw = (
                                "无法提供",
                                "无法回答",
                                "未找到",
                                "不知道",
                                "不能提供",
                                "不存在",
                                "未提及",
                                "未查询到",
                                "暂无相关信息",
                                "没有相关信息",
                            )
                            if not any(kw in clean for kw in _refuse_kw):
                                clean = KB_REJECT_TEMPLATE
                        logger.info(f"[格式提醒硬终止] 连续 {stall_streak} 次卡住，强制收尾成功")
                        final_answer = clean
                        break
                hint = (
                    "你没有按格式输出下一步。请严格遵循："
                    "如需调用工具则输出 <thought>分析</thought><action>工具名</action><input>参数</input>；"
                    "若已获得足够信息，输出 <thought>说明已经收集到足够信息，可以给出答案</thought> 并紧跟最终答案；"
                    "若你输出 <action>none</action> 表示无需工具，必须在其后直接写出最终答案文本，不能只写 none。"
                )
                self._append_trace(
                    thought=parse.thought, action="（格式提醒）", inp="", observation=hint
                )
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": "system",
                        "content": f"Observation: [干预] {hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # 走到这里说明本轮不是卡住（正常工具步或最终答案）→ 连续卡住计数清零
            stall_streak = 0

            action = parse.action
            raw_input = parse.tool_input or ""

            # ---- 重复调用检测 ----
            if self._is_repeat_loop(action, raw_input):
                self._anomaly_streak += 1
                self._loop_intervention_count[action] = (
                    self._loop_intervention_count.get(action, 0) + 1
                )
                hint = (
                    f"检测到你连续多次以高度相似参数调用工具 '{action}'，可能陷入循环。"
                    "请基于已获取的信息继续推进或更换检索/计算角度；若信息已充足，请直接给出最终答案。"
                )
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=hint
                )
                self._recent_actions.append((action, raw_input))
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )

                # 硬兜底：同一工具连续被干预 2 次仍不收敛，强制进入最终总结
                if self._loop_intervention_count.get(action, 0) >= 2:
                    logger.warning(f"[ReAct] 工具 '{action}' 循环干预达到上限，强制最终总结")
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "thinking",
                            "step": step_counter,
                            "content": f"检测到工具 '{action}' 持续循环，系统强制进入最终总结。",
                            "tools": [],
                            "phase": "thought",
                        },
                        ensure_ascii=False,
                    )
                    final_system = self._build_system_message(query, False, True)
                    self._llm_calls += 1
                    try:
                        resp_final = await llm.ainvoke(
                            [SystemMessage(content=final_system)],
                            stop=["<action>", "<observation>"],
                        )
                        raw_final = (
                            resp_final.content
                            if hasattr(resp_final, "content")
                            else str(resp_final)
                        )
                        self._accumulate_usage(resp_final)
                        logger.info(
                            f"[ReAct LLM final summary raw]\n{raw_final}\n[/ReAct LLM final summary raw]"
                        )
                        parse_final = self._parse_react(raw_final)
                        if (
                            parse_final.is_final
                            and parse_final.answer
                            and parse_final.answer.strip()
                        ):
                            final_answer = self._normalize_answer(parse_final.answer.strip())
                        else:
                            # 退化兜底：若模型仍在输出工具调用步骤，只取最后一个
                            # </observation> 之后的文本作为最终答案，避免把整段
                            # 重复的工具调用噪声当答案返回（造成关键词假成功）。
                            fallback_text = raw_final.strip()
                            if "<action>" in fallback_text or re.search(
                                r"<observation>", fallback_text, re.IGNORECASE
                            ):
                                segs = re.split(
                                    r"</observation>", fallback_text, flags=re.IGNORECASE
                                )
                                fallback_text = segs[-1].strip()
                            final_answer = (
                                self._normalize_answer(fallback_text)
                                or "检测到工具调用循环，基于当前信息无法给出更确切的答案。"
                            )
                    except Exception as e:
                        logger.error(f"[ReAct] 强制最终总结失败: {e}")
                        final_answer = "检测到工具调用循环，基于当前信息无法给出更确切的答案。"
                    break

                await self._maybe_compress_context()
                continue

            # ---- 工具存在性校验 ----
            tool = self._find_tool(action)
            if tool is None:
                self._anomaly_streak += 1
                avail = ", ".join(self._tool_map.keys()) or "（无）"
                hint = (
                    f"工具 '{action}' 不在可用工具列表中（可用：{avail}）。"
                    "请只使用可用工具，或基于已有信息直接作答。"
                )
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=hint
                )
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # ---- 文件读取后重复验证校验闸 ----
            redundant, redundant_hint = self._is_redundant_file_calc(action, raw_input)
            if redundant:
                self._anomaly_streak += 1
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=redundant_hint
                )
                self._recent_actions.append((action, raw_input))
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {redundant_hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # ---- 重复检索闸（专治 round3/round4 类「相同查询被成功检索两次」） ----
            dup_ret, dup_ret_hint = self._is_duplicate_retrieval(action, raw_input)
            if dup_ret:
                self._anomaly_streak += 1
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=dup_ret_hint
                )
                self._recent_actions.append((action, raw_input))
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {dup_ret_hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # ---- 多轮重复调用校验闸 ----
            dup, dup_hint = self._is_duplicate_tool_call(action, raw_input)
            if dup:
                self._anomaly_streak += 1
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=dup_hint
                )
                self._recent_actions.append((action, raw_input))
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {dup_hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # ---- 执行工具 ----
            result, obs_text, parsed_args = await self._execute_tool(tool, raw_input)

            # 捕获「模型原始检索意图」：下方空/无关分支可能改写 raw_input 重试，
            # 但重复检索闸应基于模型最初想查的查询比对，故在此处（改写前）留存。
            intended_query = raw_input

            # 记录本次观测，供「结果停滞 / 无新增有效信息」判定复用
            self._recent_observations.append((action, obs_text or ""))

            # ---- knowledge_search 返回空/无关 → 换关键词重试（最多 3 次），仍空才返回拒绝模板 ----
            if action == "knowledge_search" and self._kb_result_empty_or_irrelevant(
                result, obs_text
            ):
                # 记录 LLM 发起的这一次检索（用于工具调用统计 / 防幻觉诊断）
                self._called_tools.append(action)
                self._tool_calls_detail.append(
                    {"tool_name": action, "args": raw_input, "status": "empty_or_irrelevant"}
                )

                # 自动用改写后的关键词重试（查询扩展 / HyDE / 原问法回退，最多 3 次）
                found, f_res, f_obs, f_q = await self._kb_retry_with_rewrite(
                    tool, parsed_args, query
                )

                if found:
                    # 命中：替换为重试结果，落到下方「成功」分支继续正常推理
                    result = f_res
                    obs_text = f_obs
                    try:
                        parsed_args = dict(parsed_args)
                        parsed_args["query"] = f_q
                        raw_input = json_mod.dumps(parsed_args, ensure_ascii=False)
                    except Exception:
                        pass
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "thinking",
                            "step": step_counter,
                            "content": f"首次检索未命中，已自动用改写关键词「{f_q}」重试并找到相关内容。",
                            "tools": [{"name": action, "args": parsed_args}],
                            "phase": "thought",
                        },
                        ensure_ascii=False,
                    )
                    # 不 break：继续走后续「成功」处理（记录 tool_result / 输出 Observation）
                else:
                    # 3 次关键词重试仍为空/无关 → 返回拒绝模板（防幻觉）
                    self._tool_results.append(
                        {
                            "tool_name": action,
                            "tool_args": parsed_args,
                            "tool_call_id": "",
                            "status": "success",
                            "result": obs_text,
                            "error": None,
                        }
                    )
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "thinking",
                            "step": step_counter,
                            "content": "知识库检索（含 3 次关键词改写重试）均未返回相关内容，按规则直接返回拒绝模板。",
                            "tools": [{"name": action, "args": parsed_args}],
                            "phase": "thought",
                        },
                        ensure_ascii=False,
                    )
                    final_answer = KB_REJECT_TEMPLATE
                    break

            # ---- 空/无效结果检测 ----
            if self._is_empty_result(result, obs_text):
                self._anomaly_streak += 1
                detail = ""
                if result and not getattr(result, "success", True) and getattr(result, "error", ""):
                    detail = f"（错误详情：{result.error}）"
                elif obs_text and "错误" in obs_text:
                    detail = f"（{obs_text}）"
                # 工具返回非成功时禁止原样重试：把拒绝原因回灌 + 明确要求换写法。
                # 同一 (工具, 参数) 已失败过则强制换思路/拒答，禁止反复用相同参数空转。
                if (action, raw_input) in self._failed_calls:
                    hint = (
                        f"工具 '{action}' 以完全相同的参数已失败过{detail}。"
                        "禁止原样重复调用。请改用不同的表述/参数/文件路径，或换用其他工具；"
                        "若确实无法获取任何有效数据，必须直接按拒答模板回应，禁止凭空编造或猜测。"
                    )
                else:
                    hint = (
                        f"工具 '{action}' 返回结果为空或无效{detail}。"
                        "请把上面的错误原因回灌，改用不同的查询参数/表述重试；"
                        "若确实无法获取任何有效数据，必须直接按拒答模板回应，禁止基于已有信息猜测或编造。"
                    )
                self._failed_calls.append((action, raw_input))
                self._append_trace(
                    thought=parse.thought, action=action, inp=raw_input, observation=hint
                )
                self._recent_actions.append((action, raw_input))
                self._tool_calls_detail.append(
                    {
                        "tool_name": action,
                        "args": raw_input,
                        "status": "failed",
                        "error": detail,
                    }
                )
                step_counter += 1
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": action,
                        "content": f"Observation: [干预] {hint}",
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )
                await self._maybe_compress_context()
                continue

            # 成功
            self._anomaly_streak = 0
            self._has_successful_retrieval = True
            if action == "calculator" and getattr(result, "success", False):
                # 心算闸豁免：calculator 成功返回过结果（含数字列表等数值型结果），
                # 无需再强制调用；仅在 calculator 调用失败/未调用时才触发心算闸。
                self._calculator_returned_number = True
            self._called_tools.append(action)
            self._tool_calls_detail.append(
                {"tool_name": action, "args": raw_input, "status": "success"}
            )
            self._tool_results.append(
                {
                    "tool_name": action,
                    "tool_args": parsed_args,
                    "tool_call_id": "",
                    "status": "success",
                    "result": obs_text,
                    "error": None,
                }
            )

            # 重复检索闸：记录成功检索的归一化查询（基于模型原始意图），供后续轮次比对拦截
            if action == "knowledge_search":
                norm_q = self._norm_retrieval_query(action, intended_query)
                if norm_q and norm_q not in self._successful_retrieval_queries:
                    self._successful_retrieval_queries.append(norm_q)

            # Observation 预处理
            obs_text = await self._preprocess_observation(obs_text, query)

            # 输出 Action 步骤
            step_counter += 1
            action_content = f"调用 {action}" + (f"，参数：{raw_input}" if raw_input else "")
            yield json_mod.dumps(
                {
                    "type": "thinking",
                    "step": step_counter,
                    "content": action_content,
                    "tools": [{"name": action, "args": parsed_args}],
                    "phase": "action",
                },
                ensure_ascii=False,
            )

            # 输出 Observation 步骤
            step_counter += 1
            obs_content = f"Observation: [{action}] {obs_text}"
            yield json_mod.dumps(
                {
                    "type": "tool_result",
                    "step": step_counter,
                    "tool": action,
                    "content": obs_content,
                    "phase": "observation",
                },
                ensure_ascii=False,
            )

            # 写入历史推理轨迹
            self._append_trace(
                thought=parse.thought, action=action, inp=raw_input, observation=obs_text
            )
            self._recent_actions.append((action, raw_input))

            # 独立 LLM 复盘（仅在异常累积时触发一次）
            if self._need_llm_review():
                review = await self._llm_review(query, self._render_trace())
                if review:
                    self._pending_review = review

            # 上下文压缩
            await self._maybe_compress_context()

        # 兜底：若未拿到答案
        if not final_answer:
            final_answer = "抱歉，我暂时无法生成完整回答。请稍后重试或换一种问法。"

        # 归一化：去除答案可能自带的 "Final Answer:" 前缀
        final_answer = self._normalize_answer(final_answer)

        # 记录最终结论到轮间 state（供多轮对话下一轮复用 / 调试回溯）
        if getattr(self, "_state", None) is not None and final_answer:
            self._state.conclusion = final_answer

        # 产出 Final Answer 思考步骤（徽章由前端 phase:"final" 渲染为 "Final Answer"，
        # content 直接为答案本身，不再补 "Final Answer:" 前缀，避免与徽章重复）
        step_counter += 1
        yield json_mod.dumps(
            {
                "type": "thinking",
                "step": step_counter,
                "content": final_answer,
                "tools": [],
                "phase": "final",
            },
            ensure_ascii=False,
        )

        # 产出最终答案事件
        yield json_mod.dumps({"type": "answer", "content": final_answer}, ensure_ascii=False)

        if usage_collector is not None:
            usage_collector["usage"] = self._usage

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """
        归一化最终答案：
        - 去除可能自带的 'Final Answer:' 前缀；
        - 去除可能混入的 <thought>/<action>/<input>/<observation> 标签（模型偶发泄漏）；
        - 清洗内部字段名（answer/called_tools/tool_results 等）及 JSON 结构残留。
        """
        if not answer:
            return answer
        text = re.sub(r"^\s*Final\s*Answer\s*[:：]\s*", "", answer, flags=re.IGNORECASE)
        # 去除模型可能输出的 "最终的完整答案[:：]" 前缀
        text = re.sub(r"^\s*最终的完整答案\s*[:：]?\s*", "", text)
        # 去除 "最终答案[:：]" 前缀（7B 偶发把模板占位符写进正文）
        text = re.sub(r"^\s*最终答案\s*[:：]\s*", "", text)
        # 去除完整的 <thought>...</thought> 块
        text = re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL)
        # 去除残留的孤立标签
        text = re.sub(r"</?(?:thought|action|input|observation)>", "", text, flags=re.IGNORECASE)
        # 去除包含内部字段名的整行（如 answer: xxx、called_tools: [...]、
        # 以及 7B 泄漏进最终答案的元话语行「thought: 信息已足够，可以直接给出答案」）
        text = re.sub(
            r"(?im)^\s*(?:answer|thought|called_tools|tool_results|tool_name|tool_args|status|error)\s*[:：].*$",
            "",
            text,
        )
        # 去除孤立的 JSON 大括号残留
        text = re.sub(r"\{\s*\}", "", text)
        # 去除 answer/called_tools 等字段开头到下一个 } 的 JSON 片段
        text = re.sub(
            r'\{\s*"(?:answer|called_tools|tool_results|tool_name|tool_args|status|error)"[\s\S]*?\}',
            "",
            text,
        )
        # 剥离 system 提示里的「最终答案内容」占位符被模型原样复读到答案中
        text = re.sub(r"^\s*最终答案内容\s*[:：]?\s*", "", text)
        # 清洗后为空 → 返回空串（调用方走「格式提醒/兜底致歉」链路），
        # 不再回退返回原文：此时原文是纯元话语/结构残留（如「thought: 信息已足够」）。
        return text.strip()

    # 「指令复读 / 过程文本」误当最终答案的识别（7B 偶发的格式退化）。
    _ECHO_MARKERS = (
        "说明下一步要做什么",
        "说明已经收集到足够信息",
        "可以给出答案",
        "最终答案内容",
        "请直接给出最终答案",
        "当你认为信息足够",
        "使用以下格式输出最终答案",
        "思考当前问题",
        "分析当前问题",
        "下一步要做什么",
        "为什么选这个工具",
        "调用calculator完成计算",
        "调用计算器完成计算",
    )

    @staticmethod
    def _is_degenerate_answer(text: str) -> bool:
        """识别把「思考步骤 / 格式指令复读」当成最终答案的退化输出。"""
        if not text or not text.strip():
            return True
        t = text.lower()
        if any(m.lower() in t for m in SmartAgent._ECHO_MARKERS):
            return True
        # 裸格式元词反复出现：thought/action/input 作为英文单词（非标签）
        if re.search(r"\bthought\b", t) and re.search(r"\baction\b", t):
            return True
        if re.search(r"action\s*(calculator|knowledge|file|oa)", t):
            return True
        if "action" in t and "input" in t and ("calculator" in t or "knowledge" in t):
            return True
        # 流程元话语（"无需调用工具/可以直接引用原文作答"等）不是面向用户的答案
        if re.search(r"(无需调用|不需要调用|无需再调用|不必调用)[^。]{0,8}(工具|检索)", t):
            return True
        if re.search(r"可以直接(引用原文|作答|回答|给出答案)", t):
            return True
        # 以 thought[:：] 开头的元话语行（清洗后仍残留 = 整行无真实内容）
        if re.match(r"^\s*thought\s*[:：]", t):
            return True
        return False

    @staticmethod
    def _admits_gap_with_speculation(text: str) -> bool:
        """识别「承认知识库未覆盖 + 常识推测补全」的混合答案。

        例："公司食堂通常会在工作日提供午餐和晚餐服务，但具体菜单内容未在知识库中直接提及。"
        此类答案应整体替换为拒答模板，禁止模型用推测性内容填补知识库空白。
        需同时命中「未覆盖承认」与「推测措辞」两类信号，避免误伤完全基于检索的正常回答。
        """
        if not text:
            return False
        return bool(_KB_GAP_ADMISSION_RE.search(text) and _SPECULATION_RE.search(text))

    async def _request_clean_final(self, query: str, llm) -> str:
        """force_final / 卡死硬终止兑底：模型未给出干净答案时，携完整上下文额外发起一轮严格收尾。

        必须携带完整 system 上下文（KB 检索原文 + 轮间状态 + 最新轨迹），
        否则模型拿不到任何依据，只能产出泛泛的幻觉性"答案"。
        要求只输出面向用户的自然语言结论，禁止任何格式标签 / 步骤说明。
        """
        system = self._build_system_message(query, False, True) + (
            "\n\n现在只做一件事：输出面向用户的最终答案。"
            "要求：1）只输出 1-3 句自然语言结论；2）绝对不要输出 thought/action/input 等任何格式标签；"
            "3）不要复述思考过程或步骤说明；4）若知识库未覆盖问题，直接使用统一拒答句式，"
            "禁止基于常识推测补全。"
        )
        try:
            resp = await llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=f"用户问题：{query}\n请直接输出最终答案。"),
                ],
                stop=REACT_STOP_SEQUENCES,
            )
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return ""
        text = self._normalize_answer((text or "").strip())
        return text if text and not self._is_degenerate_answer(text) else ""

    # ====== 主循环：非流式 ======

    async def run(
        self,
        query: str,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        """
        执行 Agent 对话（非流式）。

        Returns:
            {
                "answer": str,
                "called_tools": list,
                "tool_results": list,
                "messages": list,
                "usage_metadata": dict,
            }
        """
        thinking_steps = []
        answer = ""

        async for chunk in self.run_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            history=history,
            llm_provider=llm_provider,
        ):
            try:
                import json as json_mod

                event = json_mod.loads(chunk)
                if event.get("type") == "answer":
                    answer = event.get("content", "")
                elif event.get("type") in ("thinking", "tool_result"):
                    thinking_steps.append(event)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "answer": answer,
            "called_tools": self._called_tools,
            "tool_calls_detail": self._tool_calls_detail,
            "tool_results": self._tool_results,
            "messages": [],
            "usage_metadata": self._usage,
        }

    def get_tools_info(self) -> list[dict]:
        """获取当前 Agent 绑定的工具信息列表"""
        result = []
        for tool in self.base_tools:
            result.append(
                {
                    "name": getattr(tool, "name", tool.__class__.__name__),
                    "description": getattr(tool, "description", ""),
                }
            )
        return result


# ====== 全局实例 ======

_smart_agent_instance = None


def get_smart_agent(llm_provider: str = None):
    global _smart_agent_instance
    if _smart_agent_instance is None:
        _smart_agent_instance = SmartAgent(llm_provider=llm_provider)
    elif llm_provider is not None and _smart_agent_instance.default_provider != llm_provider:
        _smart_agent_instance = SmartAgent(llm_provider=llm_provider)
    return _smart_agent_instance


class _SmartAgentProxy:
    def __getattr__(self, name):
        return getattr(get_smart_agent(), name)


smart_agent = _SmartAgentProxy()
