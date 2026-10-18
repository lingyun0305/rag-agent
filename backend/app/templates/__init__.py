# backend/app/templates/__init__.py
"""
场景模板注册中心

模板 = 一份**可克隆的 Agent 配置包**：system_prompt + 工具集 + 检索参数 + 模型参数 + 连接器提示。
用户在前端点"从模板创建"，即可一键生成配置好的 Agent，再做少量个性化即可用。

新增模板：往 AGENT_TEMPLATES 追加一个 dict 即可（前端零改动）。
模板引用的工具用 code_path 标识（与系统工具/自定义工具的 code_path 对齐）；
真实连接器到位后，把对应工具的 code_path 从 Python 类换成 HTTP URL 即可。

字段说明：
  id            模板唯一标识
  name          模板显示名
  icon          图标 emoji
  desc          模板简介（画廊卡片用）
  system_prompt 预填给 Agent 的系统提示词
  greeting      对话开场白
  temperature   模型温度
  tool_code_paths 需要挂载的工具 code_path 列表
  tool_configs  工具级配置（如 knowledge_search 的检索参数）
  kb_hint       建议绑定的知识库说明（KB 由用户手动绑定，模板不强制）
  tags          标签（画廊筛选/展示用）
"""

AGENT_TEMPLATES = [
    {
        "id": "reimbursement",
        "name": "报销助手",
        "icon": "💰",
        "desc": "对接 OA 的报销助手：按公司报销制度解答额度/流程问题，帮员工整理明细、核算金额，并提交 OA 发起审批。",
        "system_prompt": (
            "你是企业的【报销助手】，服务于普通员工的日常报销咨询与办理。\n\n"
            "## 你的能力\n"
            "1. 制度问答：优先用「知识库查询」检索《公司报销制度》，回答报销额度、差旅标准、发票要求、审批流程等问题。"
            "引用制度条款时必须标注来源文档名，禁止凭空编造额度数字。\n"
            "2. 金额核算：涉及合计/分摊/税率等计算，必须调用「计算器」工具，不要心算。\n"
            "3. 提交办理：当员工明确要报销、且已提供【申请人姓名、各项报销明细(类别+金额)、合计金额】三项信息后，"
            "先用计算器核算合计，与员工确认无误，再调用「报销OA连接器」的 submit 操作提交，并返回单号与审批链。\n\n"
            "## 工作准则\n"
            "- 信息不全时主动追问：先问清报销事项类别与金额，再决定是否检索/计算/提交。\n"
            "- 提交前务必与用户确认金额与明细，避免误提交。\n"
            "- 只处理报销相关事务；其他 OA 流程（请假、采购等）礼貌告知不在本助手范围。\n"
            "- 所有对外结论以公司制度与 OA 实际返回为准；本连接器为演示环境，提交结果仅为模拟。"
        ),
        "greeting": "您好，我是报销助手。请告诉我需要报销的事项（如差旅、招待、办公采购）及对应金额，我帮您核算并走 OA 提交流程。",
        "temperature": 0.3,
        "tool_code_paths": ["knowledge_search", "calculator", "oa_reimbursement"],
        "tool_configs": {
            "knowledge_search": {
                "top_k": 5,
                "similarity_threshold": 0.0,
                "strategy": "hybrid",
                "use_rerank": True,
                "enable_query_rewrite": False,
            }
        },
        "kb_hint": "建议绑定《公司报销制度》知识库（含差旅标准/审批额度/发票要求）。未绑定也能用，但无法引用具体制度条款。",
        "tags": ["财务", "OA", "报销"],
    },
]
