# backend/app/agent/tools/oa_mock_tool.py
"""
报销 OA 连接器（Mock 版）

演示"连接器"模式：把"对接 OA 系统"封装成一个标准 BaseTool。
本实现为 **Mock**，不调用任何真实 OA，返回确定性假数据，便于：
  - 在缺乏真实 OA 凭证/网络时演示"报销助手模板"完整链路；
  - 部署时直接把 code_path 换成真实 HTTP 连接器（BaseTool 子类或 URL）即可，前端零改动。

支持的操作（action）：
  - submit        : 提交一张报销单，返回假单号 + 模拟审批链
  - query_status  : 按单号查询审批状态
  - list_mine     : 查询某员工名下的待办报销单

注意：本模块不 import 任何会触发 HuggingFace 模型拉取的依赖，可离线运行。
"""

import json
import random
import time
from typing import Any

from backend.app.agent.tools.base import BaseTool, ToolResult


class OaMockTool(BaseTool):
    """
    OA 报销连接器（Mock）

    面向 LLM 的说明见 description；参数通过 parameters_schema 约束。
    """

    name = "oa_reimbursement"
    description = (
        "企业 OA 报销连接器，仅处理报销相关操作。\n"
        "支持三种 action：submit 提交报销单、query_status 按单号查审批状态、list_mine 查某员工待办报销单。\n"
        "仅当已收齐【申请人、事项明细(类别+金额)、合计金额】并已用 calculator 核对合计后才调用 submit；查询类须提供单号或申请人姓名。本工具不处理报销以外的 OA 事务（考勤、请假等）。"
    )
    permission = "safe"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型：submit=提交报销单；query_status=查询审批状态；list_mine=查询我的待办报销单",
            },
            "params": {
                "type": "string",
                "description": (
                    "JSON 字符串参数。submit 示例："
                    '{"applicant":"张三","items":[{"category":"差旅","amount":1200}],"total":1200}；'
                    'query_status 示例：{"ticket_no":"BX20260001"}；'
                    'list_mine 示例：{"applicant":"张三"}'
                ),
            },
        },
        "required": ["action"],
    }

    # 模拟审批链（真实部署时由 OA 返回）
    _APPROVAL_CHAIN = ["直属主管审批", "财务复核", "总经理终审"]

    def _gen_ticket_no(self) -> str:
        """生成确定性风格的假单号（基于时间+随机，仅演示用）"""
        ts = time.strftime("%Y%m%d")
        rand = random.randint(1000, 9999)
        return f"BX{ts}{rand}"

    async def run(self, action: str, params: str = "{}") -> ToolResult:
        try:
            parsed: dict[str, Any] = json.loads(params) if params else {}
        except json.JSONDecodeError:
            return ToolResult(success=False, error="params 不是合法 JSON 字符串")

        action = (action or "").strip().lower()

        if action == "submit":
            return self._submit(parsed)
        elif action == "query_status":
            return self._query_status(parsed)
        elif action == "list_mine":
            return self._list_mine(parsed)
        else:
            return ToolResult(
                success=False,
                error=f"不支持的 action: {action}，仅支持 submit / query_status / list_mine",
            )

    def _submit(self, parsed: dict[str, Any]) -> ToolResult:
        applicant = parsed.get("applicant") or ""
        items = parsed.get("items") or []
        total = parsed.get("total")

        if not applicant:
            return ToolResult(success=False, error="缺少申请人 applicant")
        if not items:
            return ToolResult(success=False, error="缺少报销明细 items")
        if total is None:
            return ToolResult(success=False, error="缺少合计金额 total")

        ticket_no = self._gen_ticket_no()
        data = {
            "ticket_no": ticket_no,
            "applicant": applicant,
            "items": items,
            "total": total,
            "status": "审批中",
            "approval_chain": self._APPROVAL_CHAIN,
            "current_node": self._APPROVAL_CHAIN[0],
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "（Mock）已模拟提交至 OA，真实环境将推送至审批人流转。",
        }
        return ToolResult(
            success=True,
            data=data,
            metadata={"action": "submit", "ticket_no": ticket_no},
        )

    def _query_status(self, parsed: dict[str, Any]) -> ToolResult:
        ticket_no = parsed.get("ticket_no") or ""
        if not ticket_no:
            return ToolResult(success=False, error="缺少单号 ticket_no")

        # Mock：以单号尾数决定状态，便于演示不同分支
        last = ticket_no[-1] if ticket_no else "0"
        if last in "01":
            status, node = "已通过", "已完成"
        elif last in "23":
            status, node = "被驳回", "财务复核"
        else:
            status, node = "审批中", self._APPROVAL_CHAIN[0]

        data = {
            "ticket_no": ticket_no,
            "status": status,
            "current_node": node,
            "approval_chain": self._APPROVAL_CHAIN,
            "note": "（Mock）状态为演示数据，真实环境以 OA 为准。",
        }
        return ToolResult(
            success=True,
            data=data,
            metadata={"action": "query_status", "ticket_no": ticket_no},
        )

    def _list_mine(self, parsed: dict[str, Any]) -> ToolResult:
        applicant = parsed.get("applicant") or ""
        if not applicant:
            return ToolResult(success=False, error="缺少申请人 applicant")

        # Mock：返回固定示例待办
        data = [
            {
                "ticket_no": "BX202601010001",
                "applicant": applicant,
                "total": 1200,
                "status": "审批中",
                "current_node": "财务复核",
            },
            {
                "ticket_no": "BX202601010002",
                "applicant": applicant,
                "total": 350,
                "status": "已通过",
                "current_node": "已完成",
            },
        ]
        return ToolResult(
            success=True,
            data=data,
            metadata={"action": "list_mine", "applicant": applicant},
        )
