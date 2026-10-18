# backend/app/agent/tools/mcp_client_demo.py
"""
MCP client demo：以标准 MCP client 身份走一遍协议全流程

    initialize → tools/list → tools/call → result

子进程拉起 mcp_adapter.py（stdio 传输），依次演示：
  1. tools/list        发现 server 自动暴露的全部工具（含 inputSchema）
  2. tools/call        calculator 算一笔账
  3. tools/call        oa_reimbursement submit 提交报销单 → 拿到单号
  4. tools/call        oa_reimbursement query_status 用上一步的单号查状态（跨调用串数据）

运行：
  venv/Scripts/python.exe backend/app/agent/tools/mcp_client_demo.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

_SERVER_PATH = Path(__file__).with_name("mcp_adapter.py")
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _text_of(result) -> str:
    """取 MCP CallToolResult 的文本内容"""
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


async def main() -> None:
    # 以当前 venv 的 python 拉起 server 子进程（stdio 传输）
    # Windows 坑：子进程 stdout 默认 GBK，写中文会让按 UTF-8 读的 client 炸掉 → 强制 UTF-8
    env = get_default_environment()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(_SERVER_PATH)],
        cwd=str(_PROJECT_ROOT),
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. initialize
            init = await session.initialize()
            server_info = init.serverInfo
            print("=" * 62)
            print(f"[initialize] server={server_info.name} v{server_info.version}")
            print(f"             instructions={init.instructions}")

            # 2. tools/list：自动发现全部工具
            tools = await session.list_tools()
            print("\n[tools/list] 发现工具：")
            for t in tools.tools:
                required = list((t.inputSchema or {}).get("required", []))
                print(f"  - {t.name:16s} 必填参数={required or '无'}")
                print(f"    {''.join(t.description.splitlines()[:1])[:70]}")

            #  3. tools/call：calculator
            expr = {"expression": "(128*8 + 512) / 16"}
            res = await session.call_tool("calculator", expr)
            print(f"\n[tools/call] calculator({json.dumps(expr, ensure_ascii=False)})")
            print(f"  → {_text_of(res)}")

            #  4. tools/call：oa submit → 拿单号
            submit_args = {
                "action": "submit",
                "params": json.dumps(
                    {
                        "applicant": "张三",
                        "items": [
                            {"category": "差旅", "amount": 128 * 8},
                            {"category": "办公", "amount": 512},
                        ],
                        "total": (128 * 8 + 512) / 16 * 16,  # 与 calculator 同一笔账
                    },
                    ensure_ascii=False,
                ),
            }
            res = await session.call_tool("oa_reimbursement", submit_args)
            submit_data = json.loads(_text_of(res))
            ticket_no = submit_data["ticket_no"]
            print("\n[tools/call] oa_reimbursement.submit")
            print(
                f"  → 单号={ticket_no} 状态={submit_data['status']} 当前节点={submit_data['current_node']}"
            )

            # 5. tools/call：query_status（复用上一步单号，跨调用串数据）
            res = await session.call_tool(
                "oa_reimbursement",
                {"action": "query_status", "params": json.dumps({"ticket_no": ticket_no})},
            )
            status_data = json.loads(_text_of(res))
            print(f"\n[tools/call] oa_reimbursement.query_status(ticket_no={ticket_no})")
            print(f"  → 状态={status_data['status']} 当前节点={status_data['current_node']}")

            print("\n" + "=" * 62)
            print("MCP 协议全流程走通：initialize → tools/list → tools/call → result ✓")


if __name__ == "__main__":
    asyncio.run(main())
