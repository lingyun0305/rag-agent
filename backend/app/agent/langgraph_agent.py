"""
LangGraph 范式 Agent（与手写 ReAct / 原生 FC 的三方对照实验用）。

设计原则（与 FC 版一致：尽量不改原链路，对照变量唯一 = 编排方式）：
- 继承 FunctionCallingAgent（← SmartAgent），**完整复用** 工具注册表 / 工具执行 /
  guard 判定 / 检索 / 评测 这套已验证的链路，保证三组对照"变量唯一=决策/编排方式"。
- 仅替换「循环控制」这层：
  * 手写 ReAct  = Python while + 正则解析文本
  * 原生 FC     = Python while + 解析 resp.tool_calls
  * LangGraph   = StateGraph 图建模循环（节点/边/Reducer/条件边/compile）
- 图结构（手写而非 create_react_agent 黑盒，便于讲清每个组件）：
  START → agent(节点) ──(有 tool_calls)──▶ tools(节点)
                    ▲                          │
                    └──────────────────────────┘
            (should_continue 条件边：无 tool_calls 或轮数到顶 → END)
- 状态用 LangGraph 的 Reducer（messages: Annotated[list, add_messages]）合并，
  替代手写 ReAct 的 _render_trace 字符串拼接；这正是 agent_state.py 注释里点名的
  "LangGraph state+reducer 模型"——本文件即其对照实现。
- tools 节点复用自己的 _execute_tool + guard 链（而非官方 ToolNode），目的是让对照
  公平：工具层完全一致，差异只在"谁驱动循环"。这也演示了如何把领域 guard 逻辑
  桥接到框架编排层（面试可讲点）。

接入评测：run_eval.py 的 build_eval_agent 按 AGENT_MODE=langgraph 切换返回本类，
复用同一套 run() 返回结构与每用例重置逻辑，产出 evaluation_results_langgraph_*.json。
"""

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from backend.app.agent.agent import KB_REJECT_TEMPLATE, MAX_REACT_ROUNDS
from backend.app.agent.agent_fc import FunctionCallingAgent


class ReActState(TypedDict):
    """LangGraph 维护的状态结构。

    - messages：对话/思考轨迹，由 add_messages reducer 智能合并（追加新消息、
      按 id 更新既有 tool 调用与结果）。这就是 LangGraph 的"Reducer"机制。
    - rounds：循环计数，默认覆盖式更新（无 Annotated → 直接覆盖）。
    """

    messages: Annotated[list, add_messages]
    rounds: int


class LangGraphAgent(FunctionCallingAgent):
    """
    LangGraph 范式 Agent。

    与手写 ReAct / 原生 FC 的唯一区别在于「循环如何驱动」：
    此处由 StateGraph 的节点 + 条件边驱动，state 由 reducer 合并，
    而非手写 while + 每轮代码确定性重建（见 agent_state.py 注释）。
    """

    # 主入口：镜像父类 run() 的返回结构，评测脚本无需改调用方式
    async def run(
        self,
        query: str,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        llm_provider: str | None = None,
    ) -> dict:
        # —— 每用例重置（与 FC 版 run_fc_stream 初始化对齐，避免跨用例污染）——
        self._trace: list[dict] = []
        self._trace_summary: str = ""
        self._recent_actions: list[tuple[str, str]] = []
        self._recent_observations: list[tuple[str, str]] = []
        self._llm_calls = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        self._called_tools: list[str] = []
        self._tool_calls_detail: list[dict] = []
        self._tool_results: list[dict] = []
        self._calculator_returned_number: bool = False
        self._has_successful_retrieval: bool = False
        self._failed_calls: list[tuple[str, str]] = []
        self._anomaly_streak = 0
        self._calc_forced = (
            False  # 本轮是否触发过"强制 calculator"干预（防止空串续写误释放心算答案）
        )
        self._review_done = False
        self._pending_review: str | None = None
        self._loop_intervention_count: dict[str, int] = {}
        self._conversation_history: list[dict] = list(history) if history else []

        # 供图节点闭包访问的查询 / provider
        self._query = query
        self._llm_provider = llm_provider or getattr(self, "llm_provider", None)
        # 占位 _state（本范式不依赖智能体重建逻辑，但父类部分 helper 可能引用）
        try:
            from backend.app.agent.agent_state import AgentState

            self._state = AgentState()
            self._state.reset(query)
        except Exception:  # noqa: BLE001
            self._state = None

        graph = self._build_graph()

        # 初始状态：仅放用户问题（SystemMessage 由 agent 节点按需重建，不入 state 避免重复）
        init_state = {"messages": [HumanMessage(content=query)], "rounds": 0}
        final_state = await graph.ainvoke(init_state)

        # —— 提取最终答案：找最后一条无 tool_calls 的 AIMessage ——
        answer = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                answer = m.content if isinstance(m.content, str) else ""
                if answer:
                    break

        # 拒答兜底：全程无有效检索却给出具体答案 → 视为潜在编造
        if not answer and not self._has_successful_retrieval:
            answer = KB_REJECT_TEMPLATE
        # 先清洗再兜底：清洗可能把纯元话语（如「thought: 信息已足够」）清成空串
        answer = (
            self._normalize_answer(answer or "")
            or "抱歉，我暂时无法生成完整回答。请稍后重试或换一种问法。"
        )

        return {
            "answer": answer,
            "called_tools": self._called_tools,
            "tool_calls_detail": self._tool_calls_detail,
            "tool_results": self._tool_results,
            "messages": [],
            "usage_metadata": self._usage,
        }

    # 图构建
    def _build_graph(self):
        """构建并编译 ReAct 循环图。"""
        builder = StateGraph(ReActState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_edge(START, "agent")
        # 条件边：agent 之后根据 state 决定去 tools / 回 agent(强制计算器) / 结束
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "agent": "agent", "end": END},
        )
        builder.add_edge("tools", "agent")
        return builder.compile()

    # 节点：agent（调用 LLM，决定是否调用工具）
    async def _agent_node(self, state: ReActState):
        rounds = state.get("rounds", 0)
        force_final = rounds >= MAX_REACT_ROUNDS + 1

        # SystemMessage 不入 state，每轮按 FC 同源模板重建（含 force_final 提示）
        system = self._build_fc_system_message(self._query, "", force_final)

        # 计算器闸（镜像 FC 心算闸）：若上一轮模型已输出数值结论却未调 calculator，说明它在"心算"。
        # ⚠ 不能只把铁律拼进 system 后重发——实测 7B 在"自己已给完整答案"的 trailing-AIMessage
        # 上下文里再被调用会直接返回空串（r2 empty），于是 _should_enforce_calculator(query,"")
        # 因空串短路为 False、闸释放、心算答案被当最终答案（mix_004：9416+3210=12626 心算，
        # calculator 从未被调 → mixed 工具准确率塌到 0.14）。
        # 修正：注入一条"点名数字 + 要求立即调 calculator"的 Human 指令作为末条消息，以新鲜上下文
        # 重取（对齐 FC 每轮只发 System+Human、不 trailing 自己答案的成功条件），逼模型发出
        # calculator 调用，而不是对着自己的答案续写。
        last = state["messages"][-1] if state["messages"] else None
        last_text = (
            last.content if isinstance(last, AIMessage) and isinstance(last.content, str) else ""
        )
        enforce_calc = (
            not force_final
            and isinstance(last, AIMessage)
            and not getattr(last, "tool_calls", None)
            and self._lg_should_force_calc(last_text)
        )

        bound = self._get_llm(self._llm_provider).bind_tools(self._build_fc_tool_schemas())
        if enforce_calc:
            self._calc_forced = True
            prev_text = last.content if isinstance(last.content, str) else ""
            nums = "、".join(sorted(set(re.findall(r"\d+(?:\.\d+)?", prev_text)), key=float)[:8])
            nums_part = f"（你刚才提到的数字：{nums}）" if nums else ""
            directive = (
                f"停。你尚未通过 calculator 得出数值结论{nums_part}，这属于心算，不能作为最终答案。"
                "请立即调用 calculator 工具完成计算：把需要的每个算式分别交给 calculator，"
                "拿到返回结果后再基于它给出最终答案。禁止再次直接输出未经 calculator 的数值结论。"
            )
            system += (
                "\n\n【数值计算铁律】最终结果中的所有数字必须通过 calculator 工具得出，禁止心算。"
            )
            llm_input = [SystemMessage(content=system), HumanMessage(content=directive)]
        else:
            llm_input = [SystemMessage(content=system), *state["messages"]]

        resp = await bound.ainvoke(llm_input)
        self._accumulate_usage(resp)

        # 兜底（与 FC 版 run_fc_stream 对齐）：llama.cpp 偶发把 tool call 写成正文，
        # 形如 'knowledge_search {"query": ...}'。若 resp 无结构化 tool_calls，从正文还原，
        # 否则 _should_continue 会误判为"无工具调用"直接 END，导致检索/计算被整段跳过、
        # 工具准确率骤降（这也是首跑 83.43 中 kq_012/hall_004/file_003 三类泄漏用例的根因）。
        if not getattr(resp, "tool_calls", None):
            content = resp.content if isinstance(resp.content, str) else ""
            text_calls = self._extract_tool_calls_from_text(content)
            if text_calls:
                resp = AIMessage(
                    content=content,
                    tool_calls=[
                        {"name": n, "args": a, "id": f"lg_call_{i}", "type": "tool_call"}
                        for i, (n, a) in enumerate(text_calls)
                    ],
                )

        return {"messages": [resp], "rounds": rounds + 1}

    # 节点：tools（复用父类工具执行 + guard 链，复写 tool_results 等）
    async def _tools_node(self, state: ReActState):
        last: AIMessage = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        out_messages: list[ToolMessage] = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tc_id = tc.get("id") or ""
            raw_input = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args

            # ---- 以下 guard 链与 FC 版 run_fc_stream 对齐（复用父类判定）----
            # 重复调用检测闸
            if self._is_repeat_loop(name, raw_input):
                self._anomaly_streak += 1
                self._loop_intervention_count[name] = self._loop_intervention_count.get(name, 0) + 1
                obs = (
                    f"检测到你连续多次以高度相似参数调用工具 '{name}'，可能陷入循环。"
                    "请基于已获取的信息继续推进或更换检索/计算角度；若信息已充足，请直接给出最终答案。"
                )
                self._record_failed(name, raw_input, obs)
                out_messages.append(ToolMessage(content=obs, tool_call_id=tc_id, name=name))
                continue

            # 多轮重复调用闸
            dup, dup_hint = self._is_duplicate_tool_call(name, raw_input)
            if dup:
                self._anomaly_streak += 1
                self._record_failed(name, raw_input, dup_hint)
                out_messages.append(ToolMessage(content=dup_hint, tool_call_id=tc_id, name=name))
                continue

            # 文件读取后重复验证闸
            red, red_hint = self._is_redundant_file_calc(name, raw_input)
            if red:
                self._anomaly_streak += 1
                self._record_failed(name, raw_input, red_hint)
                out_messages.append(ToolMessage(content=red_hint, tool_call_id=tc_id, name=name))
                continue

            # 工具存在性校验
            tool = self._find_tool(name)
            if tool is None:
                avail = ", ".join(self._tool_map.keys()) or "（无）"
                obs = (
                    f"工具 '{name}' 不在可用工具列表中（可用：{avail}）。"
                    "请只使用可用工具，或基于已有信息直接作答。"
                )
                self._record_failed(name, raw_input, obs)
                out_messages.append(ToolMessage(content=obs, tool_call_id=tc_id, name=name))
                continue

            # 执行工具
            result, obs_text, parsed_args = await self._execute_tool(tool, raw_input)

            # 失败/原样重试拦截
            if (name, raw_input) in self._failed_calls:
                detail = self._err_detail(result, obs_text)
                obs = (
                    f"工具 '{name}' 以完全相同的参数已失败过。禁止原样重复调用。"
                    "请改用不同的表述/参数/文件路径，或换用其他工具；"
                    "若已尝试多次仍失败，请直接基于已有信息回答或返回拒绝模板。"
                )
                self._record_failed(name, raw_input, obs, detail)
                out_messages.append(ToolMessage(content=obs, tool_call_id=tc_id, name=name))
                continue

            # 空/无效结果检测
            if self._is_empty_result(result, obs_text) or (
                result is not None and not getattr(result, "success", True)
            ):
                self._failed_calls.append((name, raw_input))
                self._anomaly_streak += 1
                detail = self._err_detail(result, obs_text)
                obs = (
                    f"工具 '{name}' 返回空或无效结果{detail}。"
                    "请调整参数后重试，或换用其他工具；若已尝试多次仍失败，请返回拒绝模板。"
                )
                self._record_failed(name, raw_input, obs, detail)
                out_messages.append(ToolMessage(content=obs, tool_call_id=tc_id, name=name))
                continue

            # ---- 成功分支（与 FC 对齐）----
            self._anomaly_streak = 0
            self._has_successful_retrieval = True
            if name == "calculator" and getattr(result, "success", False):
                self._calculator_returned_number = True
            self._called_tools.append(name)
            self._tool_calls_detail.append(
                {"tool_name": name, "args": raw_input, "status": "success"}
            )
            self._tool_results.append(
                {
                    "tool_name": name,
                    "tool_args": parsed_args,
                    "tool_call_id": tc_id,
                    "status": "success",
                    "result": obs_text,
                    "error": None,
                }
            )
            obs_text = await self._preprocess_observation(obs_text, self._query)
            out_messages.append(ToolMessage(content=obs_text or "", tool_call_id=tc_id, name=name))

            if self._need_llm_review():
                review = await self._llm_review(self._query, self._render_trace())
                if review:
                    self._pending_review = review

        return {"messages": out_messages}

    # 条件边路由
    def _should_continue(self, state: ReActState) -> str:
        # 轮数硬上限（与 FC 一致：MAX_REACT_ROUNDS + 1）
        if state.get("rounds", 0) > MAX_REACT_ROUNDS + 1:
            return "end"

        last = state["messages"][-1]
        calls = self._extract_tool_calls(last)
        if calls:
            return "tools"

        # 无 tool_calls：LG 本地计算器闸（比共享 indicator 闸更全，与 _agent_node 共用同一判定）。
        # 命中 → 回 agent 用新鲜指令再逼一轮调 calculator（含模型空串续写场景，受上方轮数硬上限约束）。
        content = last.content if isinstance(last.content, str) else ""
        if self._lg_should_force_calc(content):
            return "agent"

        return "end"

    # 小工具
    @staticmethod
    def _looks_like_text_arithmetic(text: str) -> bool:
        """答案里出现行内心算痕迹，是模型在正文里口算而非调用 calculator 的铁证。覆盖：
        - 「数字 运算符 数字 =」（如 '1,280 - 780 = 500'）
        - LaTeX 分数（如 '\\frac{7680}{640} = 12'）
        用于补 _should_enforce_calculator 的 indicator 词表覆盖不到的问法。
        要求带运算符/\\frac + 数字，避免误伤区间/编号（如 '100-200元'、'2024-10' 无运算符不命中）。
        仅 LG 臂启用，不改共享闸，避免影响 ReAct / FC 两臂基线。"""
        if not text:
            return False
        if re.search(r"\d[\d,\.]*\s*[-+*/×÷]\s*\d[\d,\.]*\s*=", text):
            return True
        return bool(re.search(r"\\frac\s*\{?\s*\d", text))

    @staticmethod
    def _lg_needs_calc_extra(query: str, answer: str) -> bool:
        """补 _should_enforce_calculator 的 indicator 词表缺口：几个明确表"运算"但词表没有的
        复合问法（'几倍'、'多多少'、'差多少'、'几成'）。答案含数字才触发。
        仅 LG 臂启用，避免改动共享闸影响另两臂。"""
        if not answer or not re.search(r"\d", answer):
            return False
        return any(w in query for w in ("几倍", "多多少", "差多少", "几成"))

    def _lg_should_force_calc(self, last_text: str) -> bool:
        """LG 本地计算器闸统一判定（比共享 _should_enforce_calculator 更全）。
        聚合：①共享 indicator 闸 ②行内心算痕迹（a op b = / LaTeX 分数）③复合运算问法
        ④已强制过但仍未拿到 calculator 成功结果。
        ⚠ _agent_node 与 _should_continue 必须共用本方法——否则两处判定不一致，
        窄闸的 _should_continue 会先路由到 end，宽闸的 _agent_node 根本没机会执行
        （mix_002「多多少」漏放行的根因）。calculator 已成功返回数字则一律不再强制。"""
        if self._calculator_returned_number:
            return False
        return (
            self._should_enforce_calculator(self._query, last_text)
            or self._looks_like_text_arithmetic(last_text)
            or self._lg_needs_calc_extra(self._query, last_text)
            or self._calc_forced
        )

    def _err_detail(self, result, obs_text: str) -> str:
        detail = ""
        if result and not getattr(result, "success", True) and getattr(result, "error", ""):
            detail = f"（错误详情：{result.error}）"
        elif obs_text and "错误" in obs_text:
            detail = f"（{obs_text}）"
        return detail

    def _record_failed(self, name: str, raw_input: str, obs: str, detail: str = "") -> None:
        self._recent_actions.append((name, raw_input))
        self._recent_observations.append((name, obs))
        self._tool_calls_detail.append(
            {"tool_name": name, "args": raw_input, "status": "failed", "error": detail or obs}
        )
