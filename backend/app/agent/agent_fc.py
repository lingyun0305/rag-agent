# backend/app/agent/agent_fc.py
"""
原生 Function Calling 范式 Agent（与手写 ReAct 的对照实验用）。

设计原则（用户要求「尽量不改原链路代码」）：
- 完全不修改 agent.py 的 SmartAgent.run_stream / run —— 现有 86 分基线零风险。
- FunctionCallingAgent 继承 SmartAgent，复用全部工具解析/执行/guard 判定 helper：
  _execute_tool / _find_tool / _parse_tool_args / _should_enforce_calculator /
  _is_repeat_loop / _is_redundant_file_calc / _is_duplicate_tool_call /
  _normalize_answer / _append_trace / _render_trace / _state /
  _maybe_compress_context / _need_llm_review / _llm_review / _get_llm / _accumulate_usage。
- 仅替换「决策接口」这一层：
  * LLM 调用从「prompt 里塞 <action> 标签 + stop 序列」改为 llm.bind_tools(schemas).ainvoke(messages)
  * 解析从「_parse_react 正则」改为解析 resp.tool_calls
- 复用与 run_stream 完全一致的 guard 判定 + 状态维护，确保对照组变量唯一 = 决策方式。

接入评测：在 run_eval.py 的 build_eval_agent 里按 AGENT_MODE 环境变量切换返回
SmartAgent / FunctionCallingAgent（react 为默认，fc 为对照）。
"""

import json
import json as json_mod
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent.agent import KB_REJECT_TEMPLATE, MAX_REACT_ROUNDS, SmartAgent
from backend.app.agent.agent_state import AgentState
from backend.app.common.logger import logger

# ====== FC 专用 system ======
FC_SYSTEM_TEMPLATE = """你是企业研发智能体，擅长自主调用可用工具帮助用户解决问题，严格遵循以下规则执行：

【工具使用】
你可以使用函数调用（function calling）获取外部信息，具体工具与参数由系统动态提供，包括（不限于）：
- knowledge_search：检索企业内部知识库
- file_reader：读取知识库/上传目录中的文档原文
- oa_reimbursement：查询 OA 报销/流程数据
- calculator：计算数值

当问题需要任何外部信息（制度、数据、文件内容、计算结果）时，必须调用对应工具，不要凭记忆作答。
工具调用失败时，应根据返回的错误信息调整参数后重试，不要用相同参数原样重试。

识别问候/致谢等社交场景直接回应：当问题属于问候（如"你好""你是谁"）、致谢（如"谢谢""感谢"）、闲聊等时，不调用 knowledge_search，直接用自然语言友好回应，不要套用拒答模板。

【数值计算铁律】
凡是需要计算任何数值（含乘方、百分比、增长率、金额、求和、同比/环比、倍数、平均值等），无论问题是否简单，都必须调用 calculator；最终答案中出现的所有数字、比例、百分比、计算类结论，必须有对应 calculator 的返回结果作为依据，禁止心算或改写计算结果。
如果题目没有要求计算精度，则计算结果保留两位小数（例如 20.31%），否则遵循题目要求。

【重复调用禁止】
多轮/多步中，前文已经通过工具获取过的数据、结论可直接复用；禁止用相同参数重复调用同一工具查询相同内容。

【拒答规则】
必须先检验（如调用 knowledge_search 检索）后才能拒答：凡业务、制度、数据、文件内容类问题，禁止在未调用任何工具的情况下以"超出知识库范围"为由直接拒答；仅当检索返回为空或与问题无关时，才可按下述模板拒答：
  - 知识库未覆盖："抱歉，我在知识库中未找到相关信息，无法提供答案。"（使用此句式前必须已完成至少一次 knowledge_search 调用）
  - 外部/时事/预测类："抱歉，该问题涉及外部信息/预测，不在企业知识库范围内，无法提供答案。"
  - 隐私/安全/敏感类（如身份证号、手机号、薪资明细、员工个人信息、账号密码等个人数据）："抱歉，该问题涉及隐私/安全信息，无法提供答案。"

请通过函数调用来获取所需信息，不要输出 XML 标签。"""


class FunctionCallingAgent(SmartAgent):
    """
    原生 Function Calling 范式 Agent。

    与手写 ReAct 的唯一区别在「模型如何表达调工具意图」：
    ReAct  = 模型输出 <action>/<input> 文本 → 正则解析
    FC     = 模型通过 OpenAI tools 参数输出结构化 tool_calls → 直接解析
    工具逻辑、guard 判定、检索、评测全部复用父类。
    """

    async def run_fc_stream(
        self,
        query: str,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        llm_provider: str | None = None,
    ):
        """
        FC 范式流式主循环。事件格式与 run_stream 完全一致
        （thinking / tool_result / answer），因此非流式 run() 与评测脚本无需改动调用方式。
        """
        llm = self._get_llm(llm_provider)

        # ====== 状态初始化（与 run_stream 完全对齐，复制以免改动父类）======
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
        self._review_done = False
        self._pending_review: str | None = None
        self._loop_intervention_count: dict[str, int] = {}
        self._conversation_history: list[dict] = list(history) if history else []

        self._state = AgentState()
        self._state.reset(query)

        step_counter = 0
        rounds = 0
        stall_streak = 0  # 连续干预（卡住）计数，达到阈值触发携上下文强制收尾，避免烧满轮数
        final_answer = ""

        schemas = self._build_fc_tool_schemas()
        # bind_tools 在底层把 tools 转成 OpenAI 兼容参数发给 /v1/chat/completions。
        # 注意：llama.cpp 需带 --jinja 启动才会启用 tool schema 解析；否则模型不会返回 tool_calls。
        bound_llm = llm.bind_tools(schemas) if schemas else llm

        while True:
            rounds += 1
            if rounds > MAX_REACT_ROUNDS + 1:
                final_answer = "抱歉，推理步数超出上限，仍未能得出确切答案。"
                break

            force_final = rounds == MAX_REACT_ROUNDS + 1

            # [2026-10-16] 卡死早停：连续 2 次干预（重复调用 / 空结果 / 工具不存在 / 失败重试）
            # 仍无进展 → 携完整上下文强制收尾，避免 7B 在异常时烧满 8 轮。每轮全量重 prefill
            # 是 80s/例 的主要耗时来源，早停直接砍掉尾部冗余轮，且不损伤正常收敛路径。
            if stall_streak >= 2 and not force_final:
                clean = await self._request_clean_final(query, llm)
                if clean:
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
                    final_answer = clean
                else:
                    final_answer = (
                        self._recent_observations[-1][1]
                        if self._recent_observations
                        else "抱歉，推理步数过多仍未能得出确切答案。"
                    )
                final_answer = self._normalize_answer(final_answer)
                break

            self._state.update(self._tool_results, query)
            # 单臂 ReAct 改造对齐：只注入「最新一轮」原文，早期轮次由常驻 AgentState 块承载，
            # prefill 从 O(轮数) 降为 O(1)。
            system = self._build_fc_system_message(
                query, self._render_trace_latest_only(), force_final
            )

            self._llm_calls += 1
            try:
                # 关键：问题必须以 user 消息送达模型。只发 SystemMessage 时
                # Qwen2.5-7B 看不到任何用户问题，会完全自由发挥（FC 跑分崩盘的真正根因）。
                resp = await bound_llm.ainvoke(
                    [SystemMessage(content=system), HumanMessage(content=query)]
                )
            except Exception as e:
                logger.error(f"[FC] LLM 调用失败 (第{rounds}轮): {e}")
                final_answer = "抱歉，模型调用出现异常，暂时无法完成推理。"
                break
            self._accumulate_usage(resp)

            parsed_calls = self._extract_tool_calls(resp)

            # ====== 兜底：llama.cpp 偶发把 tool call 写成正文（如 'knowledge_search {"query": ...}'）======
            if not parsed_calls:
                content = resp.content if isinstance(resp.content, str) else ""
                text_calls = self._extract_tool_calls_from_text(content)
                if text_calls:
                    parsed_calls = text_calls
                    self._append_trace(
                        thought="（系统兜底：从正文解析出工具调用）",
                        action=text_calls[0][0],
                        inp=json_mod.dumps(text_calls[0][1], ensure_ascii=False),
                        observation="已从正文中识别并修复工具调用，继续执行",
                    )
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": "system",
                            "content": "Observation: [干预] 已从正文中识别并修复工具调用",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )

            # ====== 无 tool_calls → 视为最终答案（或触发心算闸）======
            if not parsed_calls:
                answer_text = resp.content if isinstance(resp.content, str) else ""
                answer_text = (answer_text or "").strip()

                # 心算校验闸：非强制轮、含数字但未调 calculator → 强制再走一轮
                if (not force_final) and self._should_enforce_calculator(query, answer_text):
                    hint = (
                        "数值计算结果必须通过 calculator 工具得出，禁止心算输出未经验证的数值，"
                        "请调用计算器后再作答。"
                    )
                    self._append_trace(
                        thought="（系统校验）", action="（计算校验）", inp="", observation=hint
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

                # 强制总结轮：若仍空，尝试用最近一次观测拼答案
                if force_final and not answer_text:
                    answer_text = (
                        self._recent_observations[-1][1] if self._recent_observations else ""
                    )

                if answer_text:
                    final_answer = answer_text
                # 拒答兜底：全程无有效工具数据却给出具体答案 → 视为潜在编造
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
                    if not any(kw in final_answer for kw in _refuse_kw):
                        final_answer = KB_REJECT_TEMPLATE
                final_answer = self._normalize_answer(final_answer)
                break

            # ====== 有 tool_calls → 逐个执行（复用父类 guard + 执行链路）======
            for name, args in parsed_calls:
                raw_input = json_mod.dumps(args, ensure_ascii=False)

                # 重复调用检测闸
                if self._is_repeat_loop(name, raw_input):
                    self._anomaly_streak += 1
                    stall_streak += 1
                    self._loop_intervention_count[name] = (
                        self._loop_intervention_count.get(name, 0) + 1
                    )
                    hint = (
                        f"检测到你连续多次以高度相似参数调用工具 '{name}'，可能陷入循环。"
                        "请基于已获取的信息继续推进或更换检索/计算角度；若信息已充足，请直接给出最终答案。"
                    )
                    self._append_trace(thought="", action=name, inp=raw_input, observation=hint)
                    self._recent_actions.append((name, raw_input))
                    self._recent_observations.append((name, hint))
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": name,
                            "content": f"Observation: [干预] {hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # 多轮重复调用闸
                dup, dup_hint = self._is_duplicate_tool_call(name, raw_input)
                if dup:
                    self._anomaly_streak += 1
                    stall_streak += 1
                    self._append_trace(thought="", action=name, inp=raw_input, observation=dup_hint)
                    self._recent_actions.append((name, raw_input))
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": name,
                            "content": f"Observation: [干预] {dup_hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # 文件读取后重复验证闸
                red, red_hint = self._is_redundant_file_calc(name, raw_input)
                if red:
                    self._anomaly_streak += 1
                    stall_streak += 1
                    self._append_trace(thought="", action=name, inp=raw_input, observation=red_hint)
                    self._recent_actions.append((name, raw_input))
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": name,
                            "content": f"Observation: [干预] {red_hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # 工具存在性校验
                tool = self._find_tool(name)
                if tool is None:
                    self._anomaly_streak += 1
                    stall_streak += 1
                    avail = ", ".join(self._tool_map.keys()) or "（无）"
                    hint = (
                        f"工具 '{name}' 不在可用工具列表中（可用：{avail}）。"
                        "请只使用可用工具，或基于已有信息直接作答。"
                    )
                    self._append_trace(thought="", action=name, inp=raw_input, observation=hint)
                    self._recent_actions.append((name, raw_input))
                    step_counter += 1
                    yield json_mod.dumps(
                        {
                            "type": "tool_result",
                            "step": step_counter,
                            "tool": name,
                            "content": f"Observation: [干预] {hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # 执行工具
                result, obs_text, parsed_args = await self._execute_tool(tool, raw_input)

                # 失败/原样重试拦截
                if (name, raw_input) in self._failed_calls:
                    self._anomaly_streak += 1
                    stall_streak += 1
                    hint = (
                        f"工具 '{name}' 以完全相同的参数已失败过。禁止原样重复调用。"
                        "请改用不同的表述/参数/文件路径，或换用其他工具；"
                        "若已尝试多次仍失败，请直接基于已有信息回答或返回拒绝模板。"
                    )
                    detail = ""
                    if (
                        result
                        and not getattr(result, "success", True)
                        and getattr(result, "error", "")
                    ):
                        detail = f"（错误详情：{result.error}）"
                    elif obs_text and "错误" in obs_text:
                        detail = f"（{obs_text}）"
                    self._append_trace(thought="", action=name, inp=raw_input, observation=hint)
                    self._recent_actions.append((name, raw_input))
                    self._recent_observations.append((name, obs_text or ""))
                    self._tool_calls_detail.append(
                        {
                            "tool_name": name,
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
                            "tool": name,
                            "content": f"Observation: [干预] {hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # 空/无效结果检测
                if self._is_empty_result(result, obs_text) or (
                    result is not None and not getattr(result, "success", True)
                ):
                    self._failed_calls.append((name, raw_input))
                    self._anomaly_streak += 1
                    stall_streak += 1
                    detail = ""
                    if (
                        result
                        and not getattr(result, "success", True)
                        and getattr(result, "error", "")
                    ):
                        detail = f"（错误详情：{result.error}）"
                    elif obs_text and "错误" in obs_text:
                        detail = f"（{obs_text}）"
                    hint = (
                        f"工具 '{name}' 返回空或无效结果{detail}。"
                        "请调整参数后重试，或换用其他工具；若已尝试多次仍失败，请返回拒绝模板。"
                    )
                    self._append_trace(thought="", action=name, inp=raw_input, observation=hint)
                    self._recent_actions.append((name, raw_input))
                    self._recent_observations.append((name, obs_text or ""))
                    self._tool_calls_detail.append(
                        {
                            "tool_name": name,
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
                            "tool": name,
                            "content": f"Observation: [干预] {hint}",
                            "phase": "observation",
                        },
                        ensure_ascii=False,
                    )
                    await self._maybe_compress_context()
                    continue

                # ====== 成功分支（与 run_stream 对齐）======
                self._anomaly_streak = 0
                stall_streak = 0
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
                        "tool_call_id": "",
                        "status": "success",
                        "result": obs_text,
                        "error": None,
                    }
                )

                obs_text = await self._preprocess_observation(obs_text, query)

                step_counter += 1
                action_content = f"调用 {name}" + (f"，参数：{raw_input}" if raw_input else "")
                yield json_mod.dumps(
                    {
                        "type": "thinking",
                        "step": step_counter,
                        "content": action_content,
                        "tools": [{"name": name, "args": parsed_args}],
                        "phase": "action",
                    },
                    ensure_ascii=False,
                )

                step_counter += 1
                obs_content = f"Observation: [{name}] {obs_text}"
                yield json_mod.dumps(
                    {
                        "type": "tool_result",
                        "step": step_counter,
                        "tool": name,
                        "content": obs_content,
                        "phase": "observation",
                    },
                    ensure_ascii=False,
                )

                self._append_trace(thought="", action=name, inp=raw_input, observation=obs_text)
                self._recent_actions.append((name, raw_input))
                self._recent_observations.append((name, obs_text or ""))

                if self._need_llm_review():
                    review = await self._llm_review(query, self._render_trace())
                    if review:
                        self._pending_review = review

                await self._maybe_compress_context()

            # 注入独立 LLM 复盘结论（若有）
            if self._pending_review:
                hint = self._pending_review
                self._pending_review = None
                self._append_trace(
                    thought="（系统复盘）", action="（复盘建议）", inp="", observation=hint
                )

            await self._maybe_compress_context()

        # ====== 兜底 + 产出最终答案事件 ======
        # 先清洗再判空：清洗可能把纯元话语（如「thought: 信息已足够」）清成空串，
        # 顺序必须是 normalize → 判空兜底，避免空答案事件下发。
        final_answer = self._normalize_answer(final_answer)

        if not final_answer:
            final_answer = "抱歉，我暂时无法生成完整回答。请稍后重试或换一种问法。"

        if getattr(self, "_state", None) is not None and final_answer:
            self._state.conclusion = final_answer

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
        yield json_mod.dumps({"type": "answer", "content": final_answer}, ensure_ascii=False)

    # ====== FC 专用辅助 ======

    def _build_fc_system_message(self, query: str, trace_text: str, force_final: bool) -> str:
        """FC system：保留角色/约束，去掉 <action> 标签，trace + history 注入上下文。

        单臂 ReAct 改造对齐（2026-10-12）：trace_text 由调用方改为 _render_trace_latest_only()，
        仅注入「最新一轮」原文；跨轮事实连续性由下方常驻 AgentState 要点块承载，
        因此每轮 prefill 从 O(轮数) 降为 O(1)，与 ReAct 臂行为一致。FC 已在每轮
        self._state.update() 维护该状态、_tool_results 正常填充，故可安全复用。
        """
        parts = [FC_SYSTEM_TEMPLATE]

        if getattr(self, "_conversation_history", []):
            hist_lines = []
            for m in self._conversation_history:
                role = m.get("role", "")
                content = m.get("content") or m.get("parts") or ""
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                hist_lines.append(f"[{role}] {content}")
            parts.append("## 历史对话\n" + "\n".join(hist_lines))

        # 轮间状态归集块（代码确定性抽取，非 LLM 维护；与 ReAct 臂 _build_system_message 对齐）
        state_obj = getattr(self, "_state", None)
        if state_obj is not None:
            state_text = state_obj.render_for_prompt()
            if state_text:
                parts.append(state_text)

        # [2026-10-12] 检索原文完整块（FC 专用，补齐 latest-only trace 丢失的早期 KB 全文）。
        # 背景：FC 每轮只注入「最近 2 轮」trace，若 KB 在第 1 轮调用、答案在第 3 轮综合，
        # 其完整 observation 在后续轮被丢弃；AgentState 的 known_facts 现已不再截断
        # （完整结果原样留存），但 latest-only trace 仍只保留最近 2 轮，更早的 KB 原文需靠本区块补全。
        # 此处常驻「检索原文」块，完整保留 knowledge_search 的 result（从 _tool_results 全量重渲染，无截断）。
        # 检索通常仅 1~2 次（重复检索闸已拦截），token 增加有界，远小于历史对话块。
        kb_block = self._render_kb_sources()
        if kb_block:
            parts.append(kb_block)

        if trace_text:
            parts.append("## 已完成的推理步骤\n" + trace_text)

        if force_final:
            parts.append("【最终总结】现在必须基于已获取的信息给出最终答案，不要再调用工具。")
        return "\n\n".join(parts)

    def _build_fc_tool_schemas(self) -> list[dict]:
        """把父类工具列表转成 OpenAI function-calling schema（供 bind_tools 使用）。

        FC 对照专用清洗（不动工具代码 / 不碰 ReAct 基线）：
        - 去掉参数描述里的「如'xxx'」「例如」等具体示例，避免 7B 照抄示例而非使用
          用户真实问题（这是 FC 跑分崩盘的 #1 根因：20 个 kq 用例的检索 query 全是
           schema 示例里的三句模板）。
        - knowledge_search 仅暴露 query：top_k/strategy/use_rerank/collection_name 改由
          per-Agent tool_configs 默认控制，禁止模型自作主张降配检索。
        - calculator 简化成单一 expression 字段，去掉「expression/expressions 二选一」的
          歧义写法（7B 会因此发出空参 args={} → 计算器连错两次 → 工具准确率崩）。
        """
        _TOOL_NAMES = [n for n in self._tool_map.keys() if n != "system"]

        def _strip_example(desc):
            if not isinstance(desc, str):
                return desc
            s = re.sub(r"[\uFF08（]如[^\uFF09）]*[\uFF09）]", "", desc)
            s = re.sub(r"如['\"][^'\"]*['\"]", "", s)
            s = re.sub(r"例如[^，。；,;]*", "", s)
            return s.strip()

        schemas: list[dict] = []
        for name, tool in self._tool_map.items():
            if name in ("system",):
                continue
            params = getattr(tool, "parameters_schema", None) or {
                "type": "object",
                "properties": {},
            }
            props = dict(params.get("properties", {}) or {})

            if name == "knowledge_search":
                q = dict(props.get("query", {"type": "string"}))
                q["description"] = (
                    _strip_example(q.get("description", "用户的检索问题"))
                    or "用户的检索问题，使用用户原话或问题关键词，严禁照抄示例"
                )
                params = {
                    "type": "object",
                    "properties": {"query": q},
                    "required": ["query"],
                }
            elif name == "calculator":
                params = {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，如 2*(3+5) 或 125*8，支持 + - * / ^ () 与百分比",
                        }
                    },
                    "required": ["expression"],
                }
            else:
                props = {k: (dict(v) if isinstance(v, dict) else v) for k, v in props.items()}
                for v in props.values():
                    if isinstance(v, dict):
                        v["description"] = _strip_example(v.get("description", ""))
                params = {"type": "object", "properties": props}

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", name),
                        "description": _strip_example(getattr(tool, "description", "")),
                        "parameters": params,
                    },
                }
            )
        return schemas

    def _extract_tool_calls_from_text(self, content: str):
        """兜底：llama.cpp 偶发把 tool call 写成正文。
        兼容两种形态：
        形态一（Python 风格）：'knowledge_search {"query": ...}'
        形态二（chat-template 风格）：'<tool_call>\\n{{"name": "knowledge_search",
            "arguments": {"query": ...}}}\\n</tool_call>'（Qwen 模板原样输出，
            常伴 {{ }} 双花括号与截断前缀）。
        当 resp 无结构化 tool_calls 时，从正文识别并解析为 tool call 继续执行。"""
        if not isinstance(content, str) or not content.strip():
            return []
        text = re.sub(r"```[a-zA-Z]*", "", content).replace("```", "")
        names = "|".join(re.escape(n) for n in self._tool_map.keys() if n != "system")
        # 形态一：tool_name {...} 或 tool_name({...})
        m = re.search(rf"\b({names})\s*\(?\s*(\{{[^{{}}]*\}})", text, re.DOTALL)
        if m:
            try:
                args = json_mod.loads(m.group(2))
            except (json.JSONDecodeError, TypeError):
                args = None
            if isinstance(args, dict) and args:
                return [(m.group(1), args)]
        # 形态二：{"name": "...", "arguments": {...}}（含 {{ }} 双花括号，先归一化）
        norm = text.replace("{{", "{").replace("}}", "}")
        m = re.search(
            rf'\{{\s*"name"\s*:\s*"({names})"\s*,\s*"arguments"\s*:\s*(\{{.*?\}})\s*\}}',
            norm,
            re.DOTALL,
        )
        if m:
            try:
                args = json_mod.loads(m.group(2))
            except (json.JSONDecodeError, TypeError):
                return []
            if isinstance(args, dict) and args:
                return [(m.group(1), args)]
        return []

    @staticmethod
    def _extract_tool_calls(resp) -> list[tuple[str, dict]]:
        """
        从 LLM 响应里抽取 tool_calls，兼容两种形态：
        - LangChain AIMessage.tool_calls（list of ToolCall dataclass）
        - OpenAI 原生 additional_kwargs["tool_calls"]（list of dict）
        统一返回 [(name, args_dict), ...]
        """
        raw = getattr(resp, "tool_calls", None)
        if not raw:
            raw = (getattr(resp, "additional_kwargs", {}) or {}).get("tool_calls", [])
        calls: list[tuple[str, dict]] = []
        for tc in raw or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {}) or {}
                name = fn.get("name") or tc.get("name")
                args = fn.get("arguments") or tc.get("args") or {}
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", {}) or {}
            if isinstance(args, str):
                try:
                    args = json_mod.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if name:
                calls.append((name, args or {}))
        return calls

    # 非流式入口：与父类 run() 同构，仅调度 run_fc_stream（评测脚本无需改调用方式）
    async def run(
        self,
        query: str,
        user_id: str,
        session_id: str,
        history: list[dict] | None = None,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        thinking_steps = []
        answer = ""
        async for chunk in self.run_fc_stream(
            query=query,
            user_id=user_id,
            session_id=session_id,
            history=history,
            llm_provider=llm_provider,
        ):
            try:
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
