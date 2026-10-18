# backend/app/agent/agent_state.py
"""
轻量级 Agent 轮间状态（state）抽取。

设计原则（与 LangGraph 的 state + reducer 模型刻意不同）：
- state **不由 LLM 维护**，而是由代码在每轮结束后从 ``_trace`` / ``_tool_results``
  **确定性重建**（deterministic rebuild）。因此不存在「增量合并」问题，
  也不需要 LangGraph 的 reducer 语义——每轮直接基于事实源重算全量 state，
  避免本地模型跨轮维护结构化 state 时出现漂移 / 丢失。
- 这是「轮间记忆压缩」：把扁平冗长的推理轨迹抽稀成几行要点，注入下一轮
  system message，弥补纯 Prompt ReAct 缺乏持久化状态的问题。

字段：
- original_query：原始问题（防目标漂移锚点）
- sub_goal：目标（开局一次派生，轻量；此处直接复用 original_query，避免引入 LLM）
- known_facts：已检索 / 读取的事实（从 observation 抽，附检索词便于去重）
- calculated_values：已计算的数值（从 calculator 结果抽，形如 ``expr = value``）
- pending：待补齐提示（**代码派生**，非 LLM 写；如「已算出 N 个值请比较/排序」）
- conclusion：当前结论（推理进行中为占位；最终答案产出后置为 final_answer）
"""

import re
from dataclasses import dataclass, field

# 工具名（与 registry / tool.name 对齐）
_CALC = "calculator"
_KB = "knowledge_search"
_FILE = "file_reader"

# [2026-10-13] 移除「已知事实单行截断」：早期因单臂 ReAct 走 latest-only trace，
# AgentState 是历史 Observation 的唯一载体，曾用 240→640 截断保护。
# 现策略：①knowledge_search 在 known_facts 中只保留「指针」（检索词 + 指向
#   ## 检索原文 区块的提示），全文由 _render_kb_sources 常驻区块承载，避免重复占 token；
# ②file_reader 等需留存原文的内容仍 _extract_fact(full=True) 全量写入，跨轮不丢。
# 注：LG 臂虽继承 AgentState 但从不调用 update()，故本文件改动仅 ReAct+FC 实质生效。

# 判定「问题是否涉及数值计算」的关键词
_CALC_HINT_RE = re.compile(
    r"(人均|总和|合计|增长|率|差值|相差|排名|比较|对比|多少|占比|比例|均值|平均)"
)
# 判定「是否已在做比较/排序/聚合」的动作关键词
_COMPARE_RE = re.compile(r"(max|min|sum|排序|对比|比较|排名|rank|>=?|<=?)", re.IGNORECASE)
# 从 calculator 结果行抽取「= 数值」
_CALC_VALUE_RE = re.compile(r"=\s*([\-+]?\d[\d,]*\.?\d*)")

# 拒答 / 知识库未覆盖关键词（与 agent.py 保持一致口径），用于把拒答类结论标记为软结论
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


def _is_refusal(text: str) -> bool:
    """判断文本是否含拒答 / 知识库未覆盖措辞。"""
    return any(kw in (text or "") for kw in _REFUSAL_KW)


@dataclass
class AgentState:
    original_query: str = ""
    sub_goal: str = ""
    known_facts: list[str] = field(default_factory=list)
    calculated_values: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    conclusion: str = ""

    # ---------- 生命周期 ----------

    def reset(self, query: str) -> None:
        """新一轮推理开始：初始化锚点字段，清空可累积字段。"""
        self.original_query = query
        self.sub_goal = query  # 轻量目标：直接复用原问题（避免 LLM 维护结构化 state）
        self.known_facts = []
        self.calculated_values = []
        self.pending = []
        self.conclusion = ""

    def update(self, tool_results: list[dict], query: str = "") -> None:
        """
        每轮结束后确定性重建 state（无需 reducer）。

        - original_query / sub_goal / conclusion 跨轮保留；
        - known_facts / calculated_values / pending 每轮从全量 ``tool_results`` 重算
          （tool_results 本身随推理累积，因此重算结果天然是累积态）。
        """
        if query and not self.original_query:
            self.original_query = query
            self.sub_goal = query

        self.known_facts = []
        self.calculated_values = []
        self.pending = []

        for tr in tool_results:
            if tr.get("status") != "success":
                continue
            name = tr.get("tool_name", "")
            result = tr.get("result", "") or ""
            args = tr.get("tool_args", {}) or {}
            if name == _CALC:
                self._extract_calc(result)
            elif name == _KB:
                # KB 在 known_facts 中只保留「指针」：全文已由 _render_kb_sources
                # 常驻区块（## 检索原文）完整承载，不再在要点表重复全文以省 token。
                self._extract_fact("检索", args.get("query", ""), result, full=False)
            elif name == _FILE:
                key = args.get("file_path") or args.get("filename") or ""
                self._extract_fact("读取文件", key, result)

        self._derive_pending(tool_results)

    # ---------- 抽取 ----------

    def _extract_calc(self, result: str) -> None:
        for line in result.splitlines():
            line = line.strip()
            m = _CALC_VALUE_RE.search(line)
            if not m:
                continue
            expr_part = line[: line.rfind("=")].strip()
            value = m.group(1)
            compact = f"{expr_part} = {value}".strip(" :")
            if compact and compact not in self.calculated_values:
                self.calculated_values.append(compact)

    def _extract_fact(self, verb: str, key: str, result: str, full: bool = True) -> None:
        # [2026-10-13] 移除「单行截断」后，file_reader 等需留存原文的内容仍保持全量写入
        # （full=True，不做长度截断，避免长文件跨轮丢失）。
        # 但 knowledge_search 的全文已由 _render_kb_sources 常驻区块（## 检索原文，最近 4 次
        # 检索完整结果、无截断）承载，为避免与 known_facts 要点表重复占用大量 token，KB 走
        # full=False：only 保留「指针」（检索词 + 指向原文区块的提示），不重复全文。
        label = f"{verb}「{key}」" if key else verb
        if full:
            text = " ".join((result or "").split())
            if not text:
                return
            fact = f"{label} → {text}"
        else:
            # KB 指针：不重复全文，仅提示「已检索 + 原文位置」，模型可按需回看区块。
            fact = (
                f"{label} → [已检索，全文见下方「## 检索原文」区块"
                "（最近 4 次 knowledge_search 完整结果）]"
            )
        if fact not in self.known_facts:
            self.known_facts.append(fact)

    # ---------- 待补齐提示（代码派生，非 LLM） ----------

    def _derive_pending(self, tool_results: list[dict]) -> None:
        calc_n = sum(
            1
            for tr in tool_results
            if tr.get("tool_name") == _CALC and tr.get("status") == "success"
        )
        kb_n = sum(
            1 for tr in tool_results if tr.get("tool_name") == _KB and tr.get("status") == "success"
        )

        # 已算出多个数值却未做比较/排序/聚合 → 直接修「五大区」类 bug
        if calc_n >= 2 and not self._has_comparison(tool_results):
            self.pending.append(
                f"你已计算出 {calc_n} 个数值，请对它们进行比较 / 排序 / 汇总，"
                "以回答用户关于排名或对比的问题。"
            )

        # 已检索到事实，但问题涉及数值计算却一次计算器都没调
        if kb_n >= 1 and calc_n == 0 and self._query_needs_calc():
            self.pending.append("问题涉及数值计算或对比，请调用 calculator 完成计算后再作答。")

    @staticmethod
    def _has_comparison(tool_results: list[dict]) -> bool:
        for tr in tool_results:
            if tr.get("tool_name") != _CALC:
                continue
            args = tr.get("tool_args", {}) or {}
            # 注意：expressions / steps 是 list/dict，需整体转 str 才能扫到 max/min 等动作
            blob = " ".join(str(v) for v in args.values())
            if _COMPARE_RE.search(blob):
                return True
        return False

    def _query_needs_calc(self) -> bool:
        return bool(_CALC_HINT_RE.search(self.original_query or ""))

    # ---------- 渲染（注入 system message） ----------

    def render_for_prompt(self) -> str:
        lines = ["## 当前任务状态（每轮自动归集，请勿重复已完成的检索 / 计算）"]
        lines.append(f"- 原始问题：{self.original_query}")
        if self.sub_goal and self.sub_goal != self.original_query:
            lines.append(f"- 目标：{self.sub_goal}")
        if self.known_facts:
            lines.append("- 已知事实（已检索 / 读取）：")
            for i, f in enumerate(self.known_facts, 1):
                lines.append(f"  {i}. {f}")
        if self.calculated_values:
            lines.append("- 已计算数值：")
            for i, c in enumerate(self.calculated_values, 1):
                lines.append(f"  {i}. {c}")
        if self.pending:
            lines.append("- 待补齐（请关注）：")
            for p in self.pending:
                lines.append(f"  - {p}")
        concl = self.conclusion or "（推理进行中，尚未得出最终结论）"
        if self.conclusion and _is_refusal(self.conclusion):
            # 拒答类结论标记为"软结论"：可被本轮新检索证据推翻，不进硬事实区，
            # 避免多轮会话里旧拒答被当成既定事实反复锚定（如"历史对话已明确指出"）。
            lines.append(f"- 当前结论（软结论，可被本轮新证据推翻）：{concl}")
            lines.append(
                "  提示：该结论来自前轮检索；若本轮 knowledge_search 检索到相关新文档，"
                "请以新证据为准，不要沿用旧拒答。"
            )
        else:
            lines.append(f"- 当前结论：{concl}")
        return "\n".join(lines)
