# backend/app/rag/query_rewrite_service.py
"""
Query 改写（查询增强）服务

把 HyDE / 查询扩展 / 子问题分解 等查询增强技术统一封装，
并提供一个「智能路由器」：
- 规则优先：用查询长度、疑问词、对比词等廉价特征判定改写方式（零额外 token）
- LLM 兜底：规则置信度低时，再调一次廉价的分类调用决定

改写与底层检索策略（vector/bm25/hybrid）正交：
改写只产出"更好的 query 文本"，检索仍按用户配置的策略执行。

路由返回值：
    "none"      -> 不改写，原样检索（事实型短查询，省成本）
    "hyde"      -> 生成假设性文档，用该文档做检索
    "expand"    -> 查询扩展（同义/相关术语），用扩展后 query 检索
    "decompose" -> 拆成子问题，每个子问题分别检索后合并
"""

import json
import os
import re

from langchain_core.messages import HumanMessage

from backend.app.common.logger import logger
from backend.app.utils.models_factory import model_factory

# 路由方法枚举
METHOD_NONE = "none"
METHOD_HYDE = "hyde"
METHOD_EXPAND = "expand"
METHOD_DECOMPOSE = "decompose"

_VALID_METHODS = {METHOD_NONE, METHOD_HYDE, METHOD_EXPAND, METHOD_DECOMPOSE}

# 规则判定触发 LLM 兜底的分类置信度阈值（低于该值则用 LLM 再判）
_RULE_LLM_FALLBACK_THRESHOLD = 0.5

# 改写 LLM 护栏（2026-09-18）：限制单次改写输出 token 数，防 runaway / 卡死；
# 单行输出类改写（expand / decompose / classify）再追加 stop=["\n"] 截断尾随寒暄。
_QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "256"))


def _bind_rewrite_guardrails(llm, stop: list[str] | None = None):
    """为改写 LLM 调用绑定输出护栏（max_tokens + 可选 stop）。"""
    kwargs: dict = {"max_tokens": _QUERY_REWRITE_MAX_TOKENS}
    if stop:
        kwargs["stop"] = stop
    return llm.bind(**kwargs)


class QueryRewriteService:
    """
    Query 改写的三种具体实现
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """懒加载 LLM（默认用本地模型，temperature=0 保证确定性）"""
        if self._llm is None:
            self._llm = model_factory.get_llm_model(provider="local", temperature=0)
        return self._llm

    # ====== HyDE：生成假设性文档 ======

    def _build_hyde_prompt(self, query: str) -> str:
        return f"""请针对以下问题，写一段详细、专业的回答文档。
要求：
1. 以专业文档的风格撰写，而非对话风格
2. 包含具体的事实、数据、术语
3. 内容要详实，不少于200字
4. 不要说"假设"、"可能"等不确定的词语，直接以确定性的语气撰写

问题：{query}

请直接输出文档内容："""

    async def hyde(self, query: str) -> str:
        """
        HyDE：先用 LLM 针对 query 生成一篇假设性文档，
        返回该文档文本，后续用它去做向量检索。
        """
        try:
            prompt = self._build_hyde_prompt(query)
            response = await _bind_rewrite_guardrails(self.llm).ainvoke(
                [HumanMessage(content=prompt)]
            )
            doc = response.content.strip()
            if not doc:
                logger.warning("HyDE 生成假设文档为空，退化为原 query")
                return query
            return doc
        except Exception as e:
            logger.warning(f"HyDE 改写失败，退化为原 query: {e}")
            return query

    # ====== 查询扩展 ======

    def _build_expand_prompt(self, query: str) -> str:
        return f"""你是一个检索查询扩展助手。给定用户的检索问题，请生成一组
与原问题语义相关的同义词、近义词、缩写、上下位术语，用于提升检索召回率。

规则：
1. 只输出扩展词，用中文逗号分隔，不要解释
2. 扩展词数量控制在 3-8 个
3. 不要重复原问题本身

原问题：{query}

扩展词："""

    async def expand(self, query: str) -> str:
        """
        查询扩展：LLM 生成同义/相关术语，拼回原 query。
        """
        try:
            prompt = self._build_expand_prompt(query)
            response = await _bind_rewrite_guardrails(self.llm, stop=["\n"]).ainvoke(
                [HumanMessage(content=prompt)]
            )
            extra = response.content.strip()
            if not extra:
                return query
            # 拼回原 query（保留原意，扩展词作为补充信号）
            return f"{query} {extra}"
        except Exception as e:
            logger.warning(f"查询扩展失败，退化为原 query: {e}")
            return query

    # ====== 子问题分解 ======

    def _build_decompose_prompt(self, query: str) -> str:
        return f"""你是一个问题拆解助手。 ONLY 当用户问题确实包含多个独立子问题时，才拆成若干可被单独检索的子问题。

哪些情况**不应该**拆分：
- 单一计算题（如“Q2 的 LTV/CAC 比率是多少？”）
- 一个主问题带一个判断/追问（如“...达标了吗？”、“...健康吗？”）
- 问题中只是附带了一个已知事实或判断基准（如“行业一般认为 3 以上算健康”）

拆分规则：
1. 每个子问题必须保留原问题中的关键限定条件：时间（如 Q2）、主体（如星辰科技）、指标口径、对比维度等
2. 不要把陈述事实改写成无意义的疑问句；只把真正需要检索的信息拆出来
3. 子问题数量 2-3 个，不要过度拆解
4. 只输出 JSON 数组，例如：["子问题1", "子问题2"]，不要任何额外文本
5. 如果问题不需要拆分，直接输出空数组 []

用户问题：{query}

子问题 JSON 数组："""

    async def decompose(self, query: str) -> list[str]:
        """
        子问题分解：LLM 把复杂问题拆成多个子问题，返回子问题列表。
        失败时返回空列表（调用方据此退化为原 query 检索）。
        """
        try:
            prompt = self._build_decompose_prompt(query)
            response = await _bind_rewrite_guardrails(self.llm, stop=["\n"]).ainvoke(
                [HumanMessage(content=prompt)]
            )
            text = response.content.strip()
            # 容错：截取第一个 [ ... ] 片段
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                logger.warning("子问题分解未解析到 JSON 数组，退化为原 query")
                return []
            arr = json.loads(match.group(0))
            if not isinstance(arr, list):
                return []
            subs = [str(x).strip() for x in arr if str(x).strip()]
            return subs[:4]  # 限制最多 4 个子问题，控成本
        except Exception as e:
            logger.warning(f"子问题分解失败，退化为原 query: {e}")
            return []


class QueryRewriteRouter:
    """
    智能路由器：决定对某个 query 使用哪种改写方式。
    规则优先，低置信时 LLM 兜底。
    """

    def __init__(self, llm_fallback: bool = True):
        self.llm_fallback = llm_fallback
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = model_factory.get_llm_model(provider="local", temperature=0)
        return self._llm

    # ====== 规则判定 ======

    def _rule(self, query: str) -> tuple[str, float]:
        """
        基于廉价特征的规则判定。
        Returns:
            (method, confidence)  confidence ∈ [0,1]
        """
        q = (query or "").strip()
        if not q:
            return METHOD_NONE, 1.0

        question_marks = q.count("？") + q.count("?")

        # 事实型短查询：直接判定不改写，高置信
        factual_hints = [
            "工号",
            "编号",
            "身份证",
            "日期",
            "时间",
            "几个",
            "在哪",
            "哪里",
            "电话",
            "邮箱",
            "地址",
            "价格",
            "费用",
        ]
        if any(h in q for h in factual_hints):
            return METHOD_NONE, 0.9
        if len(q) <= 8:
            return METHOD_NONE, 0.8

        # 复杂计算/比例/预测问题：优先 none（单次检索获取数据后由 Agent 调用 calculator），不要分解
        # 例如：Q2 的 LTV/CAC 比率是多少？星辰科技达标了吗？
        calc_hints = [
            "中位数",
            "增速",
            "增长率",
            "同比",
            "环比",
            "比例",
            "占比",
            "人均",
            "平均",
            "比率",
            "倍数",
            "排名",
            "第几",
        ]
        has_calc_hint = any(h in q for h in calc_hints)
        if has_calc_hint:
            # 计算题用单次检索 + calculator 更稳，避免拆丢关键限定条件
            return METHOD_NONE, 0.85

        # 真正的多子问题：多个独立问号，且不是“主问题 + 判断/基准”结构
        # 排除“...是多少？行业认为X以上算健康，达标了吗？”这种单一问题
        multi_question_connectives = [
            "并且",
            "同时",
            "还有",
            "另外",
            "此外",
            "分别",
            "各自",
            "各",
            "依次",
        ]
        has_multi_connective = any(h in q for h in multi_question_connectives)
        # 只有存在多个真正独立的问号，且问题明显要求分别回答不同对象/维度时，才分解
        if question_marks >= 2 and has_multi_connective and len(q) > 20:
            return METHOD_DECOMPOSE, 0.75

        # 多跳/跨文档对比：明确包含“区别、对比、差异、VS、A和B的X”等，且问的是多个独立对象
        comparative_hints = ["区别", "对比", "比较", "不同", "差异", " vs ", "VS"]
        has_comparative = any(h in q for h in comparative_hints)
        # 必须出现两个以上可被比较的主体（用“和/与/、”连接），才分解
        if has_comparative and ("和" in q or "与" in q or "、" in q) and question_marks >= 1:
            return METHOD_DECOMPOSE, 0.72

        # 概念/原理/解释型：倾向 HyDE（用假设文档提升语义召回）
        conceptual_hints = [
            "为什么",
            "如何",
            "怎么",
            "原理",
            "机制",
            "含义",
            "是什么",
            "解释",
            "优点",
            "缺点",
            "影响",
        ]
        if any(h in q for h in conceptual_hints):
            return METHOD_HYDE, 0.7

        # 其余中等长度、无明确信号：默认扩展（置信低，交给 LLM 兜底再判）
        return METHOD_EXPAND, 0.4

    # ====== LLM 兜底分类 ======

    def _build_classify_prompt(self, query: str) -> str:
        return f"""你是一个检索查询改写方式选择器。根据用户的问题，
从以下四种方式中选择最合适的一种：

- none：事实型、短小、可直接检索的问题；或需要数值计算的问题（获取数据后由 calculator 计算，不需要拆分子问题）
- hyde：概念/解释/原理型问题，适合先生成假设性文档再检索
- expand：模糊、用词宽泛、需要补充同义/相关术语的问题
- decompose： ONLY 当问题确实包含多个独立子问题，或需要对比多个独立对象/维度时才选择；单一计算题、主问题带判断/基准的不要选 decompose

判断要点：
1. 如果问题涉及 LTV/CAC、比率、增速、占比、中位数、人均、排名等计算，优先选 none
2. 如果问题是“主问题 + 达标了吗/健康吗/对吗”这类单一判断，优先选 none
3. 如果问题明确说“分别/各自/各...如何”或有多个独立问号问不同对象，才选 decompose

只输出一个 JSON 对象，不要任何额外文本，例如：{{"method":"hyde"}}
可选 method 值：none / hyde / expand / decompose

用户问题：{query}

选择结果："""

    async def _llm_classify(self, query: str) -> str:
        try:
            prompt = self._build_classify_prompt(query)
            response = await _bind_rewrite_guardrails(self.llm, stop=["\n"]).ainvoke(
                [HumanMessage(content=prompt)]
            )
            text = response.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return METHOD_NONE
            obj = json.loads(match.group(0))
            method = obj.get("method", METHOD_NONE)
            return method if method in _VALID_METHODS else METHOD_NONE
        except Exception as e:
            logger.warning(f"LLM 改写路由分类失败，退化为 none: {e}")
            return METHOD_NONE

    # ====== 对外路由入口 ======

    async def route(self, query: str) -> str:
        """
        返回改写方式：none / hyde / expand / decompose
        """
        method, confidence = self._rule(query)
        if self.llm_fallback and confidence < _RULE_LLM_FALLBACK_THRESHOLD:
            logger.info(f"改写路由规则置信度低({confidence:.2f})，启用 LLM 兜底分类")
            method = await self._llm_classify(query)
        return method


# ====== 全局单例 ======

query_rewrite_service = QueryRewriteService()
query_rewrite_router = QueryRewriteRouter(llm_fallback=True)
