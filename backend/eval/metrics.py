# backend/eval/metrics.py
"""
评测指标计算模块
计算：召回准确率、幻觉率、工具调用准确率、上下文压缩节省比例等
"""

import re
from typing import Any


class EvaluationMetrics:
    """
    评测指标计算器

    指标：
    1. 答案匹配率（Answer Match Rate）
    2. 工具调用准确率（Tool Call Accuracy）
    3. 幻觉率（Hallucination Rate）
    4. 上下文压缩节省比例（Context Compression Ratio）
    5. 平均响应时间
    6. 综合评分
    """

    def __init__(self):
        self.metrics = {}

    def calculate_answer_match(
        self,
        answer: str,
        expect_keywords: list[str],
        match_type: str = "any",
    ) -> bool:
        """
        计算答案匹配率

        Args:
            answer: 实际回答
            expect_keywords: 期望关键词列表
            match_type: 匹配类型（any: 任一匹配, all: 全部匹配）

        Returns:
            是否匹配
        """
        if not expect_keywords:
            return True

        # 归一化空白后再做子串匹配（解决 '7个' vs '7 个'、'97万' vs '97 万' 这类
        # 因答案中数字与单位/量词间多出空格导致的误判）。
        answer_norm = re.sub(r"\s+", "", (answer or "").lower())
        matched_count = sum(
            1 for kw in expect_keywords if re.sub(r"\s+", "", kw.lower()) in answer_norm
        )

        if match_type == "all":
            return matched_count == len(expect_keywords)
        else:
            return matched_count > 0

    def calculate_tool_accuracy(
        self,
        actual_tools: list[str],
        expected_tools: list[str],
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        计算工具调用准确率

        评分口径（v1.4 起统一为「期望 ⊆ 实际」）：
        - 期望集 ⊆ 实际集（recall==1.0）即通过，统一测"会不会用工具"；
          允许合理附加调用（如检索后顺手用 calculator 换算百分比）；
          期望为空时恒为正确（模型可先检索确认 KB 静默再拒答）。
        - 撤销 v1.3 的 tool_call「严格集合相等」特例：评测集已确认无误，
          宽松口径更能反映真实工具使用能力，历史分数不可比亦可接受。
        - strict 仅保留为调试开关：评测调用方统一传 False（recall 口径）；
          如需回归"集合相等"严格口径，可显式传 strict=True。

        Args:
            actual_tools: 实际调用的工具列表
            expected_tools: 期望调用的工具列表
            strict: 是否使用严格集合相等口径

        Returns:
            {
                "correct": bool,
                "precision": float,
                "recall": float,
                "actual_tools": list,
                "expected_tools": list,
            }
        """
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        # 期望为空：strict 口径要求实际也为空（没要求调用就别调）；
        # 宽松口径恒为正确（允许先检索确认再拒答）。
        if not expected_set:
            correct = (not actual_set) if strict else True
            return {
                "correct": correct,
                "precision": 1.0,
                "recall": 1.0,
                "actual_tools": actual_tools,
                "expected_tools": expected_tools,
            }

        # 召回率：期望的工具是否都调用了
        recall = len(actual_set & expected_set) / len(expected_set)

        # 精确率：调用的工具是否都是期望的
        precision = len(actual_set & expected_set) / len(actual_set) if actual_set else 0

        # 正确：strict 要求无多余调用（集合相等）；宽松只要求期望全被调用（期望⊆实际）
        correct = recall == 1.0 and (precision == 1.0 if strict else True)

        return {
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
        }

    def calculate_hallucination(
        self,
        answer: str,
        expect_keywords: list[str],
        is_hallucination_if_answer: bool = False,
        retrieved_docs: list[str] = None,
    ) -> bool:
        """
        判断是否存在幻觉

        Args:
            answer: 实际回答
            expect_keywords: 期望关键词（用于判断是否"拒绝回答"）
            is_hallucination_if_answer: 如果应该拒绝但回答了，视为幻觉
            retrieved_docs: 检索到的文档（用于检查事实依据）

        Returns:
            是否存在幻觉
        """
        # 仅对「该拒答却作答」类用例判定幻觉（is_hallucination_if_answer=True）。
        # 修复（2026-09-16）：原实现仅用 13 个硬编码拒绝词，模型用其它合规措辞拒答
        # （如"我没有找到相关信息"）会被误判为幻觉扣 20 分。现改为：
        #   1) 优先命中用例自带的 expect_keywords（已按用例定制的拒绝措辞）；
        #   2) 兜底通用拒绝词（扩写原 13 词，覆盖同义说法）；
        #   3) 仅当既未命中任何拒绝措辞、又给出实质内容时，才判「该拒答却作答」= 幻觉。
        if not is_hallucination_if_answer:
            return False

        answer_lower = (answer or "").lower()

        # 1) 用例自带的期望拒答词（最权威，已覆盖各用例特有措辞）
        if any(kw.lower() in answer_lower for kw in (expect_keywords or [])):
            return False

        # 2) 通用拒绝词兜底（覆盖不在用例 expect_keywords 里的同义拒答措辞）
        refuse_keywords = [
            "不知道",
            "未找到",
            "无法回答",
            "无法提供",
            "不能提供",
            "不存在",
            "不清楚",
            "未提及",
            "未查询到",
            "暂无相关信息",
            "暂无",
            "未提供相关信息",
            "没有相关信息",
            "没有找到",
            "找不到",
            "没有相关",
            "无法获取",
            "未公开",
            "无权限",
            "不在",
            "超出",
            "未知",
        ]
        if any(kw in answer_lower for kw in refuse_keywords):
            return False

        # 3) 既没拒答也没命中期望关键词 → 视为该拒答却作答（幻觉）
        return True

    def calculate_grounded_hallucination(
        self,
        answer: str,
        expect_keywords: list[str],
        retrieved_docs: list[str] = None,
    ) -> bool:
        """
        实验性·事实接地幻觉检测（v1.3 起计入综合评分的「无幻觉」项，20 分）。

        针对「非拒答类」用例（knowledge_query / file_read 等）：抽取答案中的具体
        数值/数量短语，凡未出现在检索文档、也不在期望关键词中的，视为疑似编造。

        v1.3 降误报（9/17 基线回放发现 3 类假阳性）：
        1. 逗号切断：`3,850` 被正则切成 `3` + `850万` → 双侧统一去掉千分位逗号；
        2. 单位换算：答案 `970,000.00元` vs 文档 `97万元` → 数字等价检查（×万/÷万）；
        3. 派生数值：Agent 是计算主体，`650万 = 3,850 - 3,200` 这类由文档数字经
           加减乘除/百分比得到的值是合法输出，不应判编造 → 对文档数字做两两算术匹配。

        本方法不判断「该拒答却答了」类幻觉（由 calculate_hallucination 负责）。
        """
        if not retrieved_docs:
            return False
        ans = (answer or "").lower()
        if not ans.strip():
            return False
        doc_text = "\n".join(
            d.get("content", "") if isinstance(d, dict) else str(d) for d in retrieved_docs
        ).lower()
        # v1.3：双侧去掉千分位逗号（答案侧须在抽取前清洗，否则正则照样在逗号处切断）
        ans = ans.replace(",", "")
        doc_text_nc = doc_text.replace(",", "")
        # v1.3：文档数字清单（无单位浮点值），用于等价/算术检查；年份（1900~2100）
        # 单独标记——年份不是业务数值，参与两两算术会拼出假等价（如 3200+2025≈5200）
        doc_nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", doc_text_nc)]
        doc_nums_arith = [d for d in doc_nums if not (1900 <= d <= 2100)]

        # 抽取答案中的数字+单位片段，如 '10万' '10800元' '97万元' '3.5倍'
        claims = re.findall(
            r"\d+(?:\.\d+)?\s*(?:万|亿|千|倍|%|％|元|个|条|名|人|天|次|年|月|日|小时|分)?",
            ans,
        )
        if not claims:
            return False
        kw_set = set(
            re.sub(r"\s+", "", kw.lower()).replace(",", "") for kw in (expect_keywords or [])
        )
        units = "万|亿|千|倍|%|％|元|个|条|名|人|天|次|年|月|日|小时|分"
        unit_mult = {"万": 10000.0, "亿": 100000000.0, "千": 1000.0}

        def _num_ok(val: float) -> bool:
            """数字等价/派生检查：单位换算等价 或 文档数字两两算术可导出。

            文档数字与答案数字可能处于不同尺度（答案'650万'=6.5e6，文档记作
            3850/3200），故等价与算术均跨 {val, val/万, val×万} 三种尺度匹配。
            算术豁免仅对 val>=100 开放：小整数（如人数 6 vs 7）恰是编造检测的
            重点，不能因为文档里凑巧存在 a+b=7 就放行。
            """
            if not doc_nums:
                return False
            # 候选尺度：原始值、万制、亿制（亿制仅当 val 本就是亿级——
            # 对万制文档把 5.2e7 除到 0.52 只会制造小数值假等价）
            cands = [val, val / 10000.0, val * 10000.0]
            if val >= 1e8:
                cands.append(val / 100000000.0)
            # ① 等价：原始值 或 万/亿换算后的值 直接出现在文档
            for cand in cands:
                if any(abs(d - cand) <= max(abs(cand) * 1e-3, 1e-6) for d in doc_nums):
                    return True
            # ② 派生：两两算术（±、×、÷、÷×100 百分比），仅对 ≥100 的数值开放。
            # 算术只在业务常用的两个尺度（原始值/万制）上做——亿制 cand 太小，
            # 容差下容易被文档小数字的比值蹭上（如 20.3/6.34≈3.2 冒充 3.2亿）。
            # 容差 0.2%：合法派生值只带舍入级误差。
            if val < 100:
                return False
            for cand in (val, val / 10000.0):
                for a in doc_nums_arith:
                    for b in doc_nums_arith:
                        if b == 0:
                            continue
                        for derived in (a + b, a - b, a * b, a / b, a / b * 100.0):
                            if abs(derived - cand) <= max(abs(cand) * 2e-3, 1e-6):
                                return True
            return False

        for c in claims:
            cn = re.sub(r"\s+", "", c).replace(",", "")
            if cn in kw_set:
                continue
            if cn in doc_text_nc:
                continue
            base = re.sub(units, "", cn)
            if base:
                if base.isdigit():
                    # 纯数字 base 须整词匹配（带数字边界），否则 '7' 会命中 '78%'
                    if re.search(rf"(?<!\d){re.escape(base)}(?!\d)", doc_text_nc):
                        continue
                elif base in doc_text_nc:
                    continue
            # v1.3：数值路径——单位换算等价 / 文档数字算术派生
            m = re.match(r"(\d+(?:\.\d+)?)", cn)
            if m:
                unit_char = cn[len(m.group(1)) : len(m.group(1)) + 1]
                val = float(m.group(1))
                if unit_char in unit_mult:
                    val *= unit_mult[unit_char]
                if _num_ok(val):
                    continue
            return True  # 找到一个明确数值断言在来源文档中找不到 → 疑似编造
        return False

    def calculate_compression_ratio(
        self,
        original_tokens: int,
        compressed_tokens: int,
    ) -> float:
        """
        计算上下文压缩节省比例

        Args:
            original_tokens: 压缩前 token 数
            compressed_tokens: 压缩后 token 数

        Returns:
            压缩节省比例（0-1）
        """
        if original_tokens <= 0:
            return 0.0

        saved = original_tokens - compressed_tokens
        return saved / original_tokens

    def evaluate_single_case(
        self,
        case: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        评估单个测试用例

        Args:
            case: 测试用例
            result: 测试结果

        Returns:
            评估指标
        """
        answer = result.get("answer", "")
        actual_tools = result.get("called_tools", [])
        tool_calls_detail = result.get("tool_calls_detail", []) or []
        failed_tools = [
            d.get("tool_name") for d in tool_calls_detail if d.get("status") not in ("success",)
        ]
        retrieved_docs = result.get("retrieved_docs", [])

        # 接地诊断（仅诊断，不影响评分）：期望关键词是否真的出现在检索到的文档中。
        # 用于判断 Agent 是否"真的走了 RAG"，而非凭参数记忆硬答。
        grounded = None
        if retrieved_docs and case.get("expect_keywords"):
            doc_text = "\n".join(
                doc.get("content", "") if isinstance(doc, dict) else str(doc)
                for doc in retrieved_docs
            ).lower()
            grounded = any(kw.lower() in doc_text for kw in case["expect_keywords"])

        # 答案匹配
        answer_match = self.calculate_answer_match(
            answer=answer,
            expect_keywords=case.get("expect_keywords", []),
        )

        # 工具调用准确率（v1.4 统一口径：期望 ⊆ 实际 / recall==1.0；撤销 tool_call 严格特例）
        tool_metrics = self.calculate_tool_accuracy(
            actual_tools=actual_tools,
            expected_tools=case.get("must_call_tools", []),
            strict=False,
        )

        # 幻觉检测
        is_hallucination = self.calculate_hallucination(
            answer=answer,
            expect_keywords=case.get("expect_keywords", []),
            is_hallucination_if_answer=case.get("is_hallucination_if_answer", False),
            retrieved_docs=(
                [
                    doc.get("content", "") if isinstance(doc, dict) else str(doc)
                    for doc in retrieved_docs
                ]
                if retrieved_docs
                else None
            ),
        )

        # 实验性·接地幻觉检测（v1.3：计入综合评分「无幻觉」项 20 分）
        # 仅对「应基于检索文档作答」的类型生效，避免把 calculator 算出的派生数值误判为幻觉。
        # 注：派生数值（文档数字算术导出）的放行逻辑在 calculate_grounded_hallucination 内部。
        GROUNDING_TYPES = {"knowledge_query", "file_read"}
        grounded_hallucination = False
        if case.get("type") in GROUNDING_TYPES:
            grounded_hallucination = self.calculate_grounded_hallucination(
                answer=answer,
                expect_keywords=case.get("expect_keywords", []),
                retrieved_docs=(
                    [
                        doc.get("content", "") if isinstance(doc, dict) else str(doc)
                        for doc in retrieved_docs
                    ]
                    if retrieved_docs
                    else None
                ),
            )

        # ===== 修复版「幻觉/编造」判定 =====
        # 原 calculate_hallucination 仅在 is_hallucination_if_answer=True 的 12 条拒答类上
        # 真判定，其余 58 条恒为 False（分母却用 70），导致 knowledge_query/file_read
        # 类真编造（答案数字在检索文档里根本找不到）被完全漏报，20 分"无幻觉"形同虚设。
        # 修复：拒答类保留原拒答关键词校验；事实接地类改用「接地幻觉检测」
        # （答案中的数值性断言在来源文档中找不到才算疑似编造）；其余不涉及文档事实的
        # 类型（calculation/tool_call/long_dialog/fuzzy_query/mixed）不判幻觉，避免误扣。
        if case.get("is_hallucination_if_answer"):
            pass  # is_hallucination 已是拒答校验结果
        elif case.get("type") in GROUNDING_TYPES:
            is_hallucination = grounded_hallucination
        else:
            is_hallucination = False

        # 综合评分
        score = 0
        if answer_match:
            score += 40
        if tool_metrics["correct"]:
            score += 40
        if not is_hallucination:
            score += 20

        return {
            "case_id": case["id"],
            "type": case["type"],
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", "unknown"),
            "answer_match": answer_match,
            "tool_accuracy": tool_metrics,
            "is_hallucination": is_hallucination,
            "grounded_hallucination": grounded_hallucination,
            "score": score,
            "grounded": grounded,
            "answer": answer[:200],
            "actual_tools": actual_tools,
            "tool_calls_detail": tool_calls_detail,
            "failed_tools": failed_tools,
        }

    def aggregate_results(
        self,
        evaluations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        聚合所有评估结果，计算总体指标

        Args:
            evaluations: 评估结果列表

        Returns:
            总体指标
        """
        total = len(evaluations)
        if total == 0:
            return {}

        # 按类型统计
        type_stats = {}
        total_score = 0
        answer_match_count = 0
        tool_correct_count = 0
        hallucination_count = 0
        grounded_hall_count = 0
        grounded_total = 0
        grounded_true = 0
        failed_calls_total = 0
        cases_with_failed_calls = 0

        for eval_result in evaluations:
            case_type = eval_result["type"]

            # 初始化类型统计
            if case_type not in type_stats:
                type_stats[case_type] = {
                    "total": 0,
                    "answer_match": 0,
                    "tool_correct": 0,
                    "hallucination": 0,
                    "grounded_hallucination": 0,
                    "grounded_true": 0,
                    "grounded_total": 0,
                    "avg_score": 0,
                }

            # 统计
            type_stats[case_type]["total"] += 1
            type_stats[case_type]["avg_score"] += eval_result["score"]

            if eval_result["answer_match"]:
                type_stats[case_type]["answer_match"] += 1
                answer_match_count += 1

            if eval_result["tool_accuracy"]["correct"]:
                type_stats[case_type]["tool_correct"] += 1
                tool_correct_count += 1

            if eval_result["is_hallucination"]:
                type_stats[case_type]["hallucination"] += 1
                hallucination_count += 1

            if eval_result.get("grounded_hallucination"):
                type_stats[case_type]["grounded_hallucination"] += 1
                grounded_hall_count += 1

            # 接地率统计（仅诊断）
            g = eval_result.get("grounded")
            if g is not None:
                type_stats[case_type]["grounded_total"] += 1
                if g:
                    type_stats[case_type]["grounded_true"] += 1
                grounded_total += 1
                if g:
                    grounded_true += 1

            # 失败工具调用统计（仅诊断，不影响评分）
            fcd = eval_result.get("failed_tools") or []
            failed_calls_total += len(fcd)
            if fcd:
                cases_with_failed_calls += 1

            total_score += eval_result["score"]

        # 计算平均值
        for case_type in type_stats:
            type_stats[case_type]["avg_score"] /= type_stats[case_type]["total"]
            type_stats[case_type]["answer_match_rate"] = (
                type_stats[case_type]["answer_match"] / type_stats[case_type]["total"]
            )
            type_stats[case_type]["tool_correct_rate"] = (
                type_stats[case_type]["tool_correct"] / type_stats[case_type]["total"]
            )
            type_stats[case_type]["hallucination_rate"] = (
                type_stats[case_type]["hallucination"] / type_stats[case_type]["total"]
            )
            # 接地率（仅诊断）
            if type_stats[case_type]["grounded_total"] > 0:
                type_stats[case_type]["grounding_rate"] = (
                    type_stats[case_type]["grounded_true"] / type_stats[case_type]["grounded_total"]
                )
            else:
                type_stats[case_type]["grounding_rate"] = None

        # 总体指标
        overall = {
            "total_cases": total,
            "avg_score": total_score / total,
            "answer_match_rate": answer_match_count / total,
            "tool_correct_rate": tool_correct_count / total,
            "hallucination_rate": hallucination_count / total,
            "grounded_hallucination_rate": (grounded_hall_count / total) if total else 0,
            "grounding_rate": (grounded_true / grounded_total) if grounded_total else None,
            "failed_calls_total": failed_calls_total,
            "cases_with_failed_calls": cases_with_failed_calls,
        }

        return {
            "overall": overall,
            "by_type": type_stats,
            "total_evaluations": evaluations,
        }

    def generate_report(self, aggregated: dict[str, Any]) -> str:
        """
        生成评测报告（Markdown 表格）

        Args:
            aggregated: 聚合后的指标

        Returns:
            Markdown 格式的报告
        """
        report = "# 评测报告\n\n"

        # 总体指标
        overall = aggregated["overall"]
        report += "## 总体指标\n\n"
        report += "| 指标 | 值 |\n"
        report += "|------|------|\n"
        report += f"| 总用例数 | {overall['total_cases']} |\n"
        report += f"| 平均分数 | {overall['avg_score']:.2f} |\n"
        report += f"| 答案匹配率 | {overall['answer_match_rate']:.2%} |\n"
        report += f"| 工具调用准确率 | {overall['tool_correct_rate']:.2%} |\n"
        report += f"| 幻觉率 | {overall['hallucination_rate']:.2%} |\n"
        if overall.get("grounding_rate") is not None:
            report += f"| 接地率(检索命中期望关键词) | {overall['grounding_rate']:.2%} |\n"
        if overall.get("failed_calls_total"):
            report += f"| 含失败工具调用的用例数 | {overall['cases_with_failed_calls']} (失败调用合计 {overall['failed_calls_total']} 次) |\n"
        report += f"| 实验性·接地幻觉率(非拒答类) | {overall.get('grounded_hallucination_rate', 0):.2%} |\n"
        report += "\n"

        # 按类型统计
        report += "## 按类型统计\n\n"
        report += "| 类型 | 用例数 | 平均分 | 答案匹配率 | 工具准确率 | 幻觉率 | 接地率 |\n"
        report += "|------|--------|--------|------------|------------|--------|--------|\n"

        for case_type, stats in aggregated["by_type"].items():
            gr = (
                f"{stats['grounding_rate']:.2%}" if stats.get("grounding_rate") is not None else "-"
            )
            report += f"| {case_type} | {stats['total']} | {stats['avg_score']:.2f} | {stats['answer_match_rate']:.2%} | {stats['tool_correct_rate']:.2%} | {stats['hallucination_rate']:.2%} | {gr} |\n"

        return report


# ====== 全局实例 ======

evaluator = EvaluationMetrics()
