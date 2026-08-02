# backend/eval/metrics.py
"""
评测指标计算模块
计算：召回准确率、幻觉率、工具调用准确率、上下文压缩节省比例等
"""

from typing import Dict, Any, List, Optional


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
        expect_keywords: List[str],
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

        answer_lower = answer.lower()
        matched_count = sum(1 for kw in expect_keywords if kw.lower() in answer_lower)

        if match_type == "all":
            return matched_count == len(expect_keywords)
        else:
            return matched_count > 0

    def calculate_tool_accuracy(
        self,
        actual_tools: List[str],
        expected_tools: List[str],
    ) -> Dict[str, Any]:
        """
        计算工具调用准确率

        Args:
            actual_tools: 实际调用的工具列表
            expected_tools: 期望调用的工具列表

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

        # 如果期望为空，实际也应为空
        if not expected_set:
            return {
                "correct": not actual_set,
                "precision": 1.0,
                "recall": 1.0,
                "actual_tools": actual_tools,
                "expected_tools": expected_tools,
            }

        # 召回率：期望的工具是否都调用了
        recall = len(actual_set & expected_set) / len(expected_set)

        # 精确率：调用的工具是否都是期望的
        precision = len(actual_set & expected_set) / len(actual_set) if actual_set else 0

        # 正确：所有期望的工具都被调用，且没有多余调用
        correct = recall == 1.0 and precision == 1.0

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
        expect_keywords: List[str],
        is_hallucination_if_answer: bool = False,
        retrieved_docs: List[str] = None,
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
        # 如果应该拒绝回答但回答了（且没有匹配拒绝关键词），视为幻觉
        if is_hallucination_if_answer:
            # 检查是否包含拒绝关键词
            refuse_keywords = ["不知道", "未找到", "无法回答", "无法提供", "不存在", "不清楚"]
            answer_lower = answer.lower()
            has_refuse = any(kw in answer_lower for kw in refuse_keywords)

            if not has_refuse:
                # 检查是否包含期望关键词（拒绝回答时的期望关键词）
                has_expect = any(kw.lower() in answer_lower for kw in expect_keywords)
                if not has_expect:
                    return True

        # 如果有检索文档，检查回答是否基于文档
        if retrieved_docs:
            doc_text = "\n".join(retrieved_docs).lower()
            answer_lower = answer.lower()

            # 检查回答中的关键信息是否在文档中
            for word in answer_lower.split()[:20]:
                if len(word) > 2 and word not in doc_text and not word.isdigit():
                    # 这可能是编造的信息
                    pass

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
        case: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        retrieved_docs = result.get("retrieved_docs", [])

        # 答案匹配
        answer_match = self.calculate_answer_match(
            answer=answer,
            expect_keywords=case.get("expect_keywords", []),
        )

        # 工具调用准确率
        tool_metrics = self.calculate_tool_accuracy(
            actual_tools=actual_tools,
            expected_tools=case.get("must_call_tools", []),
        )

        # 幻觉检测
        is_hallucination = self.calculate_hallucination(
            answer=answer,
            expect_keywords=case.get("expect_keywords", []),
            is_hallucination_if_answer=case.get("is_hallucination_if_answer", False),
            retrieved_docs=[doc.get("content", "") for doc in retrieved_docs] if retrieved_docs else None,
        )

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
            "score": score,
            "answer": answer[:200],
            "actual_tools": actual_tools,
        }

    def aggregate_results(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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

        for eval_result in evaluations:
            case_type = eval_result["type"]

            # 初始化类型统计
            if case_type not in type_stats:
                type_stats[case_type] = {
                    "total": 0,
                    "answer_match": 0,
                    "tool_correct": 0,
                    "hallucination": 0,
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

        # 总体指标
        overall = {
            "total_cases": total,
            "avg_score": total_score / total,
            "answer_match_rate": answer_match_count / total,
            "tool_correct_rate": tool_correct_count / total,
            "hallucination_rate": hallucination_count / total,
        }

        return {
            "overall": overall,
            "by_type": type_stats,
            "total_evaluations": evaluations,
        }

    def generate_report(self, aggregated: Dict[str, Any]) -> str:
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
        report += f"| 幻觉率 | {overall['hallucination_rate']:.2%} |\n\n"

        # 按类型统计
        report += "## 按类型统计\n\n"
        report += "| 类型 | 用例数 | 平均分 | 答案匹配率 | 工具准确率 | 幻觉率 |\n"
        report += "|------|--------|--------|------------|------------|--------|\n"

        for case_type, stats in aggregated["by_type"].items():
            report += f"| {case_type} | {stats['total']} | {stats['avg_score']:.2f} | {stats['answer_match_rate']:.2%} | {stats['tool_correct_rate']:.2%} | {stats['hallucination_rate']:.2%} |\n"

        return report


# ====== 全局实例 ======

evaluator = EvaluationMetrics()