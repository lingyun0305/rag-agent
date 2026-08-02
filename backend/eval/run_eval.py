# backend/eval/run_eval.py
"""
自动化评测脚本
运行所有测试用例，生成评测报告
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from backend.app.agent.agent import agentic_rag
from backend.app.agent.context.compressor import context_compressor
from backend.eval.metrics import evaluator


class AutoEvaluator:
    """
    自动化评测器

    功能：
    - 运行所有测试用例
    - 支持上下文压缩策略对比
    - 生成详细评测报告
    """

    def __init__(self, llm_provider: str = "qwen"):
        self.llm_provider = llm_provider

    def load_test_cases(self, path: str = "backend/eval/testset.json") -> dict:
        """
        加载测试用例集

        Args:
            path: 测试用例文件路径

        Returns:
            测试用例数据
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run_case(
        self,
        case: dict,
        compression_strategy: str = None,
    ) -> dict:
        """
        运行单个测试用例

        Args:
            case: 测试用例
            compression_strategy: 上下文压缩策略

        Returns:
            测试结果
        """
        query = case["query"]
        history = case.get("context", [])

        # 应用上下文压缩（如果指定）
        if compression_strategy and history:
            compressed = await context_compressor.compress(
                history, strategy=compression_strategy
            )
            history = compressed["compressed_messages"]
        else:
            compressed = None

        # 调用 Agent
        try:
            result = await agentic_rag.run(
                query=query,
                user_id=1,
                session_id=f"eval_{case['id']}",
                history=history,
            )
            called_tools = result.get("called_tools", [])
            answer = result.get("answer", "")
            retrieved_docs = []

            return {
                "case_id": case["id"],
                "query": query,
                "answer": answer,
                "called_tools": called_tools,
                "retrieved_docs": retrieved_docs,
                "success": True,
                "compression": compressed,
            }

        except Exception as e:
            return {
                "case_id": case["id"],
                "query": query,
                "answer": f"ERROR: {str(e)}",
                "called_tools": [],
                "retrieved_docs": [],
                "success": False,
                "compression": compressed,
            }

    async def run_all_cases(
        self,
        compression_strategy: str = None,
        filter_type: str = None,
    ) -> dict:
        """
        运行所有测试用例

        Args:
            compression_strategy: 上下文压缩策略
            filter_type: 按类型过滤（可选）

        Returns:
            评测结果
        """
        test_data = self.load_test_cases()
        cases = test_data["test_cases"]

        # 按类型过滤
        if filter_type:
            cases = [c for c in cases if c["type"] == filter_type]
            print(f"过滤后用例数: {len(cases)} (类型: {filter_type})")

        print(f"\n{'='*60}")
        print(f"开始评测 - 压缩策略: {compression_strategy}")
        print(f"用例总数: {len(cases)}")
        print(f"{'='*60}")

        results = []
        evaluations = []

        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {case['id']} - {case['type']}")
            print(f"  查询: {case['query']}")

            result = await self.run_case(case, compression_strategy=compression_strategy)
            results.append(result)

            # 评估
            eval_result = evaluator.evaluate_single_case(case, result)
            evaluations.append(eval_result)

            print(f"  分数: {eval_result['score']}/100")
            print(f"  答案匹配: {'✓' if eval_result['answer_match'] else '✗'}")
            print(f"  工具正确: {'✓' if eval_result['tool_accuracy']['correct'] else '✗'}")
            print(f"  幻觉: {'✓' if eval_result['is_hallucination'] else '✗'}")

        # 聚合结果
        aggregated = evaluator.aggregate_results(evaluations)

        return {
            "metadata": {
                "compression_strategy": compression_strategy,
                "llm_provider": self.llm_provider,
                "timestamp": datetime.now().isoformat(),
                "total_cases": len(cases),
                "filter_type": filter_type,
            },
            "aggregated": aggregated,
            "results": results,
            "evaluations": evaluations,
        }

    async def compare_compression_strategies(self) -> dict:
        """
        对比不同上下文压缩策略的效果

        Returns:
            对比结果
        """
        strategies = [None, "sliding_window", "summary"]

        all_results = {}
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"评测压缩策略: {strategy or '无压缩'}")
            print(f"{'='*60}")

            result = await self.run_all_cases(
                compression_strategy=strategy,
                filter_type="long_dialog",  # 只测试长对话
            )
            all_results[strategy or "none"] = result

        return all_results

    def generate_full_report(self, results: dict, filename: str = None) -> str:
        """
        生成完整评测报告

        Args:
            results: 评测结果
            filename: 输出文件名（可选）

        Returns:
            Markdown 报告
        """
        report = "# Agentic RAG 智能体评测报告\n\n"
        report += f"**生成时间**: {datetime.now().isoformat()}\n\n"
        report += f"**LLM 提供商**: {self.llm_provider}\n\n"

        # 总体指标
        if "aggregated" in results:
            report += evaluator.generate_report(results["aggregated"])
        elif isinstance(results, dict):
            # 多组对比结果
            for name, data in results.items():
                report += f"## 测试组: {name}\n\n"
                if "aggregated" in data:
                    report += evaluator.generate_report(data["aggregated"])
                report += "\n---\n\n"

        # 保存报告
        if filename:
            output_path = Path(filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n报告已保存到: {output_path}")

        return report


async def main():
    """主函数"""
    print("="*60)
    print("Agentic RAG 自动化评测")
    print("="*60)

    evaluator = AutoEvaluator(llm_provider="qwen")

    # 1. 运行完整评测（单智能体模式）
    print("\n【步骤1】运行完整评测（单智能体模式）")
    full_result = await evaluator.run_all_cases(mode="single")

    # 2. 生成报告
    print("\n【步骤2】生成评测报告")
    report = evaluator.generate_full_report(full_result, "backend/eval/report.md")
    print("\n" + "="*60)
    print("评测报告摘要")
    print("="*60)
    # 打印前50行报告
    for line in report.split("\n")[:50]:
        print(line)

    # 3. 保存详细结果
    output_path = Path("backend/eval/evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")

    # 4. 可选：对比压缩策略（耗时较长，默认跳过）
    # print("\n【步骤3】对比上下文压缩策略（长对话测试）")
    # compression_result = await evaluator.compare_compression_strategies()
    # compression_report = evaluator.generate_full_report(compression_result, "backend/eval/compression_report.md")


if __name__ == "__main__":
    asyncio.run(main())