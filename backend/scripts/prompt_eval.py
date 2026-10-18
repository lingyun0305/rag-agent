# backend/scripts/prompt_eval.py
"""
Prompt 多版本对比实验脚本
比较不同 Prompt 版本在工具调用准确率、幻觉率、回答质量等指标上的差异
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from backend.app.agent.tools.registry import init_default_tools, tool_registry
from backend.app.utils.models_factory import modelFactory

# ====== 测试用例 ======

TEST_CASES = [
    {
        "id": "knowledge_001",
        "type": "knowledge_query",
        "query": "公司的请假流程是什么？",
        "expect_keywords": ["OA", "审批", "流程"],
        "must_call_tools": ["knowledge_search"],
    },
    {
        "id": "knowledge_002",
        "type": "knowledge_query",
        "query": "项目代码提交规范是什么？",
        "expect_keywords": ["git", "commit", "规范"],
        "must_call_tools": ["knowledge_search"],
    },
    {
        "id": "calculator_001",
        "type": "calculation",
        "query": "125乘以8等于多少？",
        "expect_keywords": ["1000"],
        "must_call_tools": ["calculator"],
    },
    {
        "id": "calculator_002",
        "type": "calculation",
        "query": "(3+5)*2+10等于多少？",
        "expect_keywords": ["26"],
        "must_call_tools": ["calculator"],
    },
    {
        "id": "mixed_001",
        "type": "mixed",
        "query": "请帮我查下请假扣款规则，然后算下请5天假，日薪200，扣多少钱",
        "expect_keywords": ["扣款", "800"],
        "must_call_tools": ["knowledge_search", "calculator"],
    },
    {
        "id": "mixed_002",
        "type": "mixed",
        "query": "先查报销流程，然后帮我算下报销1000元需要多久到账",
        "expect_keywords": ["报销", "流程", "到账"],
        "must_call_tools": ["knowledge_search"],
    },
    {
        "id": "no_answer_001",
        "type": "no_answer",
        "query": "公司有没有外星人福利？",
        "expect_keywords": ["不知道", "未找到", "不清楚"],
        "must_call_tools": ["knowledge_search"],
        "is_hallucination_if_answer": True,
    },
    {
        "id": "no_answer_002",
        "type": "no_answer",
        "query": "CEO的银行卡密码是多少？",
        "expect_keywords": ["不知道", "拒绝", "无法回答"],
        "must_call_tools": [],
        "is_hallucination_if_answer": True,
    },
    {
        "id": "chat_001",
        "type": "general_chat",
        "query": "你好，你是谁？",
        "expect_keywords": ["智能体", "助手"],
        "must_call_tools": [],
    },
    {
        "id": "chat_002",
        "type": "general_chat",
        "query": "今天天气怎么样？",
        "expect_keywords": ["不知道", "无法获取", "外部信息"],
        "must_call_tools": [],
        "is_hallucination_if_answer": True,
    },
]


# ====== Prompt 版本定义 ======

PROMPT_VERSIONS = {
    "v1_baseline": {
        "name": "基础版",
        "description": "无工具调用、无 CoT、无示例",
        "path": "backend/prompts/agent_system_v1_baseline.txt",
    },
    "v2_cot": {
        "name": "CoT版",
        "description": "ReAct 模式 + CoT + 工具调用",
        "path": "backend/prompts/agent_system_v2_cot.txt",
    },
    "v3_fewshot": {
        "name": "Few-shot版",
        "description": "CoT + 工具调用 + 4个示例对话",
        "path": "backend/prompts/agent_system_v3_fewshot.txt",
    },
    "v4_advanced": {
        "name": "进阶版",
        "description": "CoT + 工具调用 + 示例 + 幻觉检测 + 追问",
        "path": "backend/prompts/agent_system_v4_advanced.txt",
    },
}


class PromptEvaluator:
    """
    Prompt 评测器
    """

    def __init__(self, llm_provider: str = "qwen", temperature: float = 0.1):
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.llm = modelFactory.get_llm_model(provider=llm_provider, temperature=temperature)
        init_default_tools()
        self.tools = tool_registry.get_langchain_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def load_prompt(self, version: str) -> str:
        """加载指定版本的 Prompt"""
        config = PROMPT_VERSIONS.get(version)
        if not config:
            raise ValueError(f"Unknown prompt version: {version}")
        path = Path(config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    async def run_single_case(
        self,
        case: dict,
        system_prompt: str,
    ) -> dict:
        """
        运行单个测试用例

        Args:
            case: 测试用例
            system_prompt: 系统提示词

        Returns:
            测试结果
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=case["query"]),
        ]

        try:
            response = await self.llm_with_tools.ainvoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = [tc["name"] for tc in response.tool_calls]

            # 评估指标
            result = {
                "case_id": case["id"],
                "query": case["query"],
                "answer": answer[:500],
                "tool_calls": tool_calls,
                "metrics": self._evaluate_case(case, answer, tool_calls),
            }
            return result

        except Exception as e:
            return {
                "case_id": case["id"],
                "query": case["query"],
                "answer": f"ERROR: {str(e)}",
                "tool_calls": [],
                "metrics": {
                    "answer_match": False,
                    "tool_correct": False,
                    "is_hallucination": False,
                    "score": 0,
                },
            }

    def _evaluate_case(
        self,
        case: dict,
        answer: str,
        tool_calls: list,
    ) -> dict:
        """
        评估单个测试用例

        Args:
            case: 测试用例
            answer: 实际回答
            tool_calls: 实际调用的工具

        Returns:
            评估指标
        """
        # 答案匹配（关键词匹配）
        expect_keywords = case.get("expect_keywords", [])
        answer_match = any(kw in answer for kw in expect_keywords)

        # 工具调用正确性
        must_call_tools = case.get("must_call_tools", [])
        tool_correct = True
        if must_call_tools:
            for tool in must_call_tools:
                if tool not in tool_calls:
                    tool_correct = False
                    break

        # 幻觉检测
        is_hallucination = False
        if case.get("is_hallucination_if_answer", False):
            # 如果不应该回答但回答了，视为幻觉
            if not answer_match:
                is_hallucination = True

        # 综合评分
        score = 0
        if answer_match:
            score += 50
        if tool_correct:
            score += 50
        if is_hallucination:
            score -= 30
        score = max(0, min(100, score))

        return {
            "answer_match": answer_match,
            "tool_correct": tool_correct,
            "is_hallucination": is_hallucination,
            "score": score,
        }

    async def run_eval(self, version: str) -> dict:
        """
        运行完整评测

        Args:
            version: Prompt 版本

        Returns:
            评测结果
        """
        system_prompt = await self.load_prompt(version)
        prompt_config = PROMPT_VERSIONS[version]

        results = []
        total_score = 0
        answer_match_count = 0
        tool_correct_count = 0
        hallucination_count = 0

        for case in TEST_CASES:
            result = await self.run_single_case(case, system_prompt)
            results.append(result)

            metrics = result["metrics"]
            total_score += metrics["score"]
            if metrics["answer_match"]:
                answer_match_count += 1
            if metrics["tool_correct"]:
                tool_correct_count += 1
            if metrics["is_hallucination"]:
                hallucination_count += 1

        avg_score = total_score / len(TEST_CASES)
        answer_match_rate = answer_match_count / len(TEST_CASES)
        tool_correct_rate = tool_correct_count / len(TEST_CASES)
        hallucination_rate = hallucination_count / len(TEST_CASES)

        return {
            "version": version,
            "name": prompt_config["name"],
            "description": prompt_config["description"],
            "prompt_length": len(system_prompt),
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "avg_score": avg_score,
                "answer_match_rate": answer_match_rate,
                "tool_correct_rate": tool_correct_rate,
                "hallucination_rate": hallucination_rate,
                "total_cases": len(TEST_CASES),
            },
            "details": results,
        }

    async def run_all_versions(self) -> dict:
        """运行所有版本的评测"""
        all_results = {}
        for version in PROMPT_VERSIONS.keys():
            print(f"正在评测: {PROMPT_VERSIONS[version]['name']}...")
            result = await self.run_eval(version)
            all_results[version] = result
            self._print_summary(result)

        return all_results

    def _print_summary(self, result: dict):
        """打印单版本评测摘要"""
        print(f"\n{'=' * 60}")
        print(f"版本: {result['version']} ({result['name']})")
        print(f"描述: {result['description']}")
        print(f"Prompt长度: {result['prompt_length']} 字符")
        print(f"平均分数: {result['overall']['avg_score']:.2f}")
        print(f"答案匹配率: {result['overall']['answer_match_rate']:.2%}")
        print(f"工具调用准确率: {result['overall']['tool_correct_rate']:.2%}")
        print(f"幻觉率: {result['overall']['hallucination_rate']:.2%}")
        print(f"{'=' * 60}\n")

    def generate_report(self, all_results: dict) -> str:
        """生成评测报告（Markdown 表格）"""
        rows = []
        for version, result in all_results.items():
            rows.append(
                [
                    version,
                    result["name"],
                    f"{result['overall']['avg_score']:.2f}",
                    f"{result['overall']['answer_match_rate']:.2%}",
                    f"{result['overall']['tool_correct_rate']:.2%}",
                    f"{result['overall']['hallucination_rate']:.2%}",
                    result["prompt_length"],
                    result["description"],
                ]
            )

        # 排序（按平均分数降序）
        rows.sort(key=lambda x: float(x[2]), reverse=True)

        header = ["版本", "平均分", "答案匹配率", "工具调用准确率", "幻觉率", "Prompt长度", "说明"]
        markdown = "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in rows:
            markdown += "| " + " | ".join(row) + " |\n"

        return markdown


async def main():
    """主函数"""
    print("=" * 60)
    print("Prompt 多版本对比实验")
    print("=" * 60)

    evaluator = PromptEvaluator(llm_provider="qwen", temperature=0.1)
    results = await evaluator.run_all_versions()

    print("\n" + "=" * 60)
    print("评测报告（Markdown表格）")
    print("=" * 60)
    report = evaluator.generate_report(results)
    print(report)

    # 保存结果
    output_path = Path("backend/scripts/prompt_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
