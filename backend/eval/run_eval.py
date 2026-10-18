# backend/eval/run_eval.py
"""
自动化评测脚本
运行所有测试用例，生成评测报告
评测整条Agent链路
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import chromadb

from backend.app.agent.agent import SmartAgent
from backend.app.agent.agent_fc import FunctionCallingAgent
from backend.app.agent.context.compressor import context_compressor
from backend.app.agent.langgraph_agent import LangGraphAgent
from backend.app.agent.tools.calculator_tool import CalculatorTool
from backend.app.agent.tools.file_reader_tool import FileReaderTool
from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool
from backend.app.agent.tools.oa_mock_tool import OaMockTool
from backend.app.core.config import settings
from backend.eval.metrics import evaluator


def _normalize_kb_id(raw: str) -> str:
    """统一为短知识库ID（去掉可能的 kb_ 前缀，因为底层会再拼回 kb_{kb_id}）。"""
    raw = raw.strip()
    return raw[3:] if raw.lower().startswith("kb_") else raw


def _discover_kb_ids() -> list[str]:
    """
    发现评测时知识库检索要绑定的知识库集合（返回短ID，如 ix47sAGw7aJU）。

    默认(default_collection) 在评测环境中为空，真实文档都在 kb_{kb_id} 集合里，
    因此必须显式绑定，否则 knowledge_search 检索不到任何内容。
    multi_kb_retrieve 内部会用 f"kb_{kb_id}" 拼出集合名，所以这里必须传短ID。
    可用环境变量 EVAL_KB_IDS 手动指定（逗号分隔，可带或不带 kb_ 前缀），
    否则自动扫描所有非空的 kb_ 集合。
    """
    env = os.getenv("EVAL_KB_IDS")
    if env and env.strip():
        return [_normalize_kb_id(k) for k in env.split(",") if k.strip()]

    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
        kb_ids = [
            _normalize_kb_id(c.name)
            for c in client.list_collections()
            if c.name.startswith("kb_") and c.count() > 0
        ]
        print(f"[kb发现] 自动绑定知识库集合(短ID): {kb_ids}")
        return kb_ids
    except Exception as e:  # noqa: BLE001
        print(f"[kb发现] 扫描失败，回退到默认集合: {e}")
        return []


def build_eval_agent(llm_provider: str) -> SmartAgent:
    """
    构造一个绑定了真实知识库集合的评测用 Agent。

    - knowledge_search 绑定 _discover_kb_ids() 发现的集合
    - 同时挂载 calculator / file_reader / oa_reimbursement（与默认 agent 一致）
    - 决策范式由环境变量 AGENT_MODE 切换（默认 react=手写 ReAct；fc=原生 Function Calling），
      用于「手写 ReAct vs 原生 FC」对照实验。两范式共用全部工具/guard/评测，变量唯一=决策方式。
    """
    kb_ids = _discover_kb_ids()
    # 不传 kb_descriptions（传空），knowledge_search 会直接跨所有绑定集合检索，
    # 避免每次检索都额外发起一次 LLM 筛选，评测更稳定。
    tools = [
        KnowledgeSearchTool(
            kb_ids=kb_ids,
            retrieval_config={
                "top_k": 8,
                "strategy": "hybrid",
                "use_rerank": True,
                "similarity_threshold": 0.3,  # 前置过滤弱相关块；0.45 配合 HyDE 改写会误杀 borderline 文档致检索为空，回退到已验证的 0.3
                "enable_query_rewrite": True,  # 开启 Query 改写
            },
        ),
        CalculatorTool(),
        FileReaderTool(),
        OaMockTool(),
    ]
    # 三范式对照：AGENT_MODE 切换编排方式（变量唯一=决策/编排方法）
    #   react      → 手写 Prompt-ReAct（默认）
    #   fc         → 原生 Function Calling
    #   langgraph  → LangGraph StateGraph 图编排
    _MODE_MAP = {
        "fc": FunctionCallingAgent,
        "langgraph": LangGraphAgent,
    }
    agent_cls = _MODE_MAP.get(os.getenv("AGENT_MODE", "react").lower(), SmartAgent)
    return agent_cls(pre_loaded_base_tools=tools, llm_provider=llm_provider)


class AutoEvaluator:
    """
    自动化评测器

    功能：
    - 运行所有测试用例
    - 支持上下文压缩策略对比
    - 生成详细评测报告
    """

    def __init__(self, llm_provider: str | None = None):
        self.llm_provider = llm_provider or settings.DEFAULT_LLM_PROVIDER
        # 构造绑定了真实知识库集合的评测 Agent（替代无 KB 绑定的全局 smart_agent 单例）
        self.agent = build_eval_agent(self.llm_provider)

    def load_test_cases(self, path: str = "backend/eval/testset.json") -> dict:
        """
        加载测试用例集

        Args:
            path: 测试用例文件路径

        Returns:
            测试用例数据
        """
        with open(path, encoding="utf-8") as f:
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
            compressed = await context_compressor.compress(history, strategy=compression_strategy)
            history = compressed["compressed_messages"]
        else:
            compressed = None

        # 调用 Agent（使用绑定了真实知识库集合的评测 Agent）
        # 注：SmartAgent / FunctionCallingAgent 的调用记录在实例生命周期内不会自动重置，
        # 单个 Agent 实例服务所有用例时必须手动清空，否则工具调用（含失败记录）会跨用例累积。
        self.agent._called_tools = []
        self.agent._failed_calls = []
        if hasattr(self.agent, "_tool_calls_detail"):
            self.agent._tool_calls_detail = []
        # 单用例超时保护：防止某个慢/死循环的 case 拖垮整个评测。
        # 可通过环境变量 EVAL_CASE_TIMEOUT（秒，默认 600）调整。
        case_timeout = float(os.getenv("EVAL_CASE_TIMEOUT", "600"))
        try:
            result = await asyncio.wait_for(
                self.agent.run(
                    query=query,
                    user_id=1,
                    session_id=f"eval_{case['id']}",
                    history=history,
                    llm_provider=self.llm_provider,
                ),
                timeout=case_timeout,
            )
            called_tools = result.get("called_tools", [])
            tool_calls_detail = result.get("tool_calls_detail", []) or []
            answer = result.get("answer", "")
            # 从 Agent 实际执行记录中提取知识库检索到的文档（用于事实一致性/接地诊断）。
            # tool_results 中 tool_name=="knowledge_search" 的 result 字段即检索结果文本。
            tool_results = result.get("tool_results", []) or []
            # 同时纳入 knowledge_search 与 file_reader 的返回，作为事实接地/幻觉检测的来源。
            retrieved_docs = [
                tr.get("result", "")
                for tr in tool_results
                if tr.get("tool_name") in ("knowledge_search", "file_reader") and tr.get("result")
            ]

            return {
                "case_id": case["id"],
                "query": query,
                "answer": answer,
                "called_tools": called_tools,
                "tool_calls_detail": tool_calls_detail,
                "retrieved_docs": retrieved_docs,
                "success": True,
                "compression": compressed,
            }

        except TimeoutError:
            # 护栏②：看门狗——单用例超时被中断，标记 TIMEOUT 而非卡死整场评测。
            return {
                "case_id": case["id"],
                "query": query,
                "answer": f"TIMEOUT after {case_timeout}s",
                "called_tools": [],
                "tool_calls_detail": [],
                "retrieved_docs": [],
                "success": False,
                "compression": compressed,
            }
        except Exception as e:
            return {
                "case_id": case["id"],
                "query": query,
                "answer": f"ERROR: {str(e)}",
                "called_tools": [],
                "tool_calls_detail": [],
                "retrieved_docs": [],
                "success": False,
                "compression": compressed,
            }

    async def run_all_cases(
        self,
        compression_strategy: str = None,
        filter_type: str = None,
        resume: bool = True,
    ) -> dict:
        """
        运行所有测试用例

        Args:
            compression_strategy: 上下文压缩策略
            filter_type: 按类型过滤（可选）
            resume: 是否断点续跑（默认开；设 EVAL_RESUME=false 或传 False 则从头跑并清旧检查点）

        Returns:
            评测结果
        """
        test_data = self.load_test_cases()
        cases = test_data["test_cases"]

        # 按类型过滤
        if filter_type:
            cases = [c for c in cases if c["type"] == filter_type]
            print(f"过滤后用例数: {len(cases)} (类型: {filter_type})")

        # —— 护栏③：增量落盘 / 断点续跑（Checkpointing）——
        # 每跑完一个用例即写入检查点；中途崩溃后重跑会自动跳过已完成用例、从断点继续，
        # 避免 70 个用例从头再来。检查点按 AGENT_MODE + 压缩策略区分，互不污染。
        mode = os.getenv("AGENT_MODE", "react")
        ckpt_path = Path(
            f"backend/eval/reports/.checkpoint_{mode}_{compression_strategy or 'none'}.json"
        )
        done: dict[str, dict] = {}  # case_id -> {"case_id","result","evaluation"}
        if not resume or os.getenv("EVAL_RESUME", "true").lower() in ("false", "0", "no"):
            if ckpt_path.exists():
                ckpt_path.unlink()
        elif ckpt_path.exists():
            try:
                ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
                for item in ckpt.get("items", []):
                    done[item["case_id"]] = item
                print(f"[checkpoint] 已从断点恢复 {len(done)} 个已完成用例，续跑剩余")
            except Exception as e:  # noqa: BLE001
                print(f"[checkpoint] 读取失败，重新开始: {e}")

        def _save_checkpoint() -> None:
            # 原子写：先写临时文件再 rename，避免写入中途崩溃损坏检查点
            tmp = ckpt_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"items": list(done.values()), "updated_at": datetime.now().isoformat()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(ckpt_path)

        print(f"\n{'=' * 60}")
        print(f"开始评测 - 压缩策略: {compression_strategy}")
        print(f"用例总数: {len(cases)}")
        print(f"{'=' * 60}")

        results = []
        evaluations = []

        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {case['id']} - {case['type']}")
            print(f"  查询: {case['query']}")

            if case["id"] in done:
                item = done[case["id"]]
                result = item["result"]
                eval_result = item["evaluation"]
                print("  (命中 checkpoint，跳过)")
            else:
                result = await self.run_case(case, compression_strategy=compression_strategy)
                eval_result = evaluator.evaluate_single_case(case, result)
                # 增量落盘：跑完即存，崩溃可从这里续
                done[case["id"]] = {
                    "case_id": case["id"],
                    "result": result,
                    "evaluation": eval_result,
                }
                _save_checkpoint()

            results.append(result)
            evaluations.append(eval_result)

            print(f"  分数: {eval_result['score']}/100")
            print(f"  答案匹配: {'✓' if eval_result['answer_match'] else '✗'}")
            print(f"  工具正确: {'✓' if eval_result['tool_accuracy']['correct'] else '✗'}")
            print(f"  幻觉: {'✓' if eval_result['is_hallucination'] else '✗'}")

        # 聚合结果
        aggregated = evaluator.aggregate_results(evaluations)

        # 全部用例已跑完：检查点使命结束，清理避免下次误命中旧缓存（保证重新运行是全新评测）
        if ckpt_path.exists() and len(done) >= len(cases):
            try:
                ckpt_path.unlink()
            except OSError:
                pass

        return {
            "metadata": {
                "compression_strategy": compression_strategy,
                "llm_provider": self.llm_provider,
                "timestamp": datetime.now().isoformat(),
                "total_cases": len(cases),
                "filter_type": filter_type,
                "resumed_from_checkpoint": bool(done) and any(c["id"] in done for c in cases[:1]),
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
            print(f"\n{'=' * 60}")
            print(f"评测压缩策略: {strategy or '无压缩'}")
            print(f"{'=' * 60}")

            result = await self.run_all_cases(
                compression_strategy=strategy,
                filter_type="long_dialog",  # 只测试长对话
            )
            all_results[strategy or "none"] = result

        return all_results

    def generate_full_report(
        self, results: dict, filename: str = None, agent_mode: str = "react"
    ) -> str:
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

        # 保存报告：追加模式，避免两次运行（如 react vs fc）互相覆盖；
        # 若文件已存在则先写分隔标题，区分不同轮次。
        if filename:
            output_path = Path(filename)
            header = ""
            if output_path.exists() and output_path.stat().st_size > 0:
                header = (
                    "\n\n"
                    + "=" * 60
                    + f"\n## 评测轮次 (AGENT_MODE={agent_mode}) @ {datetime.now().isoformat()}\n"
                    + "=" * 60
                    + "\n\n"
                )
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(header + report)
            print(f"\n报告已追加到: {output_path}  (AGENT_MODE={agent_mode})")

        return report


async def main():
    """主函数"""
    print("=" * 60)
    print("Agentic RAG 自动化评测")
    print("=" * 60)

    llm_provider = os.getenv("LLM_PROVIDER", settings.DEFAULT_LLM_PROVIDER)
    agent_mode = os.getenv("AGENT_MODE", "react")
    print(f"使用 LLM provider: {llm_provider}  |  AGENT_MODE: {agent_mode}")

    evaluator = AutoEvaluator(llm_provider=llm_provider)

    # 1. 运行完整评测
    print(f"\n【步骤1】运行完整评测 (AGENT_MODE={agent_mode})")
    full_result = await evaluator.run_all_cases()

    # 2. 生成报告（追加写入，react / fc 两轮不互相覆盖）
    print("\n【步骤2】生成评测报告")
    Path("backend/eval/reports").mkdir(parents=True, exist_ok=True)
    report = evaluator.generate_full_report(
        full_result, "backend/eval/reports/report.md", agent_mode=agent_mode
    )
    print("\n" + "=" * 60)
    print("评测报告摘要")
    print("=" * 60)
    # 打印前50行报告
    for line in report.split("\n")[:50]:
        print(line)

    # 3. 保存详细结果（按 agent_mode + 时间戳命名，避免覆盖；旧 run 不会丢失）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"backend/eval/reports/evaluation_results_{agent_mode}_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")

    # 4. 可选：对比压缩策略（耗时较长，默认跳过）
    # print("\n【步骤3】对比上下文压缩策略（长对话测试）")
    # compression_result = await evaluator.compare_compression_strategies()
    # compression_report = evaluator.generate_full_report(compression_result, "backend/eval/reports/compression_report.md")


if __name__ == "__main__":
    asyncio.run(main())
