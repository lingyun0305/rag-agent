# backend/eval/run_rag_eval.py
"""
RAG 检索质量评测 CLI

用法（在项目根目录下执行）:
  # 1) 导出候选 chunk 供人工标注
  python backend/eval\run_rag_eval.py --mode build-gold --dataset backend/eval\rag_testset.example.json

  # 2) 计算检索质量指标（需先填好 expected_chunk_ids / expected_snippets）
  python backend/eval\run_rag_eval.py --mode eval --dataset backend/eval\rag_testset.example.json

  # 独立进程跑 hybrid/bm25 前，从 Chroma 重建内存 BM25 索引
  python backend/eval\run_rag_eval.py --mode eval --dataset backend/eval\rag_testset.example.json --rebuild-bm25

  # 可选：开启 Ragas LLM-judge 指标（需 ragas 包 + 联网 LLM）
  python backend/eval\run_rag_eval.py --mode eval --dataset backend/eval\rag_testset.example.json --ragas

  # 零标注诊断（不填 expected_* 也能跑，仅看召回条数 / 分数 / 耗时）
  python backend/eval\run_rag_eval.py --mode eval --dataset backend/eval\rag_testset.example.json

说明: 直接复用 retriever_service.multi_kb_retrieve，需在能访问 ./chroma_db
      与同款 embedding 模型的后端环境运行。
"""

import argparse
import json
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，使 from backend.xxx import 可解析
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # eval → backend → 项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval.rag_eval import RagRetrievalEvaluator  # noqa: E402


def _print_gold(gold: list) -> None:
    print("\n" + "=" * 70)
    print("BUILD-GOLD: 以下为每个 query 实际召回的 top-k 候选 chunk")
    print("请把你认为“应该被召回”的 chunk_id 填回数据集的 expected_chunk_ids")
    print("（或把该 chunk 的关键句写入 expected_snippets 做兜底匹配）")
    print("=" * 70)
    for item in gold:
        print(f"\n[{item['index']}] query: {item['query']}")
        print(f"    strategy: {item['strategy']}  (候选数: {len(item['candidates'])})")
        for c in item["candidates"]:
            snippet = (c["content"] or "").replace("\n", " ")
            print(
                f"    #{c['rank']:>2} | chunk_id={c['chunk_id']} | "
                f"kb={c['kb_id']} | score={c['score']}"
            )
            print(f"         {snippet[:120]}")


def _print_report(result: dict) -> None:
    agg = result.get("aggregate", {})
    print("\n" + "=" * 70)
    print(f"RAG 检索评测报告 — {result.get('dataset_name')}")
    print(f"k={result['k']}  总用例={result['total_cases']}  " f"已标注={result['labeled_cases']}")
    th = result.get("threshold")
    print(f"similarity_threshold={th if th is not None else 0}  (只卡向量召回)")
    print("=" * 70)

    if agg.get("mode") == "diagnostic (无标注)":
        print("\n[诊断模式] 未提供标注，仅输出检索诊断指标:")
        print(f"  平均召回条数 : {agg.get('avg_retrieved_count')}")
        print(f"  平均耗时(ms) : {agg.get('avg_latency_ms')}")
        print("  提示: 在数据集填入 expected_chunk_ids / expected_snippets 后可算准确率/召回率。")
    else:
        print("\n[聚合指标] (已标注用例均值)")
        for key in ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr", "ndcg@k"]:
            print(f"  avg_{key:<12}: {agg.get('avg_' + key)}")
        print(f"  平均召回条数   : {agg.get('avg_retrieved_count')}")
        print(f"  平均向量top分  : {agg.get('avg_vector_top_score')}  (阈值过滤依据此分)")
        print(f"  平均耗时(ms)   : {agg.get('avg_latency_ms')}")

    print("\n[逐条结果]")
    for c in result.get("per_case", []):
        if c.get("labeled") and c.get("metrics"):
            m = c["metrics"]
            print(
                f"  [{c['index']}] {c['query']}\n"
                f"       P@{result['k']}={m['precision@k']}  "
                f"R@{result['k']}={m['recall@k']}  "
                f"F1={m['f1@k']}  MRR={m['mrr']}  NDCG={m['ndcg@k']}  "
                f"向量top分={c.get('vector_top_score')}  "
                f"(命中 {m['relevant_in_topk']}/{m['retrieved_count']})"
            )
        else:
            print(
                f"  [{c['index']}] {c['query']}  [诊断] "
                f"召回 {c['retrieved_count']} 条, top_score={c['top_score']}, "
                f"向量top分={c.get('vector_top_score')}, "
                f"{c['latency_ms']}ms"
            )

    if "ragas" in result:
        rg = result["ragas"]
        print("\n[Ragas LLM-judge]")
        if rg.get("available"):
            for name, val in rg.get("scores", {}).items():
                print(f"  {name}: {val}")
        else:
            print(f"  未启用/不可用: {rg.get('error')}")


def _to_markdown(result: dict) -> str:
    agg = result.get("aggregate", {})
    lines = [
        f"# RAG 检索评测报告 — {result.get('dataset_name')}",
        "",
        f"- k = {result['k']}",
        f"- 总用例 = {result['total_cases']}，已标注 = {result['labeled_cases']}",
        "",
        "## 聚合指标",
        "",
    ]
    if agg.get("mode") == "diagnostic (无标注)":
        lines += [
            "| 指标 | 值 |",
            "|------|------|",
            f"| 平均召回条数 | {agg.get('avg_retrieved_count')} |",
            f"| 平均耗时(ms) | {agg.get('avg_latency_ms')} |",
            "",
            "> 未提供标注，仅输出诊断指标。在数据集填入 expected_chunk_ids / expected_snippets 后可算准确率/召回率。",
            "",
        ]
    else:
        lines += ["| 指标 | 值 |", "|------|------|"]
        for key in ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr", "ndcg@k"]:
            lines.append(f"| avg_{key} | {agg.get('avg_' + key)} |")
        lines.append(f"| 平均召回条数 | {agg.get('avg_retrieved_count')} |")
        lines.append(f"| 平均耗时(ms) | {agg.get('avg_latency_ms')} |")
        lines.append("")

    lines += [
        "## 逐条结果",
        "",
        "| # | query | 策略 | 召回数 | top_score | 向量top分 | 耗时(ms) | P@k | R@k | F1 | MRR | NDCG |",
        "|---|--------|------|-------|-----------|-----------|----------|-----|-----|----|-----|------|",
    ]
    for c in result.get("per_case", []):
        m = c.get("metrics")
        if m:
            lines.append(
                f"| {c['index']} | {c['query']} | {c['strategy']} | {c['retrieved_count']} | "
                f"{c['top_score']} | {c.get('vector_top_score')} | {c['latency_ms']} | {m['precision@k']} | {m['recall@k']} | "
                f"{m['f1@k']} | {m['mrr']} | {m['ndcg@k']} |"
            )
        else:
            lines.append(
                f"| {c['index']} | {c['query']} | {c['strategy']} | {c['retrieved_count']} | "
                f"{c['top_score']} | {c.get('vector_top_score')} | {c['latency_ms']} | - | - | - | - | - |"
            )

    if "ragas" in result:
        rg = result["ragas"]
        lines += ["", "## Ragas LLM-judge", ""]
        if rg.get("available"):
            for name, val in rg.get("scores", {}).items():
                lines.append(f"- {name}: {val}")
        else:
            lines.append(f"- 未启用/不可用: {rg.get('error')}")

    return "\n".join(lines) + "\n"


def _parse_thresholds(raw: str | None) -> list[float] | None:
    """解析 --threshold 参数：None / [float,...]。支持逗号分隔多值。"""
    if not raw:
        return None
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out or None


def _print_threshold_compare(rows: list[tuple]) -> None:
    """多阈值对比表：每行一个 threshold 的聚合指标。"""
    print("\n" + "=" * 78)
    print("阈值对比 (similarity_threshold: 只卡向量召回)")
    print("=" * 78)
    header = (
        f"{'threshold':>9} | {'avg_P@k':>8} | {'avg_R@k':>8} | {'avg_F1':>7} | "
        f"{'Hit':>5} | {'MRR':>6} | {'NDCG':>6} | {'avg召回':>7} | {'vecTop':>7} | {'avg耗时':>9}"
    )
    print(header)
    print("-" * len(header))
    for th, res in rows:  # type: (float, dict)
        agg = res.get("aggregate", {})
        p = agg.get("avg_precision@k", "-")
        r = agg.get("avg_recall@k", "-")
        f1 = agg.get("avg_f1@k", "-")
        hit = agg.get("avg_hit_rate@k", "-")
        mrr = agg.get("avg_mrr", "-")
        ndcg = agg.get("avg_ndcg@k", "-")
        rc = agg.get("avg_retrieved_count", "-")
        vec_top = agg.get("avg_vector_top_score", "-")
        lat = agg.get("avg_latency_ms", "-")

        def _fmt(v) -> str:
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        warn = ""
        if isinstance(rc, (int, float)) and rc <= 0:
            warn = "  ⚠ 召回塌缩"
        print(
            f"{th:>9} | {_fmt(p):>8} | {_fmt(r):>8} | {_fmt(f1):>7} | {_fmt(hit):>5} | "
            f"{_fmt(mrr):>6} | {_fmt(ndcg):>6} | {_fmt(rc):>7} | {_fmt(vec_top):>7} | {_fmt(lat):>9}{warn}"
        )
    print()
    print("读法: 观察 avg_P@k 是否上升、avg_R@k 是否还能撑住、avg召回条数 是否塌缩。")
    print("     诊断模式(无标注)下指标列为 '-'，仅看 avg召回条数 / avg耗时。")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测 CLI")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "rag_testset.example.json"),
        help="评测数据集 JSON 路径",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "build-gold"],
        default="eval",
        help="eval=计算指标；build-gold=导出候选 chunk 供标注",
    )
    parser.add_argument("--top-k", type=int, default=None, help="覆盖数据集默认 top_k")
    parser.add_argument("--strategy", default=None, help="覆盖数据集默认检索策略")
    parser.add_argument(
        "--threshold",
        default=None,
        help="相似度阈值(只卡向量召回)。0=不过滤。支持逗号分隔多值做对比，"
        "例如 --threshold 0,0.02,0.05,0.07",
    )
    parser.add_argument("--rebuild-bm25", action="store_true", help="从 Chroma 重建内存 BM25 索引")
    parser.add_argument(
        "--ragas", action="store_true", help="开启 Ragas LLM-judge 指标（需联网 LLM）"
    )
    parser.add_argument("--output", default=None, help="JSON 报告输出路径")
    parser.add_argument("--gold-top-k", type=int, default=20, help="build-gold 导出候选条数")
    args = parser.parse_args()

    evaluator = RagRetrievalEvaluator()
    dataset = evaluator.load_dataset(args.dataset)
    kb_ids = dataset.get("kb_ids") or []

    if args.rebuild_bm25 and kb_ids:
        print("[*] 重建 BM25 索引（从 Chroma 集合）...")
        counts = evaluator.rebuild_bm25(kb_ids)
        print(f"    BM25 文档数: {counts}")

    if args.mode == "build-gold":
        gold = evaluator.dump_gold(dataset, top_k=args.gold_top_k, override_strategy=args.strategy)
        _print_gold(gold)
        return

    thresholds = _parse_thresholds(args.threshold)

    # 多阈值对比：循环跑多次，输出聚合对比表（跳过 ragas 以免重复 LLM 调用）
    if thresholds and len(thresholds) > 1:
        rows: list[tuple] = []
        for th in thresholds:
            res = evaluator.evaluate(
                dataset,
                k=args.top_k,
                override_strategy=args.strategy,
                use_ragas=False,
                override_threshold=th,
            )
            rows.append((th, res))
        _print_threshold_compare(rows)
        reports_dir = Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        compare_out = args.output or str(
            reports_dir / "rag_threshold_compare.json"
        )
        with open(compare_out, "w", encoding="utf-8") as f:
            json.dump(
                [{"threshold": th, "result": res} for th, res in rows],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n多阈值对比报告已保存: {compare_out}")
        return

    result = evaluator.evaluate(
        dataset,
        k=args.top_k,
        override_strategy=args.strategy,
        use_ragas=args.ragas,
        override_threshold=(thresholds[0] if thresholds else None),
    )
    _print_report(result)

    out = args.output or str(
        Path(__file__).resolve().parent / "reports" / "rag_report.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md = str(Path(out).with_suffix(".md"))
    with open(md, "w", encoding="utf-8") as f:
        f.write(_to_markdown(result))
    print(f"\n报告已保存: {out}")
    print(f"Markdown 报告: {md}")


if __name__ == "__main__":
    main()
