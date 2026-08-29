# backend/eval/rag_eval.py
"""
RAG 检索质量评测核心库

直接复用 retriever_service.multi_kb_retrieve 检索链路，计算确定性 IR 指标
(准确率@k / 召回率@k / F1 / 命中率 / MRR / NDCG)，并可选接入 Ragas
LLM-judge 指标 (context_precision / context_recall / context_relevancy)。

设计要点:
- 现有系统无 ground truth，故要求评测集提供"期望召回 chunk"
  (expected_chunk_ids 或 expected_snippets)。build-gold 模式可导出候选
  chunk_id 辅助标注。
- BM25 为纯内存索引，独立评测进程需 --rebuild-bm25 从 Chroma 集合重建。
- chunk 有三套 ID (metadata.chunk_id=uuid4 / MySQL DocumentChunk.id /
  vector_id)，以 metadata.chunk_id 为主键匹配，expected_snippets 做片段指纹兜底。
- 确定性指标零外部依赖，可离线运行；Ragas 需联网 + LLM，缺失时自动跳过。
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from langchain_core.documents import Document

# 注意: retriever_service 的导入故意延迟到实例化时 (__init__ 内) 进行。
# 因为该导入链会触发 models_factory，离线环境会尝试从 HuggingFace 拉取
# reranker 模型而超时。延迟导入可让本模块在离线环境也能干净 import，
# 纯指标函数可独立单测；真实评测在能联网的后端环境才加载检索服务。


__all__ = ["RagRetrievalEvaluator"]


class RagRetrievalEvaluator:
    """RAG 检索质量评测器。"""

    def __init__(self, retriever=None):
        if retriever is not None:
            self.retriever = retriever
            return
        # 延迟导入：避免在离线/无后端环境 import 本模块时触发重模型下载
        try:
            from backend.app.rag.retriever_service import retriever_service

            self.retriever = retriever_service
        except Exception as exc:
            raise RuntimeError(
                "检索服务未能导入，请确认从后端项目根目录运行且依赖已安装。" f"原始错误: {exc}"
            )

    # ===================== 数据集 =====================

    def load_dataset(self, path: str) -> dict[str, Any]:
        """加载并规范化评测数据集。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "cases" not in data:
            raise ValueError("数据集缺少 'cases' 字段")
        data.setdefault("kb_ids", [])
        data.setdefault("default_strategy", "hybrid")
        data.setdefault("default_top_k", 10)
        return data

    # ===================== 检索 =====================

    def _retrieve(
        self,
        case: dict[str, Any],
        dataset: dict[str, Any],
        override_strategy: str | None = None,
        override_top_k: int | None = None,
        override_threshold: float | None = None,
    ) -> tuple[list[Document], float]:
        """调用真实检索链路，返回 (文档列表, 耗时ms)。

        override_threshold: 相似度/相关度分数阈值（只卡向量召回）。
            None 表示沿用 retriever 实例默认（=0 不过滤）。
        """
        query = case["query"]
        kb_ids = case.get("kb_ids") or dataset.get("kb_ids") or []
        if not kb_ids:
            raise ValueError(f"case 缺少 kb_ids 且数据集未配置 kb_ids: {query!r}")
        strategy = override_strategy or case.get("strategy") or dataset.get("default_strategy")
        top_k = override_top_k or case.get("top_k") or dataset.get("default_top_k")
        use_rerank = case.get("use_rerank")

        t0 = time.time()
        docs = self.retriever.multi_kb_retrieve(
            query=query,
            kb_ids=kb_ids,
            strategy=strategy,
            use_rerank=use_rerank,
            top_k=top_k,
            score_threshold=override_threshold,
        )
        latency = (time.time() - t0) * 1000
        return docs, latency

    # ===================== 相关性判定 =====================

    @staticmethod
    def _resolve_relevant(
        docs: list[Document],
        expected_chunk_ids: list[str],
        expected_snippets: list[str],
    ) -> list[bool]:
        """
        逐条判定检索结果是否相关。
        命中规则: doc.metadata.chunk_id ∈ expected_chunk_ids，或其内容包含任一 snippet。
        """
        exp_ids = set(expected_chunk_ids or [])
        snippets = [s.strip() for s in (expected_snippets or []) if s and s.strip()]
        is_relevant: list[bool] = []
        for doc in docs:
            md = doc.metadata or {}
            cid = md.get("chunk_id")
            matched_id = (cid in exp_ids) if cid is not None else False
            matched_snip = False
            if snippets:
                content = doc.page_content or ""
                matched_snip = any(snip in content for snip in snippets)
            is_relevant.append(bool(matched_id or matched_snip))
        return is_relevant

    # ===================== 指标计算 =====================

    @staticmethod
    def _compute_ir_metrics(
        is_relevant: list[bool],
        expected_count: int,
        k: int,
    ) -> dict[str, float]:
        """计算单条 query 的确定性 IR 指标（二元相关性）。"""
        topk = is_relevant[:k]
        rel_in_topk = sum(1 for x in topk if x)

        precision = rel_in_topk / k if k > 0 else 0.0
        recall = rel_in_topk / expected_count if expected_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        hit_rate = 1.0 if rel_in_topk > 0 else 0.0

        # MRR: 首个相关结果的排名倒数
        mrr = 0.0
        for i, rel in enumerate(is_relevant, start=1):
            if rel:
                mrr = 1.0 / i
                break

        # NDCG@k: 二元相关性下的 DCG / IDCG
        def _dcg(rels: list[bool]) -> float:
            return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

        ideal_len = min(expected_count, k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        ndcg = _dcg(topk) / idcg if idcg > 0 else 0.0

        return {
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4),
            "hit_rate@k": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "ndcg@k": round(ndcg, 4),
            "retrieved_count": len(is_relevant),
            "relevant_in_topk": rel_in_topk,
        }

    @staticmethod
    def _top_score(docs: list[Document]) -> float | None:
        """取检索结果中最高的相关性分数（relevance_score / rrf_score / score）。"""
        best: float | None = None
        for d in docs:
            md = d.metadata or {}
            s = md.get("relevance_score")
            if s is None:
                s = md.get("rrf_score")
            if s is None:
                s = md.get("score")
            if s is not None:
                try:
                    s = float(s)
                    if best is None or s > best:
                        best = s
                except (TypeError, ValueError):
                    pass
        return best

    @staticmethod
    def _vector_top_score(docs: list[Document]) -> float | None:
        """取向量来源(retrieval_source=='vector')的最高 score（余弦相似度）。

        这是 similarity_threshold 实际过滤所依据的分数，与 rerank 后的
        relevance_score 量纲可能不同，单独展示便于用户设定阈值。
        """
        best: float | None = None
        for d in docs:
            md = d.metadata or {}
            if md.get("retrieval_source") != "vector":
                continue
            s = md.get("score")
            if s is None:
                continue
            try:
                s = float(s)
                if best is None or s > best:
                    best = s
            except (TypeError, ValueError):
                pass
        return best

    @staticmethod
    def _aggregate(per_case: list[dict[str, Any]], k: int) -> dict[str, Any]:
        """聚合所有 case 的指标。"""
        if not per_case:
            return {"mode": "empty", "labeled_count": 0}

        labeled = [c for c in per_case if c.get("labeled")]
        avg_retrieved = sum(c["retrieved_count"] for c in per_case) / len(per_case)
        avg_latency = sum(c["latency_ms"] for c in per_case) / len(per_case)
        avg_vec_top = sum(c.get("vector_top_score") or 0.0 for c in per_case) / len(per_case)

        if not labeled:
            return {
                "mode": "diagnostic (无标注)",
                "avg_retrieved_count": round(avg_retrieved, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "hint": "未提供 expected_chunk_ids / expected_snippets，仅输出诊断指标。",
            }

        keys = ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr", "ndcg@k"]
        agg: dict[str, Any] = {"mode": "labeled", "labeled_count": len(labeled)}
        for key in keys:
            vals = [c["metrics"][key] for c in labeled]
            agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
        agg["avg_retrieved_count"] = round(avg_retrieved, 2)
        agg["avg_latency_ms"] = round(avg_latency, 2)
        agg["avg_vector_top_score"] = round(avg_vec_top, 4)
        return agg

    # ===================== 主评测流程 =====================

    def evaluate(
        self,
        dataset: dict[str, Any],
        k: int | None = None,
        override_strategy: str | None = None,
        use_ragas: bool = False,
        override_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        对数据集逐条检索并评测。

        Returns:
            {
              dataset_name, k, total_cases, labeled_cases,
              aggregate, per_case[], (ragas?)
            }
        """
        cases = dataset["cases"]
        k = k or dataset.get("default_top_k", 10)
        per_case: list[dict[str, Any]] = []
        labeled_cases = 0
        ragas_rows: list[dict[str, Any]] = []

        for idx, case in enumerate(cases, 1):
            docs, latency = self._retrieve(case, dataset, override_strategy, k, override_threshold)
            exp_ids = case.get("expected_chunk_ids") or []
            exp_snips = case.get("expected_snippets") or []
            has_label = bool(exp_ids or exp_snips)
            top_score = self._top_score(docs)

            case_result: dict[str, Any] = {
                "index": idx,
                "query": case["query"],
                "strategy": override_strategy
                or case.get("strategy")
                or dataset.get("default_strategy"),
                "top_k": k,
                "retrieved_count": len(docs),
                "top_score": top_score,
                "vector_top_score": self._vector_top_score(docs),
                "latency_ms": round(latency, 2),
                "labeled": has_label,
            }

            if has_label:
                labeled_cases += 1
                is_rel = self._resolve_relevant(docs, exp_ids, exp_snips)
                # expected_count 以 chunk_id 精确标注为准；snippets 仅作兜底匹配手段，
                # 不应 inflate 期望正确数（否则 3 个 snippet 就把 expected_count 撑到 3）。
                metrics = self._compute_ir_metrics(
                    is_rel, expected_count=len(exp_ids) if exp_ids else sum(is_rel), k=k
                )
                case_result["metrics"] = metrics
                case_result["relevant_chunk_ids"] = [
                    (d.metadata or {}).get("chunk_id") for d, r in zip(docs, is_rel) if r
                ]
                if use_ragas and case.get("reference"):
                    ragas_rows.append(
                        {
                            "question": case["query"],
                            "contexts": [d.page_content for d in docs],
                            "reference": case["reference"],
                        }
                    )
            else:
                case_result["metrics"] = None
                case_result["note"] = "无标注：仅输出诊断指标（召回条数/分数/耗时）"

            per_case.append(case_result)

        agg = self._aggregate(per_case, k)
        result: dict[str, Any] = {
            "dataset_name": dataset.get("name", "unnamed"),
            "k": k,
            "threshold": override_threshold,
            "total_cases": len(cases),
            "labeled_cases": labeled_cases,
            "aggregate": agg,
            "per_case": per_case,
        }
        if use_ragas and ragas_rows:
            result["ragas"] = self._run_ragas(ragas_rows)
        return result

    # ===================== build-gold =====================

    def dump_gold(
        self,
        dataset: dict[str, Any],
        top_k: int = 20,
        override_strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        build-gold 模式：跑检索并打印每个 query 的 top-k 候选 chunk，
        供人工把正确 chunk_id 回填到 expected_chunk_ids。
        """
        out: list[dict[str, Any]] = []
        for idx, case in enumerate(dataset["cases"], 1):
            docs, _ = self._retrieve(case, dataset, override_strategy, top_k)
            candidates = []
            for rank, doc in enumerate(docs, 1):
                md = doc.metadata or {}
                score = md.get("relevance_score") or md.get("rrf_score") or md.get("score")
                candidates.append(
                    {
                        "rank": rank,
                        "chunk_id": md.get("chunk_id"),
                        "kb_id": md.get("kb_id"),
                        "score": score,
                        "content": (doc.page_content or "")[:200],
                    }
                )
            out.append(
                {
                    "index": idx,
                    "query": case["query"],
                    "strategy": override_strategy
                    or case.get("strategy")
                    or dataset.get("default_strategy"),
                    "candidates": candidates,
                }
            )
        return out

    # ===================== BM25 重建 =====================

    def rebuild_bm25(self, kb_ids: list[str]) -> dict[str, int]:
        """
        独立评测进程 BM25 索引为空，hybrid/bm25 会退化成纯向量。
        本方法从 Chroma 集合 (kb_{kb_id}) 读出全部文档（保留原始 metadata，
        含 chunk_id），重建内存 BM25 索引，确保 hybrid 评测与线上一致。
        """
        import chromadb

        from backend.app.rag.bm25_service import bm25_service
        from backend.app.utils.vector_store import get_vector_store

        counts: dict[str, int] = {}
        for kb_id in kb_ids:
            collection_name = f"kb_{kb_id}"
            vs = get_vector_store(collection_name=collection_name)
            client = chromadb.PersistentClient(path=vs.persist_directory)
            try:
                coll = client.get_collection(collection_name)
            except Exception:
                counts[kb_id] = 0
                continue
            try:
                data = coll.get(include=["documents", "metadatas"])
            except Exception:
                counts[kb_id] = 0
                continue

            docs: list[Document] = []
            for text, md in zip(data.get("documents") or [], data.get("metadatas") or []):
                if text:
                    docs.append(Document(page_content=text, metadata=md or {}))

            bm25_service.add_documents(docs, collection_name=collection_name)
            counts[kb_id] = len(docs)
        return counts

    # ===================== Ragas（可选） =====================

    def _run_ragas(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """
        可选：Ragas LLM-judge 指标。依赖 ragas + datasets + 联网 LLM。
        任何失败都返回 available=False 与原因，不阻断主流程。
        注意: ragas 默认使用 OpenAI，如需用本项目 LLM，请自行配置 ragas.llm。
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                context_relevancy,
            )
        except ImportError as e:
            return {"available": False, "error": f"ragas 未安装，跳过: {e}"}

        try:
            dataset = Dataset.from_dict(
                {
                    "question": [r["question"] for r in rows],
                    "contexts": [r["contexts"] for r in rows],
                    "ground_truth": [r["reference"] for r in rows],
                }
            )
            metrics = [context_relevancy]
            if all(r["reference"] for r in rows):
                metrics += [context_precision, context_recall]
            result = evaluate(dataset, metrics=metrics)
            return {"available": True, "scores": dict(result)}
        except Exception as e:
            return {
                "available": False,
                "error": f"Ragas 运行失败（可能缺 LLM / 未联网 / 未配置 ragas.llm）: {e}",
            }
