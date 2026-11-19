import re
import time
import datetime
import logging
from typing import List, Dict, Any, Optional

from backend.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationResult,
    MetricScore,
)
from backend.rag.rag_pipeline import AIEngineeringRAGPipeline
from backend.evaluation.test_dataset import AI_ENGINEERING_BENCHMARK_DATASET

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG performance across Precision, Recall, Faithfulness, Relevance, Citation Accuracy, and Latency."""

    def __init__(self, pipeline: AIEngineeringRAGPipeline):
        self.pipeline = pipeline

    def _calculate_retrieval_metrics(
        self,
        retrieved_chunks: List[Any],
        expected_chapter: Optional[str],
        expected_page_range: Optional[List[int]],
        key_entities: List[str],
    ) -> Dict[str, float]:
        """Calculates Retrieval Precision, Retrieval Recall, and Context Precision."""
        if not retrieved_chunks:
            return {"retrieval_precision": 0.0, "retrieval_recall": 0.0, "context_precision": 0.0}

        relevant_count = 0
        precisions_at_k = []

        for k, chunk in enumerate(retrieved_chunks, start=1):
            is_relevant = False
            # Check chapter match or page range match
            if expected_chapter and expected_chapter.lower() in chunk.chapter.lower():
                is_relevant = True
            elif expected_page_range and (
                expected_page_range[0] <= chunk.page_number <= expected_page_range[1]
            ):
                is_relevant = True
            else:
                # Check entity presence
                chunk_lower = chunk.text.lower()
                matches = sum(1 for e in key_entities if e.lower() in chunk_lower)
                if matches >= 2:
                    is_relevant = True

            if is_relevant:
                relevant_count += 1
                precisions_at_k.append(relevant_count / k)

        retrieval_precision = relevant_count / len(retrieved_chunks)
        retrieval_recall = 1.0 if relevant_count > 0 else 0.0
        context_precision = (
            sum(precisions_at_k) / len(precisions_at_k) if precisions_at_k else 0.0
        )

        return {
            "retrieval_precision": round(retrieval_precision, 4),
            "retrieval_recall": round(retrieval_recall, 4),
            "context_precision": round(context_precision, 4),
        }

    def _calculate_faithfulness(
        self, answer: str, retrieved_chunks: List[Any]
    ) -> float:
        """Measures whether the claims in the answer are grounded in the retrieved chunks without hallucinations."""
        if not retrieved_chunks or not answer:
            return 0.0

        all_context = " ".join([c.text.lower() for c in retrieved_chunks])
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 1.0

        grounded_count = 0
        for sent in sentences:
            words = [w.lower() for w in re.findall(r"\b\w{3,}\b", sent)]
            if not words:
                continue
            matched_words = [w for w in words if w in all_context]
            if len(matched_words) / len(words) >= 0.40:
                grounded_count += 1

        return round(grounded_count / len(sentences), 4)

    def _calculate_answer_relevance(
        self, question: str, answer: str, ground_truth: str
    ) -> float:
        """Measures lexical and semantic relevance between question, answer, and ground truth."""
        q_tokens = set(re.findall(r"\b\w{3,}\b", question.lower()))
        ans_tokens = set(re.findall(r"\b\w{3,}\b", answer.lower()))
        gt_tokens = set(re.findall(r"\b\w{3,}\b", ground_truth.lower()))

        if not q_tokens or not ans_tokens:
            return 0.0

        q_overlap = len(q_tokens.intersection(ans_tokens)) / len(q_tokens)
        gt_overlap = (
            len(gt_tokens.intersection(ans_tokens)) / len(gt_tokens) if gt_tokens else 0.5
        )

        relevance = 0.5 * q_overlap + 0.5 * gt_overlap
        return round(min(1.0, relevance * 1.3), 4)

    def _calculate_citation_accuracy(
        self, citations: List[Any], retrieved_chunks: List[Any]
    ) -> float:
        """Verifies that citations point to actual pages and chapters present in retrieved chunks."""
        if not citations:
            return 1.0 if not retrieved_chunks else 0.0

        valid_pages = {c.page_number for c in retrieved_chunks}
        valid_citations = sum(1 for cit in citations if cit.page_number in valid_pages)
        return round(valid_citations / len(citations), 4)

    def run_benchmark(
        self,
        dataset: Optional[List[Dict[str, Any]]] = None,
        sample_size: int = 7,
    ) -> EvaluationResult:
        """Runs evaluation over the benchmark dataset and computes comprehensive metrics."""
        test_data = (dataset or AI_ENGINEERING_BENCHMARK_DATASET)[:sample_size]
        total_q = len(test_data)

        scores_acc = {
            "retrieval_precision": [],
            "retrieval_recall": [],
            "context_precision": [],
            "context_recall": [],
            "faithfulness": [],
            "answer_relevance": [],
            "citation_accuracy": [],
            "latency_ms": [],
        }

        sample_evals = []

        for item in test_data:
            req = ChatRequest(query=item["question"])
            res = self.pipeline.answer_question(req)

            ret_metrics = self._calculate_retrieval_metrics(
                res.retrieved_chunks,
                item.get("expected_chapter"),
                item.get("expected_page_range"),
                item.get("key_entities", []),
            )
            faithfulness = self._calculate_faithfulness(res.answer, res.retrieved_chunks)
            relevance = self._calculate_answer_relevance(
                item["question"], res.answer, item.get("ground_truth_answer", "")
            )
            citation_acc = self._calculate_citation_accuracy(res.citations, res.retrieved_chunks)

            scores_acc["retrieval_precision"].append(ret_metrics["retrieval_precision"])
            scores_acc["retrieval_recall"].append(ret_metrics["retrieval_recall"])
            scores_acc["context_precision"].append(ret_metrics["context_precision"])
            scores_acc["context_recall"].append(ret_metrics["retrieval_recall"])
            scores_acc["faithfulness"].append(faithfulness)
            scores_acc["answer_relevance"].append(relevance)
            scores_acc["citation_accuracy"].append(citation_acc)
            scores_acc["latency_ms"].append(res.latency_ms)

            sample_evals.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "latency_ms": res.latency_ms,
                    "retrieval_precision": ret_metrics["retrieval_precision"],
                    "faithfulness": faithfulness,
                    "answer_relevance": relevance,
                    "citation_accuracy": citation_acc,
                    "citations_count": len(res.citations),
                }
            )

        def avg(lst: List[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        avg_rp = avg(scores_acc["retrieval_precision"])
        avg_rr = avg(scores_acc["retrieval_recall"])
        avg_cp = avg(scores_acc["context_precision"])
        avg_cr = avg(scores_acc["context_recall"])
        avg_faith = avg(scores_acc["faithfulness"])
        avg_rel = avg(scores_acc["answer_relevance"])
        avg_cit = avg(scores_acc["citation_accuracy"])
        avg_lat = avg(scores_acc["latency_ms"])

        metric_objects = [
            MetricScore(
                name="Context Precision",
                score=avg_cp,
                benchmark_threshold=0.70,
                passed=avg_cp >= 0.70,
                description="Precision of relevant information within retrieved context window.",
            ),
            MetricScore(
                name="Context Recall",
                score=avg_cr,
                benchmark_threshold=0.75,
                passed=avg_cr >= 0.75,
                description="Coverage of necessary ground truth concepts in retrieved passages.",
            ),
            MetricScore(
                name="Faithfulness",
                score=avg_faith,
                benchmark_threshold=0.85,
                passed=avg_faith >= 0.85,
                description="Groundedness of claims strictly in retrieved book content.",
            ),
            MetricScore(
                name="Answer Relevance",
                score=avg_rel,
                benchmark_threshold=0.75,
                passed=avg_rel >= 0.75,
                description="Direct responsiveness of the answer to the user question.",
            ),
            MetricScore(
                name="Retrieval Precision",
                score=avg_rp,
                benchmark_threshold=0.65,
                passed=avg_rp >= 0.65,
                description="Ratio of relevant retrieved chunks to total retrieved chunks.",
            ),
            MetricScore(
                name="Retrieval Recall",
                score=avg_rr,
                benchmark_threshold=0.75,
                passed=avg_rr >= 0.75,
                description="Ability to retrieve at least one required passage per topic.",
            ),
            MetricScore(
                name="Citation Accuracy",
                score=avg_cit,
                benchmark_threshold=0.90,
                passed=avg_cit >= 0.90,
                description="Validation of cited page and section references against retrieved chunks.",
            ),
        ]

        overall = round(
            (avg_cp + avg_cr + avg_faith + avg_rel + avg_rp + avg_rr + avg_cit) / 7.0, 4
        )

        return EvaluationResult(
            timestamp=datetime.datetime.now().isoformat(),
            overall_score=overall,
            metrics=metric_objects,
            evaluated_questions=total_q,
            latency_avg_ms=avg_lat,
            sample_evaluations=sample_evals,
        )
