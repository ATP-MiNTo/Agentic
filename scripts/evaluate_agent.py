"""Simple local evaluation harness for the Medical RAG agent.

This runs a small curated set of questions and prints execution details so the
agentic loop can be checked without any paid services or external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List

from src.agent.rag_agent import MedicalRAGAgent


@dataclass
class EvalCase:
    query: str
    expected_topics: List[str]


CASES = [
    EvalCase(
        query="What are the early symptoms of diabetes?",
        expected_topics=["diabetes", "symptoms"],
    ),
    EvalCase(
        query="How is migraine different from a regular headache and when should I see a doctor?",
        expected_topics=["migraine", "doctor"],
    ),
    EvalCase(
        query="What are cancer risk factors and common treatment options?",
        expected_topics=["cancer", "treatment"],
    ),
]


def run_evaluation() -> int:
    agent = MedicalRAGAgent()
    scores = []

    print("=== Medical RAG Agent Evaluation ===")
    print(f"Cases: {len(CASES)}\n")

    for index, case in enumerate(CASES, 1):
        result = agent.chat(case.query, top_k=5, return_details=True)
        score = float(result.get("final_confidence", 0.0))
        scores.append(score)

        retrieved_ids = [doc["id"] for doc in result.get("retrieved_documents", [])]
        subqueries = result.get("subqueries", [])
        strategy = result.get("retrieval_strategy", "unknown")
        answer_preview = result["response"].splitlines()[0][:140]

        print(f"[{index}] Query: {case.query}")
        print(f"    Success: {result.get('success', False)}")
        print(f"    Confidence: {score:.2f}")
        print(f"    Iterations: {result.get('iterations', 0)}")
        print(f"    Retrieval strategy: {strategy}")
        print(f"    Subqueries: {subqueries or 'none'}")
        print(f"    Retrieved docs: {retrieved_ids[:5] or 'none'}")
        print(f"    Answer preview: {answer_preview}")
        print()

    print("=== Summary ===")
    print(f"Average critic confidence: {mean(scores):.2f}")
    print("Use this as a smoke test rather than a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
