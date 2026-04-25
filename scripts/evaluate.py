"""
Run: python scripts/evaluate.py
Evaluates the RAG pipeline against the ground-truth Q&A dataset using RAGAS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.eval_runner import RAGEvaluator, EvalConfig


def main():
    check_gate = "--check-gate" in sys.argv

    evaluator = RAGEvaluator(EvalConfig())
    metrics = evaluator.run_evaluation()

    print("\n=== RAGAS Evaluation Results ===")
    print(f"  Faithfulness:      {metrics['faithfulness']:.3f}  (threshold: 0.70)")
    print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.3f}  (threshold: 0.70)")
    print(f"  Context Precision: {metrics['context_precision']:.3f}")
    print(f"  Context Recall:    {metrics['context_recall']:.3f}  (threshold: 0.60)")
    print(f"  Samples evaluated: {metrics['num_samples']}")
    print(f"\nResults saved to data/eval/eval_results.json")

    if check_gate:
        passed, failures = evaluator.check_quality_gate(metrics)
        if passed:
            print("\nQuality gate: PASSED ✓")
            sys.exit(0)
        else:
            print("\nQuality gate: FAILED ✗")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)


if __name__ == "__main__":
    main()
