import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class EvalConfig:
    eval_data_path: str = "data/eval/eval_dataset.json"
    output_path: str = "data/eval/eval_results.json"
    min_faithfulness: float = 0.7
    min_context_recall: float = 0.6
    min_answer_relevancy: float = 0.7


class RAGEvaluator:
    """Evaluate the RAG pipeline using RAGAS metrics."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()

    def load_eval_data(self) -> list:
        with open(self.config.eval_data_path) as f:
            data = json.load(f)
        return data["pairs"]

    def run_pipeline(self, question: str) -> dict:
        from src.embeddings.embedder import Embedder
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.generation.llm_client import GroqClient
        from src.generation.prompt_manager import PromptManager

        embedder = Embedder()
        query_embedding = embedder.embed_query(question)

        vector_store = VectorStore()
        bm25_store = BM25Store()
        bm25_store.load("data/processed/bm25_index.pkl")

        retriever = HybridRetriever(vector_store, bm25_store)
        candidates = retriever.retrieve(question, query_embedding, top_k=20)

        reranker = Reranker()
        results = reranker.rerank(question, candidates, top_k=3)

        # truncate chunks to keep prompt under Groq's 6000 TPM free tier limit
        context = "\n\n".join(
            f"[{i+1}] (Source: {r['metadata'].get('source', 'unknown')})\n{r['text'][:800]}"
            for i, r in enumerate(results)
        )

        prompt_mgr = PromptManager()
        template = prompt_mgr.load_prompt("rag_prompt", version="v1")
        prompt = prompt_mgr.format_prompt(template, context=context, question=question)

        llm = GroqClient()
        answer = llm.generate(prompt)

        return {
            "answer": answer,
            "contexts": [r["text"] for r in results]
        }

    def run_evaluation(self) -> dict:
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset
        from config.settings import settings

        # Groq as judge LLM — n=1 enforced, sequential to avoid rate limits
        groq_llm = LangchainLLMWrapper(
            ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=settings.groq_api_key,
                n=1
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        pairs = self.load_eval_data()
        print(f"Running evaluation on {len(pairs)} Q&A pairs...\n")

        questions, ground_truths, answers, contexts = [], [], [], []

        for i, pair in enumerate(pairs):
            if i > 0:
                time.sleep(60)  # wait full minute between calls — 6000 TPM resets every 60s
            print(f"  [{i+1}/{len(pairs)}] {pair['question'][:60]}...")
            result = self.run_pipeline(pair["question"])
            questions.append(pair["question"])
            ground_truths.append(pair["expected_answer"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            m.llm = groq_llm
            m.embeddings = embeddings

        # max_workers=1 runs sequentially — avoids Groq rate limits
        # timeout=120 gives each scoring call enough time
        run_config = RunConfig(max_workers=1, timeout=120, max_retries=3)

        print("\nScoring with RAGAS (sequential mode to respect rate limits)...\n")
        result = evaluate(dataset, metrics=metrics, run_config=run_config)

        metrics = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
            "num_samples": len(pairs)
        }

        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def check_quality_gate(self, metrics: dict) -> tuple:
        failures = []
        if metrics["faithfulness"] < self.config.min_faithfulness:
            failures.append(f"faithfulness {metrics['faithfulness']:.3f} < {self.config.min_faithfulness}")
        if metrics["context_recall"] < self.config.min_context_recall:
            failures.append(f"context_recall {metrics['context_recall']:.3f} < {self.config.min_context_recall}")
        if metrics["answer_relevancy"] < self.config.min_answer_relevancy:
            failures.append(f"answer_relevancy {metrics['answer_relevancy']:.3f} < {self.config.min_answer_relevancy}")
        return len(failures) == 0, failures
