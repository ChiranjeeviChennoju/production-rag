"""
Run: python scripts/query.py "your question here"
End-to-end Phase 2 pipeline:
hybrid retrieval -> cross-encoder rerank -> versioned prompt -> LLM -> citation check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm_client import GroqClient
from src.generation.prompt_manager import PromptManager
from src.generation.citation_checker import CitationChecker


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py \"your question here\"")
        return

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")

    embedder = Embedder()
    query_embedding = embedder.embed_query(query)

    vector_store = VectorStore()
    bm25_store = BM25Store()
    bm25_store.load("data/processed/bm25_index.pkl")

    # 1. Hybrid retrieval (over-fetch for re-ranking)
    retriever = HybridRetriever(vector_store, bm25_store)
    candidates = retriever.retrieve(query, query_embedding, top_k=20)

    # 2. Cross-encoder re-ranking
    reranker = Reranker()
    results = reranker.rerank(query, candidates, top_k=5)

    # 3. Build numbered context block
    context = "\n\n".join(
        f"[{i+1}] (Source: {r['metadata'].get('source', 'unknown')})\n{r['text']}"
        for i, r in enumerate(results)
    )

    # 4. Load versioned prompt
    prompt_mgr = PromptManager()
    template = prompt_mgr.load_prompt("rag_prompt", version="v1")
    prompt = prompt_mgr.format_prompt(template, context=context, question=query)

    # 5. Generate
    print("Generating answer...\n")
    llm = GroqClient()
    answer = llm.generate(prompt)

    # 6. Citation grounding check
    checker = CitationChecker()
    result = checker.check(answer, [r["text"] for r in results])

    print(answer)
    print("\n---")
    print(f"Grounded: {result.is_grounded}  |  Confidence: {result.confidence:.0%}  |  Citations found: {result.citations_found}")


if __name__ == "__main__":
    main()
