"""
Run: python scripts/query.py "your question here"
Quick test to verify retrieval + generation works end-to-end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.generation.llm_client import GroqClient


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

    retriever = HybridRetriever(vector_store, bm25_store)
    results = retriever.retrieve(query, query_embedding, top_k=5)

    context = "\n\n".join(
        f"[{i+1}] (Source: {r['metadata'].get('source', 'unknown')})\n{r['text']}"
        for i, r in enumerate(results)
    )

    prompt = f"""You are a medical research assistant. Answer the question using only the context provided.
Cite the source numbers like [1], [2] when you use information from them.

Context:
{context}

Question: {query}

Answer:"""

    print("Generating answer...\n")
    llm = GroqClient()
    answer = llm.generate(prompt)
    print(answer)


if __name__ == "__main__":
    main()
