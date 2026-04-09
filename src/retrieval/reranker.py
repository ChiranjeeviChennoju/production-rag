from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder re-ranker for improved precision.

    Unlike bi-encoders (which embed query and chunks separately), a cross-encoder
    scores (query, chunk) pairs jointly, looking at both at the same time.
    Much more accurate, but slower — so we only run it on the top ~20 candidates
    from hybrid retrieval, then keep the top 5 for the LLM.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        """
        Re-rank a list of retrieval results using the cross-encoder.

        Args:
            query: the user query
            results: list of dicts from HybridRetriever (each has 'text', 'metadata', 'score')
            top_k: how many top results to keep after re-ranking
        """
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
