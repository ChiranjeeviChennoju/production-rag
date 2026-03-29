from collections import defaultdict


class HybridRetriever:
    """
    Combines vector and BM25 results using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1 / (k + rank)) across both result lists.
    """

    def __init__(self, vector_store, bm25_store, k: int = 60):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.k = k

    def retrieve(self, query: str, query_embedding: list, top_k: int = 10) -> list:
        vector_hits = self.vector_store.query(query_embedding, top_k=top_k * 2)
        bm25_hits = self.bm25_store.query(query, top_k=top_k * 2)

        scores = defaultdict(float)
        texts = {}
        metas = {}

        for rank, hit in enumerate(vector_hits):
            key = hit["text"][:100]  # use text snippet as key
            scores[key] += 1 / (self.k + rank + 1)
            texts[key] = hit["text"]
            metas[key] = hit["metadata"]

        for rank, hit in enumerate(bm25_hits):
            key = hit["text"][:100]
            scores[key] += 1 / (self.k + rank + 1)
            texts[key] = hit["text"]
            metas[key] = hit["metadata"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"text": texts[k], "metadata": metas[k], "score": s}
            for k, s in ranked
        ]
