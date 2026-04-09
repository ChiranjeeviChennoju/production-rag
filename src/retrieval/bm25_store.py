import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


class BM25Store:
    """BM25 keyword index over document chunks."""

    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def build(self, chunks: list):
        """Build BM25 index from a list of Chunk objects."""
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, query: str, top_k: int = 10) -> list:
        """Return top_k chunks by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self.chunks[i].text, "metadata": self.chunks[i].metadata, "score": scores[i]}
            for i in top_indices
        ]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.bm25 = data["bm25"]
