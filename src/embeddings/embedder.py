from sentence_transformers import SentenceTransformer
from src.ingestion.chunker import Chunk


class Embedder:
    """Wraps sentence-transformers to generate embeddings for chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list) -> list:
        """Returns list of (chunk, embedding) tuples."""
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return list(zip(chunks, embeddings))

    def embed_query(self, query: str) -> list:
        """Embed a single query string."""
        return self.model.encode(query).tolist()
