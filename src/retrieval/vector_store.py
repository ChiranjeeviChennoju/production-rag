import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB wrapper for storing and querying chunk embeddings."""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks_with_embeddings: list):
        """Add (chunk, embedding) pairs to the store."""
        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk, embedding in chunks_with_embeddings:
            ids.append(chunk.chunk_id)
            embeddings.append(embedding.tolist())
            documents.append(chunk.text)
            metadatas.append({**chunk.metadata, "doc_id": chunk.doc_id})

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embedding: list, top_k: int = 10) -> list:
        """Return top_k most similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist  # cosine similarity
            })
        return hits
