"""
Run: python scripts/ingest.py
Loads PDFs from data/raw/, chunks them, embeds, and stores in ChromaDB + BM25.
"""
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import RecursiveChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store


def main():
    raw_dir = Path("data/raw")
    if not any(raw_dir.iterdir()):
        print("No files found in data/raw/ — add some PDFs or .txt files first.")
        return

    print("Loading documents...")
    loader = DocumentLoader()
    docs = loader.load_directory(raw_dir)
    print(f"  Loaded {len(docs)} document(s)")

    print("Chunking...")
    chunker = RecursiveChunker()
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.content, doc.doc_id, doc.metadata)
        all_chunks.extend(chunks)
    print(f"  Created {len(all_chunks)} chunks")

    print("Embedding...")
    embedder = Embedder()
    chunks_with_embeddings = embedder.embed_chunks(all_chunks)

    print("Storing in ChromaDB...")
    vector_store = VectorStore()
    vector_store.add(chunks_with_embeddings)

    print("Building BM25 index...")
    bm25_store = BM25Store()
    bm25_store.build(all_chunks)
    bm25_store.save("data/processed/bm25_index.pkl")

    print("Done! Your documents are indexed and ready to query.")


if __name__ == "__main__":
    main()
