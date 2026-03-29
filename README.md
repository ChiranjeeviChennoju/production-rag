# Production-Grade RAG System

A fully open-source, zero-cost Retrieval-Augmented Generation (RAG) system built for healthcare research documents. Ingests PDFs, performs hybrid retrieval (vector + keyword), re-ranks results with a cross-encoder, and generates cited answers via a local LLM — all evaluated through a CI pipeline.

> Built as a portfolio project to demonstrate production RAG engineering: not just "call an API", but the full stack.

---

## What makes this different from a basic RAG

| Basic RAG | This Project |
|---|---|
| Single vector search | Hybrid retrieval: ChromaDB + BM25 (RRF fusion) |
| No re-ranking | Cross-encoder re-ranker for precision |
| Hallucination-prone | Citation checker enforces answer grounding |
| Prompt hardcoded | Versioned YAML prompt management |
| No evaluation | RAGAS eval suite with CI quality gates |
| API key required | 100% local — Ollama + open-source models |

---

## Architecture

```
PDF / TXT
    │
    ▼
DocumentLoader ──► RecursiveChunker ──► Embedder (MiniLM-L6-v2)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                        ChromaDB                          BM25 Index
                      (vector search)               (keyword search)
                              │                               │
                              └───────────┬───────────────────┘
                                          ▼
                                 Hybrid Retriever (RRF)
                                          │
                                          ▼
                                 Cross-Encoder Reranker
                                          │
                                          ▼
                              LLM (Ollama / Claude API)
                                          │
                                          ▼
                               Citation Checker ──► Final Answer
```

---

## Tech Stack (All Free / Open-Source)

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (local, persistent) |
| Keyword Search | `rank-bm25` (BM25Okapi) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (local) | Ollama + Mistral 7B / TinyLlama |
| LLM (cloud option) | Claude API |
| API Layer | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| CI/CD | GitHub Actions |
| Document Parsing | PyMuPDF |

---

## Project Structure

```
production-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF + text loading
│   │   └── chunker.py         # Recursive chunking (paragraph → sentence → word)
│   ├── embeddings/
│   │   └── embedder.py        # Sentence-transformers wrapper
│   ├── retrieval/
│   │   ├── vector_store.py    # ChromaDB operations
│   │   ├── bm25_store.py      # BM25 keyword index
│   │   ├── hybrid.py          # Reciprocal Rank Fusion
│   │   └── reranker.py        # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── llm_client.py      # Ollama + Claude API clients
│   │   ├── prompt_manager.py  # YAML prompt versioning
│   │   └── citation_checker.py# Hallucination detection
│   ├── evaluation/
│   │   └── eval_runner.py     # RAGAS evaluation
│   └── api/
│       ├── main.py            # FastAPI app
│       └── routes.py          # API endpoints
├── ui/
│   └── app.py                 # Streamlit frontend
├── prompts/
│   └── rag_prompt_v1.yaml     # Versioned prompts
├── scripts/
│   ├── ingest.py              # Index documents
│   ├── query.py               # CLI query
│   └── evaluate.py            # Run RAGAS eval
├── data/
│   ├── raw/                   # Source PDFs (gitignored)
│   ├── processed/             # Chunked + indexed data
│   └── eval/                  # Ground-truth Q&A pairs
├── tests/
└── .github/workflows/
    └── eval_ci.yml            # CI: run eval on every PR
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/production-rag.git
cd production-rag

python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate

pip install -r requirements.txt
```

### 2. Install Ollama (local LLM)

```bash
# macOS
brew install ollama
ollama serve &
ollama pull tinyllama    # lightweight, ~637MB
# ollama pull mistral   # better quality, needs 8GB+ free RAM
```

### 3. Add documents

Drop PDFs or `.txt` files into `data/raw/`. For healthcare research, [PubMed Central Open Access](https://pmc.ncbi.nlm.nih.gov/) is a great free source.

### 4. Ingest

```bash
python scripts/ingest.py
```

This loads, chunks, embeds, and indexes all documents in `data/raw/`.

### 5. Query

```bash
python scripts/query.py "What are the side effects of metformin?"
```

---

## How the Retrieval Works

### Hybrid Search (Vector + BM25)

Vector search is great for semantic similarity but misses exact keyword matches. BM25 is great for keywords but misses paraphrases. This system uses both and fuses results with **Reciprocal Rank Fusion (RRF)**:

```
RRF score = Σ  1 / (k + rank_i)
```

No score normalization needed — RRF is rank-based, not score-based.

### Cross-Encoder Re-Ranking

After hybrid retrieval returns the top 20 candidates, a cross-encoder scores every `(query, chunk)` pair directly — much more accurate than bi-encoder similarity. The top 5 are passed to the LLM.

### Citation Grounding

The citation checker verifies that the LLM's answer is grounded in the retrieved context using n-gram overlap. Answers with low grounding confidence are flagged.

---

## Evaluation

RAGAS metrics tracked on every PR via GitHub Actions:

| Metric | Threshold | What it measures |
|---|---|---|
| Faithfulness | ≥ 0.70 | Is the answer grounded in context? |
| Answer Relevancy | ≥ 0.70 | Does the answer address the question? |
| Context Precision | — | Are retrieved chunks relevant? |
| Context Recall | ≥ 0.60 | Did retrieval find the right chunks? |

```bash
python scripts/evaluate.py
```

---

## Running the API

```bash
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs
```

## Running the UI

```bash
streamlit run ui/app.py
```

---

## Apple Silicon Notes

This project runs fully local on M1/M2/M3/M4 Macs:
- Ollama uses Metal GPU acceleration automatically
- `sentence-transformers` runs on MPS (Apple's GPU backend)
- ChromaDB and BM25 are CPU-only (no issues)
- Recommended model on 8GB RAM: `tinyllama` — fits comfortably alongside the embedding model

---

## Roadmap

- [x] Phase 1 — Basic RAG pipeline (loader, chunker, embedder, ChromaDB, BM25, Ollama)
- [x] Phase 2 — Hybrid retrieval + cross-encoder re-ranking + citation checker
- [ ] Phase 3 — RAGAS evaluation + GitHub Actions CI
- [ ] Phase 4 — FastAPI layer + Streamlit UI
- [ ] Phase 5 — Docker + docker-compose for reproducible deployment

---

## License

MIT
