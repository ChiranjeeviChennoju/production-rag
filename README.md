# Production-Grade RAG System

A fully open-source Retrieval-Augmented Generation (RAG) system that ingests PDFs, performs hybrid retrieval, re-ranks with a cross-encoder, generates cited answers via an LLM, and verifies grounding to flag potential hallucinations.

Built end-to-end as a portfolio project to demonstrate **production RAG engineering** — not just "call an embedding API and stuff results into a prompt."

---

## What makes this different from a basic RAG

| Basic RAG | This Project |
|---|---|
| Single vector search | Hybrid retrieval: ChromaDB + BM25 (RRF fusion) |
| No re-ranking | Cross-encoder re-ranker for precision |
| Hallucination-prone | Citation grounding checker on every answer |
| Hardcoded prompts | Versioned YAML prompt management (A/B testable) |
| CLI only | Streamlit UI with retrieval inspection |

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
                                 Cross-Encoder Re-Ranker
                                          │
                                          ▼
                              Versioned Prompt Manager
                                          │
                                          ▼
                                       Groq LLM
                                          │
                                          ▼
                               Citation Grounding Check
                                          │
                                          ▼
                                    Cited Answer
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector DB | ChromaDB (local, persistent, cosine similarity) |
| Keyword Search | `rank-bm25` (BM25Okapi) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LLM | Groq API (`llama-3.1-8b-instant`)|
| Prompt Management | YAML templates + Pydantic settings |
| Frontend | Streamlit |
| Document Parsing | PyMuPDF |
| Config | `pydantic-settings` + `.env` |

---

## Project Structure

```
production-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # PDF + text loading
│   │   └── chunker.py           # Recursive chunking (paragraph → sentence → word)
│   ├── embeddings/
│   │   └── embedder.py          # Sentence-transformers wrapper
│   ├── retrieval/
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── bm25_store.py        # BM25 keyword index
│   │   ├── hybrid.py            # Reciprocal Rank Fusion
│   │   └── reranker.py          # Cross-encoder re-ranking
│   └── generation/
│       ├── llm_client.py        # Groq + Ollama clients
│       ├── prompt_manager.py    # YAML prompt loader
│       └── citation_checker.py  # Hallucination detection
├── prompts/
│   ├── rag_prompt_v1.yaml       # Versioned prompt templates
│   └── rag_prompt_v2.yaml
├── config/
│   └── settings.py              # Pydantic env-var loader
├── ui/
│   └── app.py                   # Streamlit frontend
├── scripts/
│   ├── ingest.py                # Index documents
│   └── query.py                 # CLI query
├── data/
│   ├── raw/                     # Source PDFs (gitignored)
│   └── processed/               # BM25 index (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key at [console.groq.com](https://console.groq.com).

### 2. Clone and install

```bash
git clone https://github.com/your-username/production-rag.git
cd production-rag

python -m venv rag_env
source rag_env/bin/activate          # Windows: rag_env\Scripts\activate

pip install -r requirements.txt
```

### 3. Add documents

Drop PDFs, `.txt`, or `.md` files into `data/raw/`.

### 4. Ingest

```bash
python scripts/ingest.py
```

This loads documents, chunks them, generates embeddings, and builds both the ChromaDB vector index and the BM25 keyword index.

### 5. Query — pick one

**CLI:**
```bash
python scripts/query.py "What were the main macroeconomic achievements?"
```

**Streamlit UI:**
```bash
streamlit run ui/app.py
```

The UI gives you:
- A `top_k` slider to control how many chunks are retrieved
- A toggle to enable/disable cross-encoder re-ranking
- A prompt version selector (`v1` / `v2`) for A/B testing
- A grounding badge showing whether the answer is well-supported by sources
- An expandable list of the actual chunks the LLM saw

---

## How it works

### Hybrid Search (Vector + BM25)

Vector search captures semantic meaning ("monetary tightening" matches "interest rate hikes"). BM25 captures exact keyword matches (rare drug names, acronyms, dates). The system runs both in parallel and fuses the results with **Reciprocal Rank Fusion**:

```
RRF score = Σ  1 / (k + rank_i)
```

No score normalization needed — RRF only uses rank position, not raw scores from incompatible scoring systems.

### Cross-Encoder Re-Ranking

Hybrid retrieval over-fetches the top 20 candidates. A cross-encoder then scores every `(query, chunk)` pair *jointly* — far more accurate than the bi-encoder similarity used in the first pass. The top 5 are passed to the LLM.

### Versioned Prompts

Prompts live in YAML files under `prompts/`, loaded by `PromptManager`. This means prompt changes can be versioned, A/B tested in the UI, and tracked separately from code changes.

### Citation Grounding

After the LLM generates an answer, the citation checker scans for `[1]`, `[2]` markers and verifies that key claims actually appear in the retrieved chunks (n-gram overlap). Answers below a confidence threshold are flagged in the UI as potentially hallucinated.

---

## Roadmap

- [x] Document ingestion + recursive chunking
- [x] Hybrid retrieval (vector + BM25 with RRF fusion)
- [x] Cross-encoder re-ranking
- [x] Versioned YAML prompt management
- [x] Citation grounding checker
- [x] Streamlit UI with retrieval inspection
- [ ] RAGAS evaluation suite
- [ ] GitHub Actions CI with quality gates
- [ ] FastAPI serving layer
- [ ] Dockerized deployment

---

## License

MIT
