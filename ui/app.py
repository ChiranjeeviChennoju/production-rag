"""
Streamlit UI for the RAG system.
Run: streamlit run ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm_client import GroqClient
from src.generation.prompt_manager import PromptManager
from src.generation.citation_checker import CitationChecker


# ---------- Page config ----------
st.set_page_config(
    page_title="Production RAG",
    page_icon="🔍",
    layout="wide",
)


# ---------- Cached resources (load models once per session) ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return Embedder()


@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store():
    return VectorStore()


@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25_store():
    store = BM25Store()
    store.load("data/processed/bm25_index.pkl")
    return store


@st.cache_resource(show_spinner="Loading cross-encoder re-ranker...")
def load_reranker():
    return Reranker()


@st.cache_resource
def load_llm():
    return GroqClient()


@st.cache_resource
def load_prompt_manager():
    return PromptManager()


@st.cache_resource
def load_citation_checker():
    return CitationChecker()


# ---------- RAG pipeline ----------
def run_rag(query: str, top_k: int, use_reranking: bool, prompt_version: str):
    embedder = load_embedder()
    vector_store = load_vector_store()
    bm25_store = load_bm25_store()
    llm = load_llm()
    prompt_mgr = load_prompt_manager()
    checker = load_citation_checker()

    query_embedding = embedder.embed_query(query)

    retriever = HybridRetriever(vector_store, bm25_store)
    fetch_k = top_k * 4 if use_reranking else top_k
    candidates = retriever.retrieve(query, query_embedding, top_k=fetch_k)

    if use_reranking:
        reranker = load_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    context = "\n\n".join(
        f"[{i+1}] (Source: {r['metadata'].get('source', 'unknown')})\n{r['text']}"
        for i, r in enumerate(results)
    )

    template = prompt_mgr.load_prompt("rag_prompt", version=prompt_version)
    prompt = prompt_mgr.format_prompt(template, context=context, question=query)

    answer = llm.generate(prompt)
    citation_result = checker.check(answer, [r["text"] for r in results])

    return answer, results, citation_result


# ---------- UI ----------
st.title("🔍 Production RAG System")
st.caption("Hybrid retrieval (Vector + BM25) → Cross-encoder re-rank → Versioned prompt → Groq LLM → Citation check")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=15, value=5)
    use_reranking = st.checkbox("Enable cross-encoder re-ranking", value=True)
    prompt_version = st.selectbox("Prompt version", ["v1", "v2"], index=0)
    st.divider()
    st.markdown(
        "**Pipeline:**\n\n"
        "1. Query embedded with `all-MiniLM-L6-v2`\n"
        "2. ChromaDB + BM25 retrieve candidates\n"
        "3. Reciprocal Rank Fusion merges results\n"
        "4. Cross-encoder re-ranks top candidates\n"
        "5. Versioned YAML prompt loaded\n"
        "6. Groq LLM generates cited answer\n"
        "7. Citation checker verifies grounding"
    )

query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. What are the main macroeconomic achievements in the 2025 Economic Report?",
)

if st.button("🔎 Search", type="primary", disabled=not query):
    with st.spinner("Retrieving, re-ranking, and generating answer..."):
        try:
            answer, results, citation_result = run_rag(
                query, top_k, use_reranking, prompt_version
            )

            st.subheader("Answer")
            if citation_result.is_grounded:
                st.success(f"Grounded ✓  (confidence: {citation_result.confidence:.0%})")
            else:
                st.warning(f"Low grounding confidence: {citation_result.confidence:.0%} — verify against sources below")

            st.markdown(answer)

            st.divider()

            st.subheader(f"Sources ({len(results)})")
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                with st.expander(f"[{i}] {source}"):
                    st.write(r["text"])

        except FileNotFoundError:
            st.error(
                "BM25 index not found. Run `python scripts/ingest.py` first to "
                "ingest your documents."
            )
        except Exception as e:
            st.error(f"Error: {e}")
