import os
import re
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import streamlit as st

# LangChain / loaders / vectorstores
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# --------------------------- App Config ---------------------------
st.set_page_config(
    page_title="Text-only PDF RAG (Groq + FAISS)",
    page_icon="🔎",
    layout="wide",
)


# ------------------------- Defaults (from notebook) -------------------------
# Based on the provided notebook:
# - Loader: PyPDFLoader
# - Chunker: RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# - Embeddings: sentence-transformers/all-mpnet-base-v2
# - Retriever: FAISS, similarity search k=5
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


# ------------------------- Helpers -------------------------
def preprocess_text(text: str) -> str:
    """
    Minimal preprocessing as in the notebook:
    - Collapse whitespace
    - Remove 'Page <num>'
    - Keep common punctuation and word chars
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\w\s\.\,\!\?]", "", text)
    return text.strip()


def iter_documents_from_dir(data_dir: str) -> Iterable:
    """
    Yields LangChain Documents from PDFs and TXTs under data_dir.
    Loads per-file to avoid holding everything in memory at once.
    """
    data_path = Path(data_dir)
    pdf_files = sorted(glob.glob(str(data_path / "**" / "*.pdf"), recursive=True))
    txt_files = sorted(glob.glob(str(data_path / "**" / "*.txt"), recursive=True))

    # PDF files
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        for d in docs:
            d.page_content = preprocess_text(d.page_content)
            yield d

    # TXT files
    for txt in txt_files:
        loader = TextLoader(txt, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.page_content = preprocess_text(d.page_content)
            yield d


def split_documents(
    docs_iter: Iterable,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable:
    """
    Split incoming documents into chunks using RecursiveCharacterTextSplitter.
    Yields chunks incrementally to keep memory bounded.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Buffer documents in small batches before splitting
    batch: List = []
    BATCH_DOCS = 32  # tune for memory/throughput
    for d in docs_iter:
        batch.append(d)
        if len(batch) >= BATCH_DOCS:
            for c in splitter.split_documents(batch):
                yield c
            batch.clear()
    if batch:
        for c in splitter.split_documents(batch):
            yield c


@st.cache_resource(show_spinner=False)
def get_embeddings(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    encode_batch_size: int,
) -> HuggingFaceEmbeddings:
    """
    Cache the embeddings model only; large arrays or indexes are not cached.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": encode_batch_size,
            "normalize_embeddings": normalize_embeddings,
        },
    )


def build_faiss_index(
    data_dir: str,
    index_dir: str,
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
    docs_batch_size: int = 256,
) -> Tuple[int, str]:
    """
    Incrementally build a FAISS index from PDFs/TXTs to avoid memory spikes.
    - Streams documents -> preprocess -> chunk
    - Adds to FAISS in small batches
    - Persists to index_dir
    Returns (num_chunks_added, index_dir)
    """
    os.makedirs(index_dir, exist_ok=True)

    # Stream chunks
    chunk_iter = split_documents(
        iter_documents_from_dir(data_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # First small batch initializes the store
    first_batch: List = []
    total_chunks = 0
    for c in chunk_iter:
        first_batch.append(c)
        if len(first_batch) >= docs_batch_size:
            break

    if not first_batch:
        # No data found
        return 0, index_dir

    db = FAISS.from_documents(first_batch, embeddings)
    total_chunks += len(first_batch)

    # Add subsequent chunks in batches
    add_batch: List = []
    for c in chunk_iter:
        add_batch.append(c)
        if len(add_batch) >= docs_batch_size:
            db.add_documents(add_batch)
            total_chunks += len(add_batch)
            add_batch.clear()
    if add_batch:
        db.add_documents(add_batch)
        total_chunks += len(add_batch)

    # Persist
    db.save_local(index_dir)
    return total_chunks, index_dir


def load_faiss_index(index_dir: str, embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """
    Loads a persisted FAISS index if present.
    """
    if not Path(index_dir).exists():
        return None
    try:
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,  # needed for LC persisted stores
        )
    except Exception:
        return None


# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.header("Settings")

    data_dir = st.text_input("Data directory (PDF/TXT)", value="data")
    index_dir = st.text_input("Index directory (FAISS)", value="vecdb")

    st.markdown("---")
    st.subheader("Chunking")
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=10)

    st.markdown("---")
    st.subheader("Embeddings")
    embed_model = st.text_input(
        "HuggingFace Embedding Model",
        value=DEFAULT_EMBED_MODEL,
        help="E.g., sentence-transformers/all-mpnet-base-v2 or sentence-transformers/all-MiniLM-L6-v2",
    )
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    normalize = st.toggle("Normalize embeddings", value=True)
    encode_bs = st.slider("Encode batch size", min_value=4, max_value=128, value=16)

    st.markdown("---")
    st.subheader("Retrieval")
    top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=DEFAULT_TOP_K)

    st.markdown("---")
    st.subheader("Groq LLM")
    groq_key = st.text_input("GROQ_API_KEY", type="password", help="Enter Groq API key")
    groq_model = st.text_input("Groq Model", value=DEFAULT_GROQ_MODEL)

    st.markdown("---")
    st.caption("Only PDFs and TXTs are indexed; image features are intentionally removed.")


# ------------------------------ Cached resources ------------------------------
embeddings = get_embeddings(
    model_name=embed_model,
    device=device,
    normalize_embeddings=normalize,
    encode_batch_size=encode_bs,
)


# ------------------------------- Main UI -------------------------------
st.title("🔎 Text-only PDF RAG — Streamlit (Groq + FAISS)")
st.write("Index PDFs/TXTs into FAISS and query with a Groq LLM over retrieved context.")


tab_search, tab_build = st.tabs(["Search", "Build / Update Index"])


with tab_build:
    st.subheader("Index Builder")
    st.write("Build or refresh the FAISS index from files under the data directory.")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Build / Rebuild Text Index", use_container_width=True):
            with st.status("Building text index…", expanded=True):
                n_chunks, out_dir = build_faiss_index(
                    data_dir=data_dir,
                    index_dir=index_dir,
                    embeddings=embeddings,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    docs_batch_size=256,  # tune for memory
                )
                st.write(f"Chunks embedded: {n_chunks}")
                st.write(f"Saved index dir: {out_dir}")
            st.success("Text index ready.")

    with cols[1]:
        if st.button("Load Existing Index", use_container_width=True):
            with st.status("Loading index…", expanded=False):
                db_tmp = load_faiss_index(index_dir, embeddings)
                if db_tmp is None:
                    st.error("No valid index found at the specified directory.")
                else:
                    st.success("Index loaded into memory.")

    st.markdown("---")
    st.write("Paths:")
    st.code(
        f"DATA_DIR={os.path.abspath(data_dir)}\nINDEX_DIR={os.path.abspath(index_dir)}",
        language="bash",
    )


with tab_search:
    st.subheader("Search & Ask")
    query = st.text_input("Ask a question grounded in the indexed PDFs/TXTs")
    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Retrieve Only", type="secondary"):
            db = load_faiss_index(index_dir, embeddings)
            if db is None:
                st.error("No index found; build it first.")
            else:
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
                docs = retriever.get_relevant_documents(query)
                st.write("### Retrieved Context Snippets")
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {d.page_content}")

    with colB:
        if st.button("Search & Generate", type="primary"):
            if not groq_key:
                st.error("Please provide GROQ_API_KEY in the sidebar.")
            else:
                db = load_faiss_index(index_dir, embeddings)
                if db is None:
                    st.error("No index found; build it first.")
                else:
                    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

                    llm = ChatGroq(
                        temperature=0.0,
                        model_name=groq_model,
                        api_key=groq_key,
                    )

                    rag_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                    )

                    with st.status("Generating answer…", expanded=False):
                        result = rag_chain.invoke({"query": query})
                    st.write("### Answer")
                    st.write(result.get("result", "").strip())
