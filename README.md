Streamlit PDF/TXT RAG App

A Streamlit application that builds a local FAISS index from PDFs and plain text files, then answers questions using retrieved context with a Groq LLM. The pipeline uses LangChain loaders, recursive chunking, HuggingFace embeddings, FAISS retrieval, and a minimal UI for building and querying the index.

Features
Text-only ingestion from PDFs and TXTs with lightweight preprocessing to normalize whitespace and remove page-number noise.

Recursive chunking with configurable size and overlap (defaults: 300, 50) for robust retrieval.

Sentence embeddings via HuggingFace embeddings (default: sentence-transformers/all-mpnet-base-v2) with optional normalization.

Local FAISS vector store persisted to disk, enabling fast reloads without re-embedding.

Groq LLM integration (default model: openai/gpt-oss-20b, temperature=0) for grounded answers over retrieved context.

Streamlit UI with two tabs: Index build/update and Search/RAG answer generation.

Repository structure
text
./
├── data/              # Source documents (PDF/TXT); place files here
├── vecdb/             # Persisted FAISS index directory (created on build)
├── notebooks/
│   └── app.ipynb      # Notebook the app was built from
├── app.py             # Streamlit application entrypoint
├── requirements.txt   # Python dependencies (optional)
└── README.md          # This file
Image features are intentionally excluded to keep the app focused on PDF/TXT RAG.

Prerequisites
Python 3.9+

A Groq API key for answer generation (can build and retrieve without it)

Installation
bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Option A: if requirements.txt exists
pip install -r requirements.txt

# Option B: install explicit deps
pip install streamlit langchain langchain-community langchain-huggingface \
            sentence-transformers faiss-cpu pypdf langchain-groq
Configuration
Export the Groq API key (or enter it in the app sidebar):

bash
export GROQ_API_KEY=your_groq_key_here
Default components and parameters can be changed in the sidebar at runtime:

Embedding model: sentence-transformers/all-mpnet-base-v2

Device: cpu or cuda

Chunking: chunk_size=300, chunk_overlap=50

Retrieval: top_k=5

LLM: openai/gpt-oss-20b, temperature=0

Running
bash
streamlit run app.py
The app reads documents from data/ and persists the FAISS index to vecdb/.

All build and search actions occur from the UI.

Usage
Place PDFs and/or TXTs under data/

Open the Build / Update Index tab and click “Build / Rebuild Text Index”

Switch to the Search tab

Enter a question and choose either:

Retrieve Only: view the top-k chunks

Search & Generate: run full RAG with the Groq LLM

Notes on performance and stability
For large corpora, keep encode batch size modest in the sidebar to limit RAM usage.

Run on CPU by default for broad compatibility; switch to CUDA if available for faster embedding.

Persisted FAISS in vecdb/ avoids recomputation across app restarts.

Security
Do not hardcode the API key in code or commit it to source control.

Prefer environment variables or secure secret storage.

Troubleshooting
Missing index: ensure vecdb/ exists after a successful build; if not, rebuild the index.

No results: verify data/ contains readable PDFs/TXTs and that chunking settings are reasonable.

Memory pressure: reduce encode batch size, and avoid placing extremely large monolithic files without chunking.