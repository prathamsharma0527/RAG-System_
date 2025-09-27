import os
import glob
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
from pypdf import PdfReader
import torch


class VectorRetriever:
    """
    Handles:
    - Loading raw text files from data/ (txt, md, pdf)
    - Loading images from data/ (jpg, png, jpeg, webp)
    - Chunking (for text)
    - Embedding via SentenceTransformer (MiniLM for text, CLIP for images)
    - Separate FAISS indexes for text and images with on-disk persistence

    You can extend this to other modalities as needed.
    """

    def __init__(self, data_dir: str, chunks_dir: str, index_path: str, model_name: str):
        self.data_dir = data_dir
        self.chunks_dir = chunks_dir
        # Maintain a directory for indexes; accept either a directory or a file path
        if os.path.isdir(index_path):
            self.index_dir = index_path
        else:
            self.index_dir = os.path.dirname(index_path) or "."
        self.text_index_path = os.path.join(self.index_dir, "faiss_text.index")
        self.image_index_path = os.path.join(self.index_dir, "faiss_image.index")
        self.model_name = model_name

        # GPU detection and optimization
        self.device = self._get_optimal_device()
        
        # Text and image models with device optimization
        self._text_model = SentenceTransformer(model_name, device=self.device)
        # CLIP model from sentence-transformers supports both text and image encoders
        self._image_model = SentenceTransformer("clip-ViT-B-32", device=self.device)
        
        # Cached indexes and payloads
        self._text_index = None
        self._image_index = None
        self._text_chunks: List[str] = []
        self._image_paths: List[str] = []
        
        # Performance controls (can be overridden via env vars)
        self.text_batch_size = int(os.getenv("TEXT_EMB_BATCH", "32"))
        self.image_batch_size = int(os.getenv("IMAGE_EMB_BATCH", "8"))
        
        # Optimize threading based on device
        self._optimize_threading()
        
        print(f"Initialized VectorRetriever with device: {self.device}")

    # ------------------------- Public API -------------------------
    def build_or_update_index(self, modality: str = "text", chunk_size: int = 500, overlap: int = 50) -> int:
        if modality == "text":
            return self._build_text_index(chunk_size=chunk_size, overlap=overlap)
        elif modality == "image":
            return self._build_image_index()
        else:
            raise ValueError("modality must be 'text' or 'image'")

    def search(self, query: str, top_k: int = 5, modality: str = "text") -> List[str]:
        if modality == "text":
            return self._search_text(query, top_k)
        elif modality == "image":
            return self._search_images(query, top_k)
        else:
            raise ValueError("modality must be 'text' or 'image'")

    # ------------------------- Internals -------------------------
    def _load_texts(self) -> List[str]:
        """Load .txt, .md, and .pdf files from data/."""
        texts = []
        for path in glob.glob(os.path.join(self.data_dir, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in {".txt", ".md"}:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
                except Exception:
                    continue
            elif ext == ".pdf":
                try:
                    reader = PdfReader(path)
                    txt = []
                    for p in reader.pages:
                        try:
                            txt.append(p.extract_text() or "")
                        except Exception:
                            continue
                    if txt:
                        texts.append("\n".join(txt))
                except Exception:
                    continue
        if not texts:
            texts = ["This is a placeholder text. Add files to data/ to build a real index."]
        return texts

    def _chunk_texts(self, texts: List[str], chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        for t in texts:
            start = 0
            while start < len(t):
                end = min(start + chunk_size, len(t))
                chunks.append(t[start:end])
                start = end - overlap
                if start < 0:
                    start = 0
                if start >= len(t):
                    break
        return chunks

    def _save_chunks(self, chunks: List[str]):
        os.makedirs(self.chunks_dir, exist_ok=True)
        with open(os.path.join(self.chunks_dir, "chunks.txt"), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.replace("\n", " ") + "\n")

    def _read_chunks(self) -> List[str]:
        path = os.path.join(self.chunks_dir, "chunks.txt")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    # --------- Text index helpers ---------
    def _build_text_index(self, chunk_size: int, overlap: int) -> int:
        texts = self._load_texts()
        chunks = self._chunk_texts(texts, chunk_size=chunk_size, overlap=overlap)
        self._text_chunks = chunks
        os.makedirs(self.index_dir, exist_ok=True)

        # Create index lazily using model embedding dimension
        dim = getattr(self._text_model, "get_sentence_embedding_dimension", lambda: None)()
        if not dim:
            # Fallback: compute a small probe batch to infer dim
            probe = self._text_model.encode([chunks[0] if chunks else ""], convert_to_numpy=True, normalize_embeddings=True)
            dim = int(probe.shape[1])
        index = faiss.IndexFlatIP(dim)

        # Encode and add in batches to reduce peak memory
        bs = max(8, self.text_batch_size)
        total_batches = (len(chunks) + bs - 1) // bs

        # Process with progress indication
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            emb = self._text_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=bs,
            ).astype(np.float32)
            index.add(emb)

        faiss.write_index(index, self.text_index_path)
        self._text_index = index
        self._save_chunks(chunks)
        return len(chunks)

    def _ensure_text_loaded(self):
        if self._text_index is None:
            if os.path.exists(self.text_index_path):
                self._text_index = faiss.read_index(self.text_index_path)
            else:
                self._build_text_index(chunk_size=500, overlap=50)
        if not self._text_chunks:
            self._text_chunks = self._read_chunks()

    def _search_text(self, query: str, top_k: int) -> List[str]:
        self._ensure_text_loaded()
        q = self._text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self._text_index.search(q, top_k)
        I = I[0]
        return [self._text_chunks[i] for i in I if 0 <= i < len(self._text_chunks)]

    # --------- Image index helpers ---------
    def _iter_image_paths(self) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = []
        for path in glob.glob(os.path.join(self.data_dir, "**", "*"), recursive=True):
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts:
                paths.append(path)
        return paths

    def _build_image_index(self) -> int:
        paths = self._iter_image_paths()
        if not paths:
            # Create a tiny dummy index to avoid errors
            self._image_paths = []
            dim = 512
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, self.image_index_path)
            self._image_index = index
            return 0

        # Create index using known CLIP embedding dimension (512 for ViT-B/32)
        dim = getattr(self._image_model, "get_sentence_embedding_dimension", lambda: 512)()
        if not dim:
            dim = 512
        index = faiss.IndexFlatIP(int(dim))

        bs = max(4, self.image_batch_size)
        total_batches = (len(paths) + bs - 1) // bs

        # Encode images in small batches to limit memory usage
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i + bs]
            batch_imgs = []
            try:
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        batch_imgs.append(img)
                    except Exception as e:
                        print(f"Warning: Could not load image {p}: {e}")
                        continue

                if batch_imgs:
                    emb = self._image_model.encode(
                        batch_imgs,
                        batch_size=len(batch_imgs),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)
                    index.add(emb)

            finally:
                # Explicitly close images to free memory
                for img in batch_imgs:
                    try:
                        img.close()
                    except Exception:
                        pass

        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, self.image_index_path)
        self._image_index = index
        self._image_paths = paths
        # Save image list for retrieval mapping
        os.makedirs(self.chunks_dir, exist_ok=True)
        with open(os.path.join(self.chunks_dir, "images.txt"), "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")
        return len(paths)

    def _ensure_image_loaded(self):
        if self._image_index is None:
            if os.path.exists(self.image_index_path):
                self._image_index = faiss.read_index(self.image_index_path)
            else:
                self._build_image_index()
        if not self._image_paths:
            path = os.path.join(self.chunks_dir, "images.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._image_paths = [line.strip() for line in f.readlines()]
            else:
                self._image_paths = []

    def _search_images(self, query: str, top_k: int) -> List[str]:
        self._ensure_image_loaded()
        # Encode query text using CLIP text encoder (same model)
        q = self._image_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self._image_index.search(q, top_k)
        I = I[0]
        return [self._image_paths[i] for i in I if 0 <= i < len(self._image_paths)]
