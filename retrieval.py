#!/usr/bin/env python3
"""FAISS-based approximate nearest-neighbour retrieval.

Builds an exact inner-product index (`IndexFlatIP`) over pre-computed job
embeddings.  Because vectors are L2-normalized, inner product equals
cosine similarity — giving us an exact-cosine search.

Usage:
    python retrieval.py          # Build and save the index

API:
    build_index(embeddings) → faiss.IndexFlatIP
    load_index(path)         → faiss.IndexFlatIP
    search(index, query, k)  → (distances, indices)
"""

import os

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = os.path.join("embeddings", "job_embeddings.npy")
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "job_index.faiss")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS IndexFlatIP from an (N, D) embedding matrix."""
    embeddings = embeddings.astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: str = INDEX_PATH) -> None:
    """Persist a FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str = INDEX_PATH) -> faiss.IndexFlatIP:
    """Load a previously saved FAISS index."""
    return faiss.read_index(path)


def search(
    index: faiss.IndexFlatIP,
    query: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index and return (distances, indices)."""
    query = query.astype("float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    return index.search(query, k)


def main():
    """Build index from saved embeddings and write to disk."""
    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  Shape: {embeddings.shape}")

    print("Building FAISS IndexFlatIP...")
    index = build_index(embeddings)
    save_index(index)

    print(f"\n  FAISS index saved to {INDEX_PATH}")
    print(f"  Vectors: {index.ntotal}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Type: IndexFlatIP (exact cosine similarity)")


if __name__ == "__main__":
    main()
