#!/usr/bin/env python3
"""Hybrid re-ranking — blend semantic (FAISS) and keyword (BM25) scores.

The re-ranker combines dense-vector cosine similarity with sparse BM25
scores using a tunable blending coefficient α:

    hybrid_score = α × semantic_norm + (1 − α) × bm25_norm

Both score distributions are min-max normalised to [0, 1] before blending
so that neither signal dominates due to scale differences.

Usage (module):
    from rerank import build_bm25_index, hybrid_rank
"""

import os

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOBS_CSV = os.path.join("data", "jobs_processed.csv")


def _detect(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of {names} found")


def build_bm25_index(df: pd.DataFrame | None = None) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25 index over job texts (title + description)."""
    if df is None:
        df = pd.read_csv(JOBS_CSV)
    title_col = _detect(df, ["title", "job_title", "Job_Ttl"])
    desc_col = _detect(df, ["description", "job_description", "Job_Desc"])

    job_texts = (df[title_col].fillna("") + ". " + df[desc_col].fillna("")).tolist()
    tokenized = [text.lower().split() for text in job_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, job_texts


def hybrid_rank(
    query_text: str,
    ann_indices: np.ndarray,
    semantic_scores: np.ndarray,
    bm25: BM25Okapi,
    alpha: float = 0.6,
) -> list[tuple[int, float]]:
    """Re-rank FAISS results using a semantic + BM25 hybrid score.

    Parameters
    ----------
    query_text : str
        The raw text query used for BM25 scoring.
    ann_indices : np.ndarray
        Job indices returned by FAISS (may contain -1 for missing).
    semantic_scores : np.ndarray
        Cosine similarity scores from FAISS.
    bm25 : BM25Okapi
        Pre-built BM25 index.
    alpha : float
        Blending weight — 0.6 means 60 % semantic, 40 % keyword.

    Returns
    -------
    list[tuple[int, float]]
        (job_index, hybrid_score) pairs sorted descending by score.
    """
    query_tokens = query_text.lower().split()
    full_bm25 = np.array(bm25.get_scores(query_tokens), dtype=np.float64)

    # Filter out invalid FAISS results (-1 index)
    valid = [(i, s) for i, s in zip(ann_indices, semantic_scores) if i != -1]
    if not valid:
        return []

    indices, sem_scores = zip(*valid)
    indices = np.array(indices)
    sem_scores = np.array(sem_scores, dtype=np.float64)
    bm25_scores = full_bm25[indices]

    def normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.ones_like(arr)

    sem_norm = normalize(sem_scores)
    bm25_norm = normalize(bm25_scores)
    hybrid = alpha * sem_norm + (1 - alpha) * bm25_norm

    results = [(int(idx), float(score)) for idx, score in zip(indices, hybrid)]
    return sorted(results, key=lambda x: -x[1])
