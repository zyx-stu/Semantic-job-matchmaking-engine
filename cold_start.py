#!/usr/bin/env python3
"""Cold-start query-vector strategies.

Handles three user scenarios:

1. **Cold-start (new user)** — No interaction history. The query vector is
   built purely from the user's profile text (bio + skills + desired role).

2. **Warm user (returning)** — Has previously interacted with jobs. The
   query vector blends the profile embedding with the mean of historical
   job embeddings (50/50), then re-normalizes.

3. **Extreme cold-start (sparse profile)** — Profile is too sparse to embed
   meaningfully. Falls back to a popularity-based ranking using the
   ``applies`` or ``views`` column, or simply returns the first *k* jobs.

Usage (module):
    from cold_start import get_query_vector, popularity_fallback
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_query_vector(
    profile_text: str,
    model: SentenceTransformer,
    job_embeddings: np.ndarray | None = None,
    history_job_ids: list[int] | None = None,
) -> np.ndarray:
    """Build a query vector from a user profile ± interaction history.

    Parameters
    ----------
    profile_text : str
        Concatenation of the candidate's bio, skills, and desired role.
    model : SentenceTransformer
        Loaded sentence-transformer used for encoding.
    job_embeddings : np.ndarray, optional
        Full job-embedding matrix (needed when *history_job_ids* is provided).
    history_job_ids : list[int], optional
        Indices of jobs the user previously interacted with.

    Returns
    -------
    np.ndarray
        A unit-norm float32 vector suitable for FAISS inner-product search.
    """
    profile_emb = model.encode([profile_text], normalize_embeddings=True)[0]

    if not history_job_ids:
        return profile_emb.astype(np.float32)

    # Blend: 50 % profile + 50 % mean(history)
    history_embs = np.array([job_embeddings[i] for i in history_job_ids])
    blended = 0.5 * profile_emb + 0.5 * history_embs.mean(axis=0)

    # Re-normalize so inner product ≡ cosine
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    return blended.astype(np.float32)


def popularity_fallback(
    df: pd.DataFrame,
    top_k: int = 5,
) -> list[int]:
    """Return the *top_k* most popular jobs when the profile is too sparse.

    Popularity is determined by the ``applies`` column if present, falling
    back to ``views``, and finally to row order (recency proxy).
    """
    if "applies" in df.columns:
        return df.nlargest(top_k, "applies").index.tolist()
    if "views" in df.columns:
        return df.nlargest(top_k, "views").index.tolist()
    return list(range(min(top_k, len(df))))
