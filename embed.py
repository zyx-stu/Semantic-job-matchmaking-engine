#!/usr/bin/env python3
"""Embedding pipeline — encode job descriptions and candidate profiles.

Uses the `all-MiniLM-L6-v2` sentence-transformer to produce 384-dimensional
normalized dense vectors.  Normalization ensures that inner-product search
in FAISS is equivalent to cosine similarity.

Usage:
    python embed.py

Outputs:
    embeddings/job_embeddings.npy        — shape (N_jobs, 384)
    embeddings/candidate_embeddings.npy  — shape (N_candidates, 384)
"""

import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
JOBS_CSV = os.path.join("data", "jobs_processed.csv")
CANDIDATES_JSON = os.path.join("data", "candidates.json")
EMBEDDINGS_DIR = "embeddings"


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # --- Load model --------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded — {model.get_sentence_embedding_dimension()}-d embeddings\n")

    # --- Encode jobs -------------------------------------------------------
    df = pd.read_csv(JOBS_CSV)
    title_col = detect_column(df, ["title", "job_title", "Job_Ttl"])
    desc_col = detect_column(df, ["description", "job_description", "Job_Desc"])

    job_texts = (
        df[title_col].fillna("") + ". " + df[desc_col].fillna("")
    ).tolist()

    print(f"Encoding {len(job_texts)} jobs...")
    job_embeddings = model.encode(
        job_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    out_jobs = os.path.join(EMBEDDINGS_DIR, "job_embeddings.npy")
    np.save(out_jobs, job_embeddings)
    print(f"  → {out_jobs}  shape={job_embeddings.shape}\n")

    # --- Encode candidates -------------------------------------------------
    with open(CANDIDATES_JSON) as f:
        candidates = json.load(f)

    candidate_texts = [
        c["bio"] + " " + " ".join(c["skills"]) + " " + c.get("desired_role", "")
        for c in candidates
    ]

    print(f"Encoding {len(candidate_texts)} candidates...")
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    out_cand = os.path.join(EMBEDDINGS_DIR, "candidate_embeddings.npy")
    np.save(out_cand, candidate_embeddings)
    print(f"  → {out_cand}  shape={candidate_embeddings.shape}\n")

    print("All embeddings saved!")


if __name__ == "__main__":
    main()
