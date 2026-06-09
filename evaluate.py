#!/usr/bin/env python3
"""Offline evaluation — compute NDCG@K and Precision@K.

Ground truth is constructed via a skill-overlap heuristic: a job is
considered "relevant" to a candidate if the job description mentions
at least 2 of the candidate's listed skills (case-insensitive substring
match).

Only the held-out 20 % of candidates are evaluated to prevent
overfitting to the synthetic data.

Usage:
    python evaluate.py

Outputs:
    metrics.json — {ndcg_5, precision_5, candidates_evaluated}
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer

from cold_start import get_query_vector
from rerank import build_bm25_index, hybrid_rank
from retrieval import load_index

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOBS_CSV = os.path.join("data", "jobs_processed.csv")
CANDIDATES_JSON = os.path.join("data", "candidates.json")
JOB_EMBEDDINGS = os.path.join("embeddings", "job_embeddings.npy")
METRICS_OUT = "metrics.json"


def create_ground_truth(
    candidates: list[dict],
    descriptions_lower: list[str],
    top_n: int = 20,
) -> dict[int, set[int]]:
    """Build relevance sets using a skill-overlap heuristic.

    A job is relevant if the description contains ≥ 2 of the candidate's
    skills (case-insensitive substring matching).
    """
    gt: dict[int, set[int]] = {}
    for c in candidates:
        skills = [s.lower() for s in c.get("skills", [])]
        matches = []
        for idx, desc in enumerate(descriptions_lower):
            count = sum(1 for s in skills if s in desc)
            if count >= 2:
                matches.append((idx, count))
        matches.sort(key=lambda x: -x[1])
        gt[c["id"]] = {i for i, _ in matches[:top_n]}
    return gt


def precision_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Fraction of top-*k* results that are relevant."""
    return sum(1 for r in retrieved[:k] if r in relevant) / k if k > 0 else 0.0


def main():
    # --- Load resources ----------------------------------------------------
    df = pd.read_csv(JOBS_CSV)
    desc_col = next(
        c for c in ["description", "job_description", "Job_Desc"] if c in df.columns
    )
    with open(CANDIDATES_JSON) as f:
        all_candidates = json.load(f)

    descriptions_lower = df[desc_col].fillna("").str.lower().tolist()
    job_embeddings = np.load(JOB_EMBEDDINGS)
    index = load_index()
    bm25, _ = build_bm25_index(df)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Train/test split --------------------------------------------------
    split = int(len(all_candidates) * 0.8)
    test_candidates = all_candidates[split:]
    print(f"Total candidates:  {len(all_candidates)}")
    print(f"Test candidates:   {len(test_candidates)}")

    # --- Ground truth ------------------------------------------------------
    print("\nBuilding ground truth (skill-overlap heuristic)...")
    gt = create_ground_truth(test_candidates, descriptions_lower)
    gt_with_rel = {k: v for k, v in gt.items() if v}
    print(f"Candidates with ≥1 relevant job: {len(gt_with_rel)}/{len(test_candidates)}")

    # --- Evaluate ----------------------------------------------------------
    print(f"\nRunning evaluation (k=5)...\n")
    ndcg_list: list[float] = []
    prec_list: list[float] = []
    total = len(test_candidates)

    for i, c in enumerate(test_candidates):
        relevant = gt.get(c["id"], set())
        if not relevant:
            continue

        query_text = (
            c["bio"] + " " + " ".join(c["skills"]) + " " + c.get("desired_role", "")
        )
        query_vec = get_query_vector(query_text, model)

        D, I = index.search(query_vec.reshape(1, -1), 20)
        ranked = hybrid_rank(query_text, I[0], D[0], bm25, alpha=0.6)
        retrieved = [idx for idx, _ in ranked[:5]]

        prec = precision_at_k(retrieved, relevant, 5)
        prec_list.append(prec)

        if retrieved:
            true_rel = np.array(
                [[1.0 if idx in relevant else 0.0 for idx in retrieved]]
            )
            pred_scores = np.array(
                [[float(5 - j) for j in range(len(retrieved))]]
            )
            try:
                ndcg = float(ndcg_score(true_rel, pred_scores, k=5))
            except Exception:
                ndcg = 0.0
            ndcg_list.append(ndcg)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{total} candidates...")

    mean_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    mean_prec = np.mean(prec_list) if prec_list else 0.0

    print(f"\n{'=' * 55}")
    print(f"   EVALUATION RESULTS (k=5)")
    print(f"{'=' * 55}")
    print(f"  Candidates evaluated : {len(prec_list)}")
    print(f"  Mean NDCG@5          : {mean_ndcg:.4f}")
    print(f"  Mean Precision@5     : {mean_prec:.4f}")
    print(f"{'=' * 55}")

    metrics = {
        "ndcg_5": round(float(mean_ndcg), 4),
        "precision_5": round(float(mean_prec), 4),
        "candidates_evaluated": len(prec_list),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_OUT}")


if __name__ == "__main__":
    main()
