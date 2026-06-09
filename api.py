#!/usr/bin/env python3
"""FastAPI application — semantic job recommendation endpoint.

Exposes a REST API that accepts a candidate profile and returns the
top-K most relevant jobs using the two-stage retrieval + re-ranking
pipeline:

    1. Build query vector (cold-start aware)
    2. FAISS approximate nearest-neighbour search (k=20 recall buffer)
    3. Hybrid re-ranking (semantic + BM25) → top-5 results

Usage:
    uvicorn api:app --reload
    # Then visit http://localhost:8000/docs for interactive Swagger UI
"""

import json
import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from cold_start import get_query_vector, popularity_fallback
from rerank import build_bm25_index, hybrid_rank
from retrieval import load_index

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Input schema for a job recommendation request."""

    bio: str = Field(..., min_length=5, description="Short biography / summary of the candidate.")
    skills: list[str] = Field(..., min_length=1, description="List of technical skills.")
    experience_years: int = Field(0, ge=0, description="Years of professional experience.")
    desired_role: str = Field("", description="Target job title the candidate is looking for.")
    history: list[int] = Field(
        default_factory=list,
        description="List of previously interacted job indices (for warm-start).",
    )


class JobResult(BaseModel):
    """A single recommended job."""

    rank: int
    job_index: int
    title: str
    company: str
    location: str
    score: float
    description_snippet: str


class RecommendResponse(BaseModel):
    """Response schema for /recommend."""

    strategy: str
    cold_start: bool
    results: list[JobResult]


class HealthResponse(BaseModel):
    """Response schema for /health."""

    status: str
    index_size: int
    model_name: str
    jobs_loaded: int


# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server starts."""
    print("Loading model...")
    _state["model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    _state["index"] = load_index()

    print("Loading job data...")
    _state["df"] = pd.read_csv(os.path.join("data", "jobs_processed.csv"))

    print("Loading job embeddings...")
    _state["job_embeddings"] = np.load(os.path.join("embeddings", "job_embeddings.npy"))

    print("Building BM25 index...")
    _state["bm25"], _ = build_bm25_index(_state["df"])

    # Detect column names
    df = _state["df"]
    _state["title_col"] = next(c for c in ["title", "job_title"] if c in df.columns)
    _state["company_col"] = next(
        (c for c in ["company_name", "company"] if c in df.columns), None
    )
    _state["location_col"] = next(
        (c for c in ["location", "job_location"] if c in df.columns), None
    )
    _state["desc_col"] = next(
        c for c in ["description", "job_description"] if c in df.columns
    )

    print(f"\n  Server ready — {_state['index'].ntotal} jobs indexed\n")
    yield
    _state.clear()


app = FastAPI(
    title="Semantic Job Matchmaking Engine",
    description=(
        "A two-stage semantic retrieval + re-ranking system that matches "
        "job seekers to relevant positions using dense embeddings (FAISS) "
        "and keyword scoring (BM25), with cold-start handling."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Return system health and index metadata."""
    return HealthResponse(
        status="healthy",
        index_size=_state["index"].ntotal,
        model_name="all-MiniLM-L6-v2",
        jobs_loaded=len(_state["df"]),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(profile: UserProfile):
    """Return top-5 job recommendations for the given candidate profile."""
    model = _state["model"]
    index = _state["index"]
    df = _state["df"]
    bm25 = _state["bm25"]
    job_embeddings = _state["job_embeddings"]
    title_col = _state["title_col"]
    company_col = _state["company_col"]
    location_col = _state["location_col"]
    desc_col = _state["desc_col"]

    # --- Determine strategy ------------------------------------------------
    profile_text = (
        profile.bio + " " + " ".join(profile.skills) + " " + profile.desired_role
    )

    is_cold_start = len(profile.history) == 0

    # Extreme cold-start: very sparse profile
    if len(profile.bio.split()) < 3 and len(profile.skills) < 2:
        pop_ids = popularity_fallback(df, top_k=5)
        results = []
        for rank, idx in enumerate(pop_ids, 1):
            row = df.iloc[idx]
            results.append(
                JobResult(
                    rank=rank,
                    job_index=int(idx),
                    title=str(row[title_col]),
                    company=str(row[company_col]) if company_col else "N/A",
                    location=str(row[location_col]) if location_col else "N/A",
                    score=0.0,
                    description_snippet=str(row[desc_col])[:200],
                )
            )
        return RecommendResponse(
            strategy="popularity_fallback",
            cold_start=True,
            results=results,
        )

    # Normal or warm-start
    if is_cold_start:
        strategy = "profile_only"
        query_vec = get_query_vector(profile_text, model)
    else:
        strategy = "history_augmented"
        query_vec = get_query_vector(
            profile_text, model, job_embeddings, profile.history
        )

    # --- Stage 1: FAISS retrieval (k=20 recall buffer) ---------------------
    D, I = index.search(query_vec.reshape(1, -1), 20)

    # --- Stage 2: Hybrid re-ranking → top 5 -------------------------------
    ranked = hybrid_rank(profile_text, I[0], D[0], bm25, alpha=0.6)

    results = []
    for rank, (idx, score) in enumerate(ranked[:5], 1):
        row = df.iloc[idx]
        results.append(
            JobResult(
                rank=rank,
                job_index=int(idx),
                title=str(row[title_col]),
                company=str(row[company_col]) if company_col else "N/A",
                location=str(row[location_col]) if location_col else "N/A",
                score=round(score, 4),
                description_snippet=str(row[desc_col])[:200],
            )
        )

    return RecommendResponse(
        strategy=strategy,
        cold_start=is_cold_start,
        results=results,
    )
