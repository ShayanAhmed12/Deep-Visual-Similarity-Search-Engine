"""FastAPI entry point for the visual similarity search API."""

import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.search import SimilaritySearchEngine
import backend.routes.search as search_routes

load_dotenv()


def _parse_cors_origins(raw_value: str) -> List[str]:
    value = (raw_value or "*").strip()
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retrieval engine once at startup and reuse it for all requests."""
    print("[startup] Loading search engine...")
    embeddings_dir = os.getenv("EMBEDDINGS_DIR", "data/embeddings")
    backbone = os.getenv("BACKBONE", "resnet50")
    search_routes.engine = SimilaritySearchEngine(
        embeddings_dir=embeddings_dir,
        backbone=backbone,
    )
    app.state.search_engine = search_routes.engine
    print("[startup] Search engine ready.")
    yield
    print("[shutdown] Cleaning up.")


app = FastAPI(
    title="Visual Similarity Search API",
    description="Find visually similar images using deep embeddings and FAISS nearest-neighbor search.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS", "*")),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(search_routes.router, prefix="/api/v1", tags=["search"])


@app.get("/")
async def root():
    return {"message": "Visual Similarity Search API", "docs": "/docs"}
