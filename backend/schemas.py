from typing import List

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    rank: int
    image_path: str
    similarity: float = Field(..., description="Cosine similarity score; higher means more similar")
    index_id: int


class SearchResponse(BaseModel):
    query_filename: str
    top_k: int
    gallery_size: int
    results: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    gallery_size: int
    device: str
