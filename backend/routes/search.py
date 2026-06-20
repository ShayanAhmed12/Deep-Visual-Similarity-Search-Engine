from pathlib import Path
import io
import mimetypes
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from backend.schemas import HealthResponse, SearchResponse, SearchResult
from pipeline.search import SimilaritySearchEngine

router = APIRouter()
engine: Optional[SimilaritySearchEngine] = None


def _require_engine() -> SimilaritySearchEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized.")
    return engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    current_engine = _require_engine()
    return HealthResponse(
        status="ok",
        gallery_size=current_engine.gallery_size,
        device=current_engine.device,
    )


@router.post("/search", response_model=SearchResponse)
async def search_similar_images(
    file: UploadFile = File(..., description="Query image to search with"),
    top_k: int = Query(default=9, ge=1, le=50, description="Number of results"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image. Check file format.") from exc

    current_engine = _require_engine()
    results = current_engine.search(image, top_k=top_k)

    return SearchResponse(
        query_filename=file.filename or "unknown",
        top_k=top_k,
        gallery_size=current_engine.gallery_size,
        results=[SearchResult(**result) for result in results],
    )


@router.get("/image/{index_id}")
async def get_gallery_image(index_id: int):
    current_engine = _require_engine()
    if index_id < 0 or index_id >= len(current_engine.index.image_paths):
        raise HTTPException(status_code=404, detail="Image not found.")

    image_path = Path(current_engine.index.image_paths[index_id])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file is missing on disk.")

    media_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    return FileResponse(str(image_path), media_type=media_type)
