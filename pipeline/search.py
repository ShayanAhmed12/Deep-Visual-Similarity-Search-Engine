"""Runtime search engine used by the API and CLI tools."""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from models.embedding_model import load_model
from models.faiss_index import FaissIndex
from pipeline.preprocess import get_transform, load_image


class SimilaritySearchEngine:
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        backbone: str = "resnet50",
        device: str = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.model = load_model(backbone, self.device)
        self.transform = get_transform()
        self.index = FaissIndex.load(embeddings_dir)

        if self.index.dim != self.model.embedding_dim:
            raise ValueError(
                "Model/index dim mismatch: backbone '{0}' produces {1} dims but the index stores {2}.".format(
                    backbone, self.model.embedding_dim, self.index.dim
                )
            )

        print("[SearchEngine] Ready | {0} images indexed | device={1}".format(len(self.index), self.device))

    def embed(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """Convert a single image into a normalized embedding."""
        if isinstance(image, (str, Path)):
            image = load_image(image)
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model(tensor).cpu().numpy().astype("float32")
        return embedding

    def search(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """Return the nearest gallery images for a query image."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        fetch_k = top_k + 1 if exclude_self else top_k
        query_embedding = self.embed(image)
        results = self.index.search(query_embedding, top_k=fetch_k)

        if exclude_self:
            query_path = None
            if isinstance(image, (str, Path)):
                query_path = str(Path(image).resolve())

            filtered = []
            for result in results:
                if query_path is not None and Path(result["image_path"]).resolve() == Path(query_path):
                    continue
                if query_path is None and result["similarity"] >= 0.9999:
                    continue
                filtered.append(result)
            results = filtered

        for rank, result in enumerate(results[:top_k], start=1):
            result["rank"] = rank

        return results[:top_k]

    @property
    def gallery_size(self) -> int:
        return len(self.index)
