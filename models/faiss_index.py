"""FAISS index wrapper for exact and approximate nearest-neighbor search."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - dependency-specific behavior
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


class FaissIndex:
    def __init__(self, dim: int = 2048, index_type: str = "flat_ip") -> None:
        self._require_faiss()
        self.dim = dim
        self.index_type = index_type.lower()

        if self.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_type == "hnsw":
            try:
                self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            except TypeError:
                self.index = faiss.IndexHNSWFlat(dim, 32)
                try:
                    self.index.metric_type = faiss.METRIC_INNER_PRODUCT
                except Exception:
                    pass
            self.index.hnsw.efSearch = 64
            try:
                self.index.hnsw.efConstruction = 40
            except Exception:
                pass
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = 100
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported index type: {0}".format(index_type))

        self.image_paths: List[str] = []

    @staticmethod
    def _require_faiss():
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for similarity search. Install it with 'pip install faiss-cpu'."
            ) from _FAISS_IMPORT_ERROR
        return faiss

    def train_if_needed(self, embeddings: np.ndarray) -> None:
        """Train IVF-based indexes before adding vectors."""
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            nlist = int(getattr(self.index, "nlist", 0))
            if nlist and embeddings.shape[0] < nlist:
                raise ValueError(
                    "Not enough embeddings to train IVF index: need at least {0}, got {1}.".format(
                        nlist, embeddings.shape[0]
                    )
                )
            print("[FaissIndex] Training index...")
            self.index.train(embeddings.astype("float32"))

    def add(self, embeddings: np.ndarray, paths: List[str]) -> None:
        """Add a batch of embeddings and their corresponding image paths."""
        array = np.asarray(embeddings, dtype="float32")

        if array.ndim != 2:
            raise ValueError("Embeddings must have shape (N, dim).")
        if array.shape[1] != self.dim:
            raise ValueError(
                "Embedding dim mismatch: got {0}, expected {1}.".format(array.shape[1], self.dim)
            )
        if len(paths) != array.shape[0]:
            raise ValueError(
                "Path count mismatch: got {0} paths for {1} embeddings.".format(len(paths), array.shape[0])
            )

        self.train_if_needed(array)
        self.index.add(np.ascontiguousarray(array))
        self.image_paths.extend(str(Path(path)) for path in paths)

    def _distances_to_similarity(self, distances: np.ndarray) -> np.ndarray:
        """Convert FAISS scores into cosine-like similarity values."""
        metric_type = getattr(self.index, "metric_type", None)
        if faiss is not None and metric_type == faiss.METRIC_L2:
            similarities = 1.0 - (distances / 2.0)
        else:
            similarities = distances
        return np.clip(similarities, -1.0, 1.0)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for the nearest gallery images."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if len(self.image_paths) == 0 or getattr(self.index, "ntotal", 0) == 0:
            raise RuntimeError("The FAISS index is empty. Build it before searching.")

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim != 2:
            raise ValueError("Query embedding must have shape (1, dim).")
        if query.shape[1] != self.dim:
            raise ValueError(
                "Query dim mismatch: got {0}, expected {1}.".format(query.shape[1], self.dim)
            )

        distances, indices = self.index.search(np.ascontiguousarray(query), top_k)
        similarities = self._distances_to_similarity(distances)

        results: List[Dict[str, Any]] = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
            if idx == -1:
                continue
            results.append(
                {
                    "rank": rank,
                    "image_path": self.image_paths[int(idx)],
                    "similarity": float(similarity),
                    "index_id": int(idx),
                }
            )
        return results

    def save(self, save_dir: str) -> None:
        """Persist the FAISS index and path mapping to disk."""
        self._require_faiss()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path / "faiss.index"))
        path_array = np.array(self.image_paths, dtype=object)
        np.save(save_path / "image_paths.npy", path_array)
        np.save(save_path / "image_paths_indexed.npy", path_array)
        print("[FaissIndex] Saved index ({0} vectors) to {1}".format(len(self.image_paths), save_path))

    @classmethod
    def load(cls, save_dir: str, dim: Optional[int] = None) -> "FaissIndex":
        """Load a saved index without rebuilding it."""
        faiss_module = cls._require_faiss()
        save_path = Path(save_dir)
        index_file = save_path / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError("Missing FAISS index file: {0}".format(index_file))

        loaded_index = faiss_module.read_index(str(index_file))
        loaded_dim = int(getattr(loaded_index, "d", 0))
        if dim is not None and loaded_dim and dim != loaded_dim:
            raise ValueError(
                "Loaded index dim mismatch: file has {0}, requested {1}.".format(loaded_dim, dim)
            )

        paths_file = save_path / "image_paths_indexed.npy"
        if not paths_file.exists():
            paths_file = save_path / "image_paths.npy"
        if not paths_file.exists():
            raise FileNotFoundError("Missing image path mapping in {0}.".format(save_path))

        obj = cls.__new__(cls)
        obj.dim = loaded_dim or int(dim or 0)
        obj.index = loaded_index
        obj.index_type = type(loaded_index).__name__
        obj.image_paths = np.load(paths_file, allow_pickle=True).tolist()
        print("[FaissIndex] Loaded index ({0} vectors) from {1}".format(len(obj.image_paths), save_path))
        return obj

    def __len__(self) -> int:
        return int(getattr(self.index, "ntotal", len(self.image_paths)))
