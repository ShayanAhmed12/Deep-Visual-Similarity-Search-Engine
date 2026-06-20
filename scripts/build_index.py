"""Build a FAISS index from saved gallery embeddings."""

import argparse
from pathlib import Path

import numpy as np

from models.faiss_index import FaissIndex


def build_index(embeddings_dir: str = "data/embeddings", index_type: str = "flat_ip") -> None:
    embeddings_path = Path(embeddings_dir) / "embeddings.npy"
    paths_path = Path(embeddings_dir) / "image_paths_indexed.npy"
    if not paths_path.exists():
        paths_path = Path(embeddings_dir) / "image_paths.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not paths_path.exists():
        raise FileNotFoundError(f"Missing image path file: {paths_path}")

    print(f"[build_index] Loading embeddings from {embeddings_dir}...")
    embeddings = np.load(embeddings_path)
    paths = np.load(paths_path, allow_pickle=True).tolist()

    print(
        f"[build_index] Building {index_type.upper()} index | N={len(paths)} | dim={embeddings.shape[1]}"
    )
    index = FaissIndex(dim=embeddings.shape[1], index_type=index_type)
    index.add(embeddings, paths)
    index.save(embeddings_dir)

    print(f"[build_index] Index built with {len(index)} vectors. Ready for search.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", default="data/embeddings")
    parser.add_argument("--index_type", default="flat_ip", choices=["flat_ip", "hnsw", "ivf"])
    args = parser.parse_args()
    build_index(args.embeddings_dir, args.index_type)
