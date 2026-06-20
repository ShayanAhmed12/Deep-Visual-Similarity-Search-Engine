"""Offline gallery embedding job."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from models.embedding_model import load_model
from pipeline.preprocess import get_transform, load_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_image_paths(image_dir: str) -> List[Path]:
    """Collect gallery images in a deterministic order."""
    base_dir = Path(image_dir)
    paths = [path for path in base_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    paths.sort()
    return paths


def embed_gallery(
    image_dir: str,
    output_dir: str,
    backbone: str = "resnet50",
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """Embed every gallery image and save the resulting arrays."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[embed_gallery] Device: {0} | Backbone: {1} | Batch: {2}".format(device, backbone, batch_size))

    model = load_model(backbone, device)
    transform = get_transform()

    image_paths = collect_image_paths(image_dir)
    print("[embed_gallery] Found {0} images in {1}".format(len(image_paths), image_dir))
    if not image_paths:
        raise ValueError("No supported images found in {0}.".format(image_dir))

    all_embeddings = []
    valid_paths: List[str] = []
    failed = 0

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Embedding batches"):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        batch_valid_paths: List[str] = []

        for path in batch_paths:
            try:
                image = load_image(path)
                tensors.append(transform(image))
                batch_valid_paths.append(str(path.resolve()))
            except Exception as exc:  # pragma: no cover - image-dependent failure path
                print("\n[WARN] Skipping {0}: {1}".format(path, exc))
                failed += 1

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.inference_mode():
            embeddings = model(batch_tensor).cpu().numpy()

        all_embeddings.append(embeddings)
        valid_paths.extend(batch_valid_paths)

    if not all_embeddings:
        raise ValueError("No valid embeddings were produced from {0}.".format(image_dir))

    embeddings_array = np.vstack(all_embeddings).astype("float32")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "embeddings.npy", embeddings_array)
    np.save(output_path / "image_paths.npy", np.array(valid_paths, dtype=object))
    np.save(output_path / "image_paths_indexed.npy", np.array(valid_paths, dtype=object))

    print(
        "\n[embed_gallery] Done. Embedded {0} images | Failed {1} | Shape: {2} | Saved to {3}".format(
            len(valid_paths), failed, embeddings_array.shape, output_path
        )
    )
    return embeddings_array, valid_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed gallery images")
    parser.add_argument("--image_dir", default="data/gallery")
    parser.add_argument("--output_dir", default="data/embeddings")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    embed_gallery(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )
