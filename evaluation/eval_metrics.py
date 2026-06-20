"""Retrieval evaluation helpers for class-labeled galleries."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline.search import SimilaritySearchEngine


def get_class_label(path: str) -> str:
    return Path(path).parent.name


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def average_precision(relevant: set, retrieved: list) -> float:
    hits = 0
    total_precision = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total_precision += hits / rank
    return total_precision / len(relevant) if relevant else 0.0


def evaluate(
    engine: SimilaritySearchEngine,
    gallery_dir: str,
    k_values: List[int] = None,
    num_queries: int = 200,
) -> Dict:
    if k_values is None:
        k_values = [1, 5, 10]

    gallery_path = Path(gallery_dir)
    all_paths = [
        str(path)
        for path in gallery_path.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not all_paths:
        raise ValueError("No evaluation images were found in {0}.".format(gallery_dir))

    rng = np.random.default_rng(42)
    query_paths = rng.choice(all_paths, size=min(num_queries, len(all_paths)), replace=False)

    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ap_scores = []

    for query_path in tqdm(query_paths, desc="Evaluating"):
        query_label = get_class_label(query_path)
        relevant = {
            path
            for path in all_paths
            if get_class_label(path) == query_label and path != query_path
        }

        if not relevant:
            continue

        image = Image.open(query_path).convert("RGB")
        results = engine.search(image, top_k=max(k_values))
        retrieved = [result["image_path"] for result in results]

        for k in k_values:
            precision_scores[k].append(precision_at_k(relevant, retrieved, k))
            recall_scores[k].append(recall_at_k(relevant, retrieved, k))
        ap_scores.append(average_precision(relevant, retrieved))

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k in k_values:
        print("  P@{0:2d}: {1:.4f}   R@{0:2d}: {2:.4f}".format(k, np.mean(precision_scores[k]), np.mean(recall_scores[k])))
    print("  mAP:   {0:.4f}".format(np.mean(ap_scores)))
    print("=" * 50)

    return {
        "precision": {k: float(np.mean(values)) for k, values in precision_scores.items()},
        "recall": {k: float(np.mean(values)) for k, values in recall_scores.items()},
        "mAP": float(np.mean(ap_scores)),
    }


if __name__ == "__main__":
    engine = SimilaritySearchEngine()
    evaluate(engine, gallery_dir="data/gallery", k_values=[1, 5, 10], num_queries=200)
