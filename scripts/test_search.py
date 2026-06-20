"""Simple CLI smoke test for similarity search."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.search import SimilaritySearchEngine


def run_search(
    query_image: str,
    embeddings_dir: str = "data/embeddings",
    backbone: str = "resnet50",
    top_k: int = 5,
    exclude_self: bool = True,
):
    engine = SimilaritySearchEngine(embeddings_dir=embeddings_dir, backbone=backbone)
    results = engine.search(query_image, top_k=top_k, exclude_self=exclude_self)

    print(f"Query: {query_image}")
    for result in results:
        print(f"{result['rank']:>2}. {result['similarity']:.6f} | {result['image_path']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick retrieval sanity check")
    parser.add_argument("query_image", help="Path to the query image")
    parser.add_argument("--embeddings_dir", default="data/embeddings")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--include_self", action="store_true")
    args = parser.parse_args()

    query_path = Path(args.query_image)
    if not query_path.exists():
        raise SystemExit(f"Query image not found: {query_path}")

    run_search(
        query_image=str(query_path),
        embeddings_dir=args.embeddings_dir,
        backbone=args.backbone,
        top_k=args.top_k,
        exclude_self=not args.include_self,
    )
