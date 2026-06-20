"""Visualize the embedding space using UMAP."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap


def plot_umap(embeddings_dir: str = "data/embeddings", max_points: int = 3000):
    embeddings_path = Path(embeddings_dir)
    embeddings = np.load(embeddings_path / "embeddings.npy")
    paths = np.load(embeddings_path / "image_paths.npy", allow_pickle=True)

    labels = [Path(path).parent.name for path in paths]
    unique_labels = sorted(set(labels))
    label_to_int = {label: index for index, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])

    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        int_labels = int_labels[indices]

    print("Running UMAP on {0} points...".format(len(embeddings)))
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for index, label in enumerate(unique_labels):
        mask = int_labels == index
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[index]], label=label, s=8, alpha=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax.set_title("ResNet50 Embedding Space (UMAP projection)", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    output_file = embeddings_path / "umap_embeddings.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to {0}".format(output_file))


if __name__ == "__main__":
    plot_umap()
