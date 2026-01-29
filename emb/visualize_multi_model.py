"""
Multi-Model Embedding 시각화 스크립트

4개 모델의 embedding을 버전별로 PCA로 시각화하여 비교.

사용법:
    python visualize_multi_model.py
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


MODELS = {
    "gptoss": {"label": "gpt-oss-20b", "color": "#3498db"},
    "nvidia": {"label": "Nemotron", "color": "#e74c3c"},
    "llama-4-maverick": {"label": "Llama4-Maverick", "color": "#2ecc71"},
    "gpt-5.2": {"label": "GPT-5.2", "color": "#9b59b6"},
}

VERSIONS = ["v1", "v2", "v3"]

VERSION_TITLES = {
    "v1": "v1: Generate an investment scenario in 3-4 sentences",
    "v2": "v2: Include thesis, catalyst, and risk",
    "v3": "v3: Key risks and opportunities (balanced, 5 sentences)",
}


def load_emb(model_key, version, emb_dir="emb/data/embedding"):
    path = os.path.join(emb_dir, f"emb_{model_key}_{version}.npy")
    if not os.path.exists(path):
        return None, None
    emb = np.load(path)
    meta_path = path.replace(".npy", "_metadata.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return emb, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", default="emb/data/embedding")
    parser.add_argument("--output-dir", default="emb/data/pic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ver in VERSIONS:
        print(f"\n{'='*60}")
        print(f"Version: {ver}")
        print(f"{'='*60}")

        embeddings_list = []
        labels_list = []
        colors_list = []
        model_labels = []
        model_counts = []

        for key, info in MODELS.items():
            emb, meta = load_emb(key, ver, args.emb_dir)
            if emb is None:
                print(f"  [SKIP] {info['label']}: embedding not found")
                continue
            print(f"  {info['label']}: {emb.shape}")
            embeddings_list.append(emb)
            labels_list.extend([info["label"]] * len(emb))
            colors_list.extend([info["color"]] * len(emb))
            model_labels.append(info["label"])
            model_counts.append(len(emb))

        if len(embeddings_list) < 2:
            print("  Not enough models, skipping")
            continue

        all_emb = np.vstack(embeddings_list)
        print(f"  Combined: {all_emb.shape}")

        # PCA
        pca = PCA(n_components=2, random_state=args.seed)
        coords = pca.fit_transform(all_emb)
        ev = pca.explained_variance_ratio_

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        offset = 0
        for i, (mlabel, mcount) in enumerate(zip(model_labels, model_counts)):
            c = list(MODELS.values())[list(MODELS.keys()).index(
                [k for k, v in MODELS.items() if v["label"] == mlabel][0]
            )]["color"]
            ax.scatter(
                coords[offset:offset+mcount, 0],
                coords[offset:offset+mcount, 1],
                c=c, alpha=0.5, s=40, edgecolors="white", linewidth=0.3,
                label=f"{mlabel} (n={mcount})"
            )
            offset += mcount

        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=12)
        ax.set_title(f"PCA: 4-Model Comparison ({ver})", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # subtitle
        fig.text(0.5, 0.01, VERSION_TITLES.get(ver, ver), ha="center", fontsize=9, color="gray")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out_path = os.path.join(args.output_dir, f"pca_4model_{ver}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

        # Stats
        offset = 0
        centers = []
        for mlabel, mcount in zip(model_labels, model_counts):
            center = coords[offset:offset+mcount].mean(axis=0)
            var = coords[offset:offset+mcount].var(axis=0).sum()
            centers.append((mlabel, center))
            print(f"  {mlabel}: center=({center[0]:.2f}, {center[1]:.2f}), var={var:.2f}")
            offset += mcount

        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i][1] - centers[j][1])
                print(f"  Distance {centers[i][0]} ↔ {centers[j][0]}: {dist:.2f}")

    # Combined figure (3 versions side by side)
    print(f"\n{'='*60}")
    print("Creating combined figure...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for vi, ver in enumerate(VERSIONS):
        ax = axes[vi]
        embeddings_list = []
        model_labels = []
        model_counts = []

        for key, info in MODELS.items():
            emb, _ = load_emb(key, ver, args.emb_dir)
            if emb is None:
                continue
            embeddings_list.append(emb)
            model_labels.append(info["label"])
            model_counts.append(len(emb))

        if len(embeddings_list) < 2:
            continue

        all_emb = np.vstack(embeddings_list)
        pca = PCA(n_components=2, random_state=args.seed)
        coords = pca.fit_transform(all_emb)
        ev = pca.explained_variance_ratio_

        offset = 0
        for mlabel, mcount in zip(model_labels, model_counts):
            c = list(MODELS.values())[list(MODELS.keys()).index(
                [k for k, v in MODELS.items() if v["label"] == mlabel][0]
            )]["color"]
            ax.scatter(
                coords[offset:offset+mcount, 0],
                coords[offset:offset+mcount, 1],
                c=c, alpha=0.5, s=30, edgecolors="white", linewidth=0.3,
                label=f"{mlabel} (n={mcount})"
            )
            offset += mcount

        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=11)
        ax.set_title(f"{ver}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("PCA: 4-Model Comparison by Prompt Version", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "pca_4model_combined.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
