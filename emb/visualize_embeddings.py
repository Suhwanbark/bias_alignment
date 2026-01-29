"""
Embedding 시각화 스크립트

두 모델의 embedding을 t-SNE로 시각화하여 비교.

사용법:
    python visualize_embeddings.py \
        --emb1 data/embedding/emb_gptoss.npy \
        --emb2 data/embedding/emb_nemotron.npy \
        --labels "gpt-oss,Nemotron" \
        --output data/pic/tsne_gptoss_vs_nemotron.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def load_embeddings_and_metadata(emb_path: str, metadata_path: str = None):
    """Embedding과 메타데이터 로드"""
    embeddings = np.load(emb_path)

    if metadata_path is None:
        metadata_path = emb_path.replace('.npy', '_metadata.json')

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None

    return embeddings, metadata


def visualize_embeddings(emb1: np.ndarray, emb2: np.ndarray,
                         label1: str, label2: str,
                         metadata1: dict = None, metadata2: dict = None,
                         output_path: str = "visualization.png",
                         method: str = "tsne",
                         perplexity: int = 15,
                         random_state: int = 42):
    """Embedding 시각화 (t-SNE 또는 PCA)"""

    # 두 embedding 합치기
    all_embeddings = np.vstack([emb1, emb2])
    n1, n2 = len(emb1), len(emb2)

    print(f"Combined embeddings: {all_embeddings.shape}")
    print(f"  - {label1}: {n1}")
    print(f"  - {label2}: {n2}")

    # 차원 축소
    if method.lower() == "pca":
        print(f"\nRunning PCA...")
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(all_embeddings)
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}, Total={sum(explained_var):.3f}")
        method_label = "PCA"
        axis_labels = (f'PC1 ({explained_var[0]:.1%})', f'PC2 ({explained_var[1]:.1%})')
    else:
        print(f"\nRunning t-SNE (perplexity={perplexity})...")
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=1000)
        coords = reducer.fit_transform(all_embeddings)
        method_label = "t-SNE"
        axis_labels = ('t-SNE 1', 't-SNE 2')

    # 라벨 생성
    labels = [label1] * n1 + [label2] * n2
    colors = ['#3498db'] * n1 + ['#e74c3c'] * n2  # 파랑, 빨강

    # 티커 정보 (있으면)
    tickers = []
    if metadata1 and 'items' in metadata1:
        tickers.extend([item['ticker'] for item in metadata1['items']])
    else:
        tickers.extend([f"T{i}" for i in range(n1)])

    if metadata2 and 'items' in metadata2:
        tickers.extend([item['ticker'] for item in metadata2['items']])
    else:
        tickers.extend([f"T{i}" for i in range(n2)])

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 왼쪽: 모델별 색상
    ax1 = axes[0]
    for i, (x, y, label, color) in enumerate(zip(coords[:, 0], coords[:, 1], labels, colors)):
        ax1.scatter(x, y, c=color, alpha=0.7, s=100, edgecolors='white', linewidth=0.5)

    # 범례
    ax1.scatter([], [], c='#3498db', s=100, label=f'{label1} (n={n1})')
    ax1.scatter([], [], c='#e74c3c', s=100, label=f'{label2} (n={n2})')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_xlabel(axis_labels[0], fontsize=12)
    ax1.set_ylabel(axis_labels[1], fontsize=12)
    ax1.set_title(f'{method_label}: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 오른쪽: 티커 라벨 표시
    ax2 = axes[1]
    for i, (x, y, label, color, ticker) in enumerate(zip(coords[:, 0], coords[:, 1], labels, colors, tickers)):
        ax2.scatter(x, y, c=color, alpha=0.7, s=100, edgecolors='white', linewidth=0.5)
        ax2.annotate(ticker, (x, y), fontsize=8, alpha=0.8,
                     xytext=(3, 3), textcoords='offset points')

    ax2.scatter([], [], c='#3498db', s=100, label=f'{label1}')
    ax2.scatter([], [], c='#e74c3c', s=100, label=f'{label2}')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_xlabel(axis_labels[0], fontsize=12)
    ax2.set_ylabel(axis_labels[1], fontsize=12)
    ax2.set_title(f'{method_label} with Ticker Labels', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")

    # 통계 출력
    print("\n" + "="*50)
    print("Statistics:")
    print("="*50)

    # 각 모델의 중심점
    center1 = coords[:n1].mean(axis=0)
    center2 = coords[n1:].mean(axis=0)
    print(f"{label1} center: ({center1[0]:.2f}, {center1[1]:.2f})")
    print(f"{label2} center: ({center2[0]:.2f}, {center2[1]:.2f})")

    # 중심점 간 거리
    center_dist = np.linalg.norm(center1 - center2)
    print(f"Distance between centers: {center_dist:.2f}")

    # 각 모델의 분산
    var1 = coords[:n1].var(axis=0).sum()
    var2 = coords[n1:].var(axis=0).sum()
    print(f"{label1} variance: {var1:.2f}")
    print(f"{label2} variance: {var2:.2f}")

    return coords, labels


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings with t-SNE or PCA")
    parser.add_argument("--emb1", type=str, required=True, help="First embedding numpy file")
    parser.add_argument("--emb2", type=str, required=True, help="Second embedding numpy file")
    parser.add_argument("--labels", type=str, default="Model1,Model2", help="Labels for models (comma-separated)")
    parser.add_argument("--output", type=str, default="visualization.png", help="Output image path")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--perplexity", type=int, default=15, help="t-SNE perplexity (only for t-SNE)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    labels = args.labels.split(',')
    label1, label2 = labels[0].strip(), labels[1].strip()

    print(f"Loading embeddings...")
    emb1, meta1 = load_embeddings_and_metadata(args.emb1)
    emb2, meta2 = load_embeddings_and_metadata(args.emb2)

    print(f"  - {label1}: {emb1.shape}")
    print(f"  - {label2}: {emb2.shape}")

    visualize_embeddings(
        emb1=emb1, emb2=emb2,
        label1=label1, label2=label2,
        metadata1=meta1, metadata2=meta2,
        output_path=args.output,
        method=args.method,
        perplexity=args.perplexity,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
