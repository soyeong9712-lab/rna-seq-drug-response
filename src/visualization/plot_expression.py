import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# ✅ (추가) volcano에서 t-test 사용
from scipy.stats import ttest_ind


def plot_heatmap(X, y, genes, out_path):
    """
    X: samples x genes (DataFrame)
    y: Series (index = sample)
    genes: list of gene names(columns)
    out_path: Path or str
    """
    genes = [g for g in genes if g in X.columns]
    if len(genes) == 0:
        raise ValueError("Heatmap genes list is empty after filtering columns.")

    # samples x genes -> genes x samples 로 보기 편하게 전치
    M = X[genes].T

    # z-score per gene(row)
    Mz = (M - M.mean(axis=1).values.reshape(-1, 1)) / (M.std(axis=1).replace(0, 1).values.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Mz.values, aspect="auto")

    ax.set_yticks(range(len(Mz.index)))
    ax.set_yticklabels(Mz.index, fontsize=9)

    ax.set_xticks(range(len(Mz.columns)))
    ax.set_xticklabels(Mz.columns, rotation=90, fontsize=8)

    ax.set_xlabel("sample")
    ax.set_ylabel("gene")
    ax.set_title("Heatmap (z-score) - top genes")

    # 상단에 라벨 색 띠(0/1)
    # y가 X.index와 정렬돼 있어야 함
    y_aligned = y.loc[X.index]
    # 0=blue, 1=red
    colors = np.array([[0.2, 0.4, 0.9] if v == 0 else [0.9, 0.2, 0.2] for v in y_aligned.values])
    # 얇은 띠를 그리기 위한 축 추가
    ax2 = fig.add_axes([0.125, 0.92, 0.775, 0.03])
    ax2.imshow(colors.reshape(1, -1, 3), aspect="auto")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("y (0=blue, 1=red)", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pca(X, y, out_path):
    """
    (기존 호환용) PCA scatter만 저장
    """
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X.values)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_vals = y.loc[X.index].values

    for cls in sorted(np.unique(y_vals)):
        mask = (y_vals == cls)
        ax.scatter(Xp[mask, 0], Xp[mask, 1], label=str(cls))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of RNA-seq expression")
    ax.legend(title="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pca_with_variance(X, y, out_path):
    """
    ✅ [추가] PCA + explained variance(PC1/PC2 %)까지 제목에 표시
    """
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X.values)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    y_vals = y.loc[X.index].values

    for cls in sorted(np.unique(y_vals)):
        mask = (y_vals == cls)
        ax.scatter(Xp[mask, 0], Xp[mask, 1], label=str(cls))

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title("PCA of RNA-seq expression")
    ax.legend(title="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sample_correlation_heatmap(X, y, out_path):
    """
    ✅ [추가] Sample-to-sample correlation heatmap
    """
    # 샘플 상관계수 (samples x samples)
    corr = np.corrcoef(X.values)
    samples = list(X.index)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=90, fontsize=8)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples, fontsize=8)

    ax.set_title("Sample-to-sample correlation (Pearson)")

    # 라벨 색 표기(축 옆에 간단한 점으로 표현)
    y_vals = y.loc[X.index].values
    for i, v in enumerate(y_vals):
        ax.text(len(samples)-0.3, i, "●", color=("red" if v == 1 else "blue"), va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_volcano(X, y, out_path, log2fc_thr=1.0, p_thr=0.05):
    """
    ✅ [추가] Volcano plot
    - log2FC: treated(mean) - control(mean)
    - p-value: two-sample t-test (샘플 적어서 참고용)
    """
    y_aligned = y.loc[X.index]

    # control=0, treated=1 가정
    X0 = X.loc[y_aligned == 0]
    X1 = X.loc[y_aligned == 1]

    # log2FC
    log2fc = (X1.mean(axis=0) - X0.mean(axis=0))

    # p-value (gene-wise t-test)
    pvals = []
    for g in X.columns:
        a = X1[g].values
        b = X0[g].values
        # equal_var=False로 완화
        stat, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        pvals.append(p if np.isfinite(p) else 1.0)
    pvals = np.array(pvals)

    # volcano 축
    x = log2fc.values
    yv = -np.log10(np.clip(pvals, 1e-300, 1.0))

    sig = (np.abs(x) >= log2fc_thr) & (pvals <= p_thr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[~sig], yv[~sig], s=12, alpha=0.7)
    ax.scatter(x[sig], yv[sig], s=14, alpha=0.9)

    ax.axvline(+log2fc_thr, linestyle="--", linewidth=1)
    ax.axvline(-log2fc_thr, linestyle="--", linewidth=1)
    ax.axhline(-np.log10(p_thr), linestyle="--", linewidth=1)

    ax.set_xlabel("log2 Fold Change (treated - control)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano plot (t-test, exploratory)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_topgene_boxplots(X, y, genes, out_path):
    """
    ✅ [추가] Top gene boxplot (Control vs Treated)
    """
    genes = [g for g in genes if g in X.columns]
    if len(genes) == 0:
        raise ValueError("Boxplot genes list is empty after filtering columns.")

    y_aligned = y.loc[X.index]

    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, genes):
        data0 = X.loc[y_aligned == 0, g].values
        data1 = X.loc[y_aligned == 1, g].values
        ax.boxplot([data0, data1], labels=["Control(0)", "Treated(1)"])
        ax.set_title(g, fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle("Top genes expression (boxplot)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
