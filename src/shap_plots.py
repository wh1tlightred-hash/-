"""SHAP 可解释性图表：蜂群图、依赖图、瀑布图、六模型重要性热力图、折外 vs 全量对照。

全部为描述性分析：不是因果证据，也不是外部验证。颜色约定：
- 特征取值：蓝（低）↔ 灰 ↔ 红（高）发散色；
- 瀑布图贡献：红=推高该类输出，蓝=拉低该类输出；
- 类别：CLASS_COLORS（与逐类曲线一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .models import MODEL_SHORT  # noqa: E402
from .plot import (  # noqa: E402
    AXIS_COLOR,
    CLASS_COLORS,
    DIV_CMAP,
    DIVERGING,
    INK_SECONDARY,
    SEQ_CMAP,
    STYLE,
)

plt.rcParams.update(STYLE)


def _feature_order(values) -> np.ndarray:
    """按三类聚合 mean(|SHAP|) 从大到小排序的特征索引。"""
    return np.argsort(-np.mean(np.abs(values), axis=(0, 2)), kind="stable")


def _swarm_offsets(v: np.ndarray, width: float = 0.38, nbins: int = 40) -> np.ndarray:
    """蜂群纵向偏移：SHAP 值落在同一小区间的点上下交替排开，避免重叠。"""
    if np.ptp(v) == 0:
        bins = np.zeros(len(v), dtype=int)
    else:
        bins = np.floor((v - v.min()) / np.ptp(v) * (nbins - 1)).astype(int)
    offsets = np.zeros(len(v))
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        k = np.arange(len(idx))
        step = (k + 1) // 2 * np.where(k % 2 == 0, 1, -1)
        offsets[idx] = step
    mx = np.abs(offsets).max()
    return offsets / mx * width if mx > 0 else offsets


def _norm_feature(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    if hi <= lo:
        return np.full(len(x), 0.5)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def plot_shap_beeswarm(values, X, feature_names, class_names, outpath: str, title: str):
    """每类一个蜂群图：横轴 SHAP 值，纵轴特征（按聚合重要性排序），颜色为特征取值高低。"""
    values = np.asarray(values)
    X = np.asarray(X, dtype=float)
    order = _feature_order(values)[::-1]  # 最重要的在最上面
    fig, axes = plt.subplots(1, len(class_names), figsize=(5.0 * len(class_names), 5.6), sharey=True)
    for c, ax in enumerate(np.atleast_1d(axes)):
        ax.axvline(0, color=AXIS_COLOR, lw=1)
        for row, f in enumerate(order):
            v = values[:, f, c]
            ax.scatter(v, row + _swarm_offsets(v), c=_norm_feature(X[:, f]), cmap=DIV_CMAP,
                       vmin=0, vmax=1, s=22, edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(np.asarray(feature_names)[order])
        ax.set_xlabel(f"SHAP 值（对「{class_names[c]}」输出的贡献）")
        ax.set_title(f"{class_names[c]} (class {c})")
        ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=np.atleast_1d(axes).tolist(), fraction=0.02, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["低", "高"])
    cb.set_label("特征取值（5–95 分位归一化）")
    cb.outline.set_visible(False)
    fig.suptitle(title, y=1.0)
    fig.savefig(outpath)
    plt.close(fig)


def plot_shap_dependence(values, X, feature_names, class_names, outpath: str, top_k: int = 4):
    """前 top_k 个重要特征的依赖图：横轴特征取值，纵轴 SHAP 值，三类分色。"""
    values = np.asarray(values)
    X = np.asarray(X, dtype=float)
    top = _feature_order(values)[:top_k]
    ncols = min(top_k, 2) if top_k > 1 else 1
    nrows = int(np.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), top):
        ax.axhline(0, color=AXIS_COLOR, lw=1)
        for c in range(len(class_names)):
            x, v = X[:, f], values[:, f, c]
            ax.scatter(x, v, s=26, color=CLASS_COLORS[c], edgecolor="white", linewidth=0.5,
                       zorder=3, label=class_names[c])
            # 简单趋势线（线性拟合），只作方向参考
            if np.ptp(x) > 0:
                k, b = np.polyfit(x, v, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(xs, k * xs + b, color=CLASS_COLORS[c], lw=1.2, alpha=0.7)
        ax.set_xlabel(f"{feature_names[f]} 取值")
        ax.set_ylabel("SHAP 值")
        ax.set_title(f"{feature_names[f]}")
        ax.legend(fontsize=8, title="解释的输出类别", title_fontsize=8)
    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)
    fig.suptitle("SHAP 依赖图：重要特征取值与贡献的关系（细线为线性趋势，仅作方向参考）", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def pick_representative_samples(scores: np.ndarray, y: np.ndarray) -> list[int]:
    """每个真实类别挑一条：该类别得分最高的样本（「典型」样本）。"""
    picks = []
    for c in range(scores.shape[1]):
        idx = np.where(np.asarray(y) == c)[0]
        picks.append(int(idx[np.argmax(scores[idx, c])]))
    return picks


def _waterfall(ax, contrib, base, feature_names, x_row, max_display=8):
    order = np.argsort(-np.abs(contrib), kind="stable")
    shown = order[:max_display]
    rest = order[max_display:]
    items = [(f"{feature_names[f]} = {x_row[f]:.4g}", contrib[f]) for f in shown]
    if len(rest):
        items.append((f"其余 {len(rest)} 个特征", float(contrib[rest].sum())))
    items = items[::-1]  # 最大的贡献画在最上方
    # 自下而上累计：先从 base 加上「其余」，最后到 f(x)
    ends = base + np.cumsum([v for _, v in items])
    lo, hi = min(base, ends.min()), max(base, ends.max())
    span = hi - lo if hi > lo else 1.0
    pad = 0.012 * span
    cum = base
    for i, (label, v) in enumerate(items):
        color = DIVERGING[2] if v >= 0 else DIVERGING[0]
        ax.barh(i, v, left=cum, color=color, height=0.65)
        # 数值标在条的外侧端点（正值在右端之右、负值在左端之左）
        ax.text(cum + v + (pad if v >= 0 else -pad), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_SECONDARY)
        cum += v
    ax.set_xlim(lo - 0.16 * span, hi + 0.16 * span)  # 给两侧数值标签留出空间
    ax.axvline(base, color=AXIS_COLOR, lw=1, ls="--")
    ax.axvline(cum, color=INK_SECONDARY, lw=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for lab, _ in items], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return cum


def plot_shap_waterfalls(values, base_values, X, y, sample_ids, picks, feature_names, class_names,
                         outpath: str, output_space: str):
    """代表样本的 SHAP 瀑布图：从基准值 E[f(x)] 出发，各特征贡献逐项累加到该样本输出 f(x)。"""
    values = np.asarray(values)
    X = np.asarray(X, dtype=float)
    fig, axes = plt.subplots(1, len(picks), figsize=(6.2 * len(picks), 4.8))
    for ax, i in zip(np.atleast_1d(axes), picks):
        c = int(y[i])
        base = float(base_values[c])
        fx = _waterfall(ax, values[i, :, c], base, feature_names, X[i])
        ax.set_title(f"{sample_ids[i]}（真实：{class_names[c]}）\n解释「{class_names[c]}」输出："
                     f"E[f]={base:.3f} → f(x)={fx:.3f}", fontsize=10)
        ax.set_xlabel(output_space)
    fig.suptitle("SHAP 瀑布图：每个真实类别中该类得分最高的样本（红=推高，蓝=拉低；虚线为基准值）", y=1.03)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_shap_model_heatmap(table: pd.DataFrame, outpath: str, models: list[str]):
    """六模型 SHAP 重要性占比热力图（每行和为 1），格内标注排名。"""
    feats = (table.groupby("feature")["share"].mean().sort_values(ascending=False).index.tolist())
    share = table.pivot(index="model", columns="feature", values="share").loc[models, feats]
    rank = table.pivot(index="model", columns="feature", values="rank").loc[models, feats]
    fig, ax = plt.subplots(figsize=(1.0 * len(feats) + 2.6, 0.62 * len(models) + 1.8))
    im = ax.imshow(share.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=float(share.to_numpy().max()), aspect="auto")
    thr = float(share.to_numpy().max()) * 0.55
    for i in range(len(models)):
        for j in range(len(feats)):
            v = share.iat[i, j]
            ax.text(j, i, f"{v:.0%}\n#{int(rank.iat[i, j])}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > thr else "#0b0b0b")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean(|SHAP|) 占比")
    cb.outline.set_visible(False)
    ax.set_title("六模型 SHAP 重要性对比（各模型内归一化为占比；#n 为该模型内排名）")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_shap_cv_compare(full_values, cv_values, fold_ranks: pd.DataFrame, feature_names,
                         outpath: str, model_label: str):
    """左：全量模型 SHAP 与折外 SHAP 的重要性占比对照；右：逐折特征排名热力图（含全量）。"""
    full_imp = np.mean(np.abs(full_values), axis=(0, 2))
    cv_imp = np.mean(np.abs(cv_values), axis=(0, 2))
    full_share, cv_share = full_imp / full_imp.sum(), cv_imp / cv_imp.sum()
    order = np.argsort(-full_share, kind="stable")
    feats = np.asarray(feature_names)[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(feats))[::-1]
    h = 0.38
    ax1.barh(y + h / 2, full_share[order], height=h, color="#1c5cab", label="全量拟合（训练后）")
    ax1.barh(y - h / 2, cv_share[order], height=h, color="#86b6ef", label="折外（未见过该样本）")
    ax1.set_yticks(y)
    ax1.set_yticklabels(feats)
    ax1.set_xlabel("mean(|SHAP|) 占比")
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.grid(axis="y", visible=False)
    ax1.set_title("重要性占比：全量 vs 折外")

    folds = sorted(fold_ranks["fold"].unique())
    piv = fold_ranks.pivot(index="feature", columns="fold", values="rank").loc[feats, folds]
    full_rank = np.empty(len(feature_names), dtype=int)
    full_rank[order] = np.arange(1, len(feature_names) + 1)
    mat = np.column_stack([piv.to_numpy(), full_rank[order]])
    # 排名越靠前颜色越深：用 (n+1-rank) 映射到顺序色
    n = len(feature_names)
    ax2.imshow(n + 1 - mat, cmap=SEQ_CMAP, vmin=1, vmax=n, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8.5,
                     color="white" if mat[i, j] <= n * 0.4 else "#0b0b0b")
    ax2.set_xticks(range(mat.shape[1]))
    ax2.set_xticklabels([f"折 {f}" for f in folds] + ["全量"])
    ax2.set_yticks(range(len(feats)))
    ax2.set_yticklabels(feats)
    ax2.grid(False)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title("逐折特征排名（数字为名次，颜色越深越靠前）")
    fig.suptitle(f"{model_label}：SHAP 重要性的折间稳定性", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
