"""补充对比曲线（全部基于真实 OOF / 逐折记录 / 交叉验证分析结果）。

与 plot.py 共用配色与样式：颜色跟随模型（MODEL_COLORS），Dummy 基线为中性灰虚线。
每张图都带图例，报告里同时给出对应数值表（浅色槽位对比度 <3:1，不能只靠颜色）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import compute_metrics, pr_curves_per_class, roc_curves_per_class  # noqa: E402
from .models import MODEL_SHORT, REAL_MODELS, SIMPLICITY_ORDER, VERSION_SHORT  # noqa: E402
from .plot import (  # noqa: E402
    AXIS_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    MODEL_COLORS,
    STYLE,
    SURFACE,
    _GRID,
    _macro_roc,
)

plt.rcParams.update(STYLE)

ORDERED = [m for m in SIMPLICITY_ORDER if m in REAL_MODELS]
_METRIC_LABEL = {"accuracy": "Accuracy", "f1_macro": "macro-F1", "roc_auc_macro": "macro ROC-AUC",
                 "precision_macro": "macro Precision", "recall_macro": "macro Recall",
                 "mAP": "mAP", "pr_auc_macro": "macro PR-AUC"}


def _scores(df: pd.DataFrame):
    return (df["y_true"].to_numpy(dtype=int),
            df[[c for c in df.columns if c.startswith("score_")]].to_numpy(dtype=float),
            df["y_pred"].to_numpy(dtype=int))


def _grid_axes(n: int, ncols: int = 3, size=(4.4, 3.8), sharey=True):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                             sharey=sharey, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _save(fig, outpath):
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ---------------------------------------------------------------- ROC / PR 补充

def plot_fold_roc_bands(oof: dict, outpath: str, models=None):
    """每个模型（调参版）的逐折 macro ROC + 均值曲线 + ±1 标准差带。

    与「跨折拼接」曲线不同，这里每折单独画，均值与标准差反映折间变动，
    避免拼接时 SVM 决策分数等跨折尺度差异混进同一条曲线。
    """
    models = models or ORDERED
    fig, axes = _grid_axes(len(models))
    for ax, m in zip(axes, models):
        df = oof[(m, "tuned")]
        tprs, aucs = [], []
        for f, sub in df.groupby("fold"):
            yt, ys, yp = _scores(sub)
            if len(np.unique(yt)) < 3:
                continue
            fpr, tpr = _macro_roc(yt, ys)
            tprs.append(tpr)
            aucs.append(compute_metrics(yt, ys, yp)["roc_auc_macro"])
            ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=0.8, alpha=0.35)
        tprs = np.array(tprs)
        mean, std = tprs.mean(axis=0), tprs.std(axis=0)
        ax.fill_between(_GRID, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=MODEL_COLORS[m], alpha=0.18, lw=0, label="±1 标准差")
        ax.plot(_GRID, mean, color=MODEL_COLORS[m], lw=2,
                label=f"折均值（AUC {np.mean(aucs):.2f}±{np.std(aucs):.2f}）")
        ax.plot([0, 1], [0, 1], ls="--", color=AXIS_COLOR, lw=1)
        ax.set_title(MODEL_SHORT[m])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("逐折 macro ROC 与折间变动（细线为单折，带为 ±1 标准差；调参版，种子 42）", y=1.01)
    _save(fig, outpath)


def plot_tuned_vs_untuned_roc(oof: dict, oof_metrics: pd.DataFrame, outpath: str, models=None):
    """调参前后 macro ROC 对比（同一组外层切分）。实线=调参，虚线=未调参。"""
    models = models or ORDERED
    fig, axes = _grid_axes(len(models))
    for ax, m in zip(axes, models):
        for version, ls in (("untuned", "--"), ("tuned", "-")):
            yt, ys, _ = _scores(oof[(m, version)])
            fpr, tpr = _macro_roc(yt, ys)
            auc = oof_metrics[(oof_metrics["model"] == m) & (oof_metrics["version"] == version)]["roc_auc_macro"].iloc[0]
            ax.plot(fpr, tpr, ls=ls, color=MODEL_COLORS[m], label=f"{VERSION_SHORT[version]}（AUC {auc:.3f}）")
        ax.plot([0, 1], [0, 1], ls=":", color=AXIS_COLOR, lw=1)
        ax.set_title(MODEL_SHORT[m])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("调参前后 macro ROC 对比（跨折拼接 OOF；图例 AUC 为逐类 AUC 平均）", y=1.01)
    _save(fig, outpath)


def plot_per_class_model_curves(oof: dict, oom: pd.DataFrame, class_names, outpath: str):
    """按类别分面：每个类别一列，上行 ROC、下行 PR，六模型（调参）+ Dummy 同图对比。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    entries = [(m, "tuned") for m in ORDERED] + [("dummy", "untrained")]
    for m, version in entries:
        yt, ys, _ = _scores(oof[(m, version)])
        row = oom[(oom["model"] == m) & (oom["version"] == version)].iloc[0]
        roc, pr = roc_curves_per_class(yt, ys), pr_curves_per_class(yt, ys)
        style = dict(color=MODEL_COLORS[m], lw=1.0 if m == "dummy" else 1.8,
                     ls="--" if m == "dummy" else "-")
        for c in range(3):
            fpr, tpr = roc[c]
            auc_c = row["roc_auc_ovr"][c]
            axes[0, c].plot(fpr, tpr, label=f"{MODEL_SHORT[m]} {auc_c:.2f}", **style)
            prec, rec = pr[c]
            ap_c = row["ap_ovr"][c]
            axes[1, c].plot(rec, prec, label=f"{MODEL_SHORT[m]} {ap_c:.2f}", **style)
    for c in range(3):
        axes[0, c].plot([0, 1], [0, 1], ls=":", color=AXIS_COLOR, lw=1)
        axes[0, c].set_title(f"{class_names[c]} vs 其余：ROC（图例为 AUC）")
        axes[0, c].set_xlabel("FPR")
        axes[0, c].set_ylabel("TPR")
        axes[1, c].axhline(1 / 3, ls=":", color=AXIS_COLOR, lw=1)
        axes[1, c].set_title(f"{class_names[c]} vs 其余：PR（图例为 AP）")
        axes[1, c].set_xlabel("Recall")
        axes[1, c].set_ylabel("Precision")
        axes[1, c].set_ylim(0, 1.02)
        for r in range(2):
            axes[r, c].legend(fontsize=7.5, loc="lower right" if r == 0 else "upper right",
                              frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.85)
    fig.suptitle("逐类别的模型对比（OvR，跨折拼接 OOF；PR 图虚线为类别先验 1/3）", y=1.0)
    _save(fig, outpath)


# ---------------------------------------------------------------- 指标对比

def plot_fold_metric_box(folds_df: pd.DataFrame, outpath: str,
                         metrics=("accuracy", "f1_macro", "roc_auc_macro")):
    """外层 5 折逐折指标分布：每个模型未调参（浅）/调参（深）两个箱体 + 逐折散点。"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.6), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), metrics):
        pos, labels = 0, []
        for m in ORDERED + ["dummy"]:
            versions = ("untrained",) if m == "dummy" else ("untuned", "tuned")
            for v in versions:
                sub = folds_df[(folds_df["model"] == m) & (folds_df["version"] == v)]
                vals = [r["metrics"][key] for r in sub.to_dict("records") if r["metrics"][key] is not None]
                light = v == "untuned"
                bp = ax.boxplot([vals], positions=[pos], widths=0.6, patch_artist=True,
                                showfliers=False, medianprops=dict(color=INK_SECONDARY, lw=1.5))
                for b in bp["boxes"]:
                    b.set(facecolor=MODEL_COLORS[m], alpha=0.35 if light else 0.75, edgecolor=MODEL_COLORS[m])
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set(color=MODEL_COLORS[m])
                jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else [0]
                ax.scatter(pos + np.asarray(jitter), vals, s=14, color=INK_SECONDARY, zorder=3)
                pos += 1
            labels.append((pos - len(versions) / 2 - 0.5, MODEL_SHORT[m]))
            pos += 0.6
        ax.set_xticks([p for p, _ in labels])
        ax.set_xticklabels([t for _, t in labels], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(_METRIC_LABEL[key])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="x", visible=False)
    fig.suptitle("外层 5 折逐折指标分布（每个模型左浅=未调参、右深=调参；点为单折）", y=1.02)
    _save(fig, outpath)


def plot_radar(oof_metrics: pd.DataFrame, outpath: str,
               metrics=("accuracy", "precision_macro", "recall_macro", "f1_macro",
                        "roc_auc_macro", "mAP", "pr_auc_macro")):
    """多指标雷达图（调参版六模型 + Dummy 基线）。"""
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.4, 7.4), subplot_kw=dict(polar=True))
    for m in ORDERED + ["dummy"]:
        version = "untrained" if m == "dummy" else "tuned"
        row = oof_metrics[(oof_metrics["model"] == m) & (oof_metrics["version"] == version)].iloc[0]
        vals = [float(row[k]) for k in metrics]
        vals += vals[:1]
        ls = "--" if m == "dummy" else "-"
        ax.plot(angles, vals, color=MODEL_COLORS[m], ls=ls, lw=1.2 if m == "dummy" else 1.8,
                label=MODEL_SHORT[m])
        if m != "dummy":
            ax.fill(angles, vals, color=MODEL_COLORS[m], alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_METRIC_LABEL[k] for k in metrics], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=7, labelcolor=INK_MUTED)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=8.5)
    ax.set_title("多指标雷达图（调参版 OOF，种子 42）", pad=22)
    _save(fig, outpath)


def plot_seed_stability(stability: pd.DataFrame, outpath: str,
                        metrics=("accuracy", "f1_macro", "roc_auc_macro")):
    """三个切分种子下各模型（调参）OOF 指标：点=单个种子，横线=极差。"""
    seeds = sorted(stability["seed"].unique())
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.4), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), metrics):
        for i, m in enumerate(ORDERED + ["dummy"]):
            v = "untrained" if m == "dummy" else "tuned"
            sub = stability[(stability["model"] == m) & (stability["version"] == v)].set_index("seed")
            vals = [float(sub.loc[s, key]) for s in seeds]
            ax.plot([i, i], [min(vals), max(vals)], color=MODEL_COLORS[m], lw=3, alpha=0.4,
                    solid_capstyle="round")
            for j, val in enumerate(vals):
                ax.scatter(i, val, marker=markers[j % len(markers)], s=46, color=MODEL_COLORS[m],
                           edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(ORDERED) + 1))
        ax.set_xticklabels([MODEL_SHORT[m] for m in ORDERED + ["dummy"]], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(_METRIC_LABEL[key])
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D

    np.atleast_1d(axes)[0].legend(handles=[
        Line2D([], [], ls="", marker=markers[j % len(markers)], ms=7, color=INK_MUTED, label=f"种子 {s}")
        for j, s in enumerate(seeds)], fontsize=8, loc="upper left")
    fig.suptitle("切分稳定性：种子 42/43/44 的 OOF 指标（竖条为三种子极差）", y=1.02)
    _save(fig, outpath)


def plot_train_vs_oof(train_oof: pd.DataFrame, outpath: str):
    """训练分数 vs 折外分数的哑铃图（过拟合差距）。train_oof 列：model, version, train_f1_macro, oof_f1_macro。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, version in zip(axes, ("untuned", "tuned")):
        sub = train_oof[train_oof["version"] == version]
        for i, m in enumerate(ORDERED):
            r = sub[sub["model"] == m].iloc[0]
            ax.plot([r["oof_f1_macro"], r["train_f1_macro"]], [i, i], color=MODEL_COLORS[m], lw=2, alpha=0.6)
            ax.scatter(r["train_f1_macro"], i, s=60, color=MODEL_COLORS[m], marker="o", zorder=3)
            ax.scatter(r["oof_f1_macro"], i, s=60, facecolor="white", edgecolor=MODEL_COLORS[m],
                       linewidth=1.8, zorder=3)
            ax.text(1.04, i, f"差 {r['train_f1_macro'] - r['oof_f1_macro']:+.2f}", va="center",
                    fontsize=8, color=INK_SECONDARY, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(ORDERED)))
        ax.set_yticklabels([MODEL_SHORT[m] for m in ORDERED])
        ax.set_xlim(0, 1.04)
        ax.set_xlabel("macro-F1")
        ax.set_title(VERSION_SHORT[version])
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D

    axes[0].legend(handles=[
        Line2D([], [], ls="", marker="o", ms=7, color=INK_MUTED, label="训练折（实心）"),
        Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=INK_MUTED, mew=1.8, label="外层测试折（空心）"),
    ], fontsize=8, loc="lower left")
    fig.suptitle("训练分数与外层折外分数差距（逐折均值；差距越大越可能过拟合）", y=1.03)
    _save(fig, outpath)


# ---------------------------------------------------------------- 学习 / 验证 / 校准

def plot_learning_curves(lc: pd.DataFrame, outpath: str, dummy_f1: float | None = None):
    """分层学习曲线：训练子集 macro-F1（虚线）与测试折 macro-F1（实线），带 ±1 标准差。"""
    fig, axes = _grid_axes(len(ORDERED))
    for ax, m in zip(axes, ORDERED):
        g = lc[lc["model"] == m].groupby("n_train")
        n = np.array(sorted(lc["n_train"].unique()))
        for col, ls, lab in (("train_f1_macro", "--", "训练子集"), ("test_f1_macro", "-", "测试折")):
            mu = g[col].mean().reindex(n).to_numpy()
            sd = g[col].std(ddof=0).reindex(n).to_numpy()
            ax.plot(n, mu, ls=ls, marker="o", ms=4, color=MODEL_COLORS[m], label=lab)
            ax.fill_between(n, mu - sd, mu + sd, color=MODEL_COLORS[m], alpha=0.12, lw=0)
        if dummy_f1 is not None:
            ax.axhline(dummy_f1, ls=":", color=INK_MUTED, lw=1.2, label=f"Dummy {dummy_f1:.2f}")
        ax.set_title(MODEL_SHORT[m])
        ax.set_xlabel("训练样本数（每类等量）")
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(n)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("学习曲线（默认超参数；重复分层 5 折，每折内按类等量子采样）", y=1.01)
    _save(fig, outpath)


def plot_validation_curves(vc: pd.DataFrame, outpath: str):
    """单个关键超参数的验证曲线：训练（虚线）与验证（实线）macro-F1 ± 标准差。"""
    fig, axes = _grid_axes(len(ORDERED))
    for ax, m in zip(axes, ORDERED):
        sub = vc[vc["model"] == m].sort_values("order")
        x = sub["order"].to_numpy()
        for col, ls, lab in (("train", "--", "训练"), ("test", "-", "验证")):
            mu, sd = sub[f"{col}_mean"].to_numpy(), sub[f"{col}_std"].to_numpy()
            ax.plot(x, mu, ls=ls, marker="o", ms=4, color=MODEL_COLORS[m], label=lab)
            ax.fill_between(x, mu - sd, mu + sd, color=MODEL_COLORS[m], alpha=0.12, lw=0)
        best = sub.loc[sub["test_mean"].idxmax()]
        ax.scatter(best["order"], best["test_mean"], s=90, facecolor="none", edgecolor=INK_SECONDARY,
                   linewidth=1.4, zorder=4, label=f"验证最高 {best['test_mean']:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["value"].tolist(), fontsize=8, rotation=30)
        ax.set_xlabel(sub["param"].iloc[0])
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(MODEL_SHORT[m])
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("验证曲线（单参数扫描，其余默认；重复分层 5 折×3，描述性，不参与调参）", y=1.01)
    _save(fig, outpath)


def plot_calibration(cal: dict, class_names, outpath: str):
    """可靠性曲线（逐类 OvR，分位数分箱）+ Brier 分数图例。SVM 为决策分数，不在其中。"""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    table = cal["table"].set_index("model")
    for c in range(3):
        ax = axes[c]
        ax.plot([0, 1], [0, 1], ls=":", color=AXIS_COLOR, lw=1, label="完美校准")
        for m, curves in cal["curves"].items():
            mp, fp = curves[c]
            ls = "--" if m == "dummy" else "-"
            ax.plot(mp, fp, ls=ls, marker="o", ms=4, color=MODEL_COLORS[m],
                    label=f"{MODEL_SHORT[m]}（Brier {table.loc[m, 'brier']:.3f}）")
        ax.set_title(f"{class_names[c]} vs 其余")
        ax.set_xlabel("平均预测概率")
        ax.set_ylabel("实际阳性比例")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
    axes[-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("概率校准：可靠性曲线（调参版 OOF 概率；图例 Brier 为三类多分类 Brier，越低越好）", y=1.03)
    _save(fig, outpath)


# ---------------------------------------------------------------- 交叉验证方案

def plot_cv_schemes(summary: pd.DataFrame, outpath: str, metrics=("accuracy", "f1_macro")):
    """不同 CV 方案下的拼接折外指标（默认超参数）。每个方案一组，组内按模型错开。"""
    schemes = list(dict.fromkeys(summary["scheme"]))
    labels = [summary[summary["scheme"] == s]["scheme_label"].iloc[0] for s in schemes]
    models = ORDERED + ["dummy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 4.6), sharey=True)
    width = 0.8 / len(models)
    for ax, key in zip(np.atleast_1d(axes), metrics):
        for j, m in enumerate(models):
            sub = summary[summary["model"] == m].set_index("scheme")
            x = np.arange(len(schemes)) - 0.4 + width * (j + 0.5)
            vals = [float(sub.loc[s, key]) for s in schemes]
            ax.scatter(x, vals, s=36, color=MODEL_COLORS[m], marker="D" if m == "dummy" else "o",
                       edgecolor="white", linewidth=0.6, zorder=3, label=MODEL_SHORT[m])
            # 重复 CV 加 ±1 标准差（逐次重复）
            for xi, s in zip(x, schemes):
                if s.startswith("rskf") and key == "f1_macro":
                    sd = float(sub.loc[s, "unit_f1_std"])
                    ax.plot([xi, xi], [sub.loc[s, key] - sd, sub.loc[s, key] + sd],
                            color=MODEL_COLORS[m], lw=1.5)
        for k in range(1, len(schemes)):
            ax.axvline(k - 0.5, color=AXIS_COLOR, lw=0.6)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"拼接折外 {_METRIC_LABEL[key]}")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("交叉验证方案对比（默认超参数；重复 CV 为各次重复的均值，竖线为 ±1 标准差）", y=1.02)
    _save(fig, outpath)


def plot_repeated_cv_box(units: pd.DataFrame, outpath: str, scheme_prefix: str = "rskf"):
    """重复分层 5 折：每次重复的拼接 macro-F1 / Accuracy 分布（反映仅换切分带来的波动）。"""
    sub = units[units["scheme"].str.startswith(scheme_prefix)]
    label = sub["scheme_label"].iloc[0]
    models = ORDERED + ["dummy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, key in zip(axes, ("accuracy", "f1_macro")):
        data = [sub[sub["model"] == m][key].to_numpy(dtype=float) for m in models]
        bp = ax.boxplot(data, positions=range(len(models)), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK_SECONDARY, lw=1.5))
        for b, m in zip(bp["boxes"], models):
            b.set(facecolor=MODEL_COLORS[m], alpha=0.55, edgecolor=MODEL_COLORS[m])
        for i, vals in enumerate(data):
            ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=10, color=INK_SECONDARY, zorder=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(_METRIC_LABEL[key])
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{label}：每次重复的拼接折外指标分布（默认超参数）", y=1.02)
    _save(fig, outpath)


def plot_permutation(perm: dict, outpath: str):
    """置换检验：打乱标签后的 macro-F1 零分布（直方图）与真实分数（竖线）。"""
    table = perm["table"].set_index("model")
    fig, axes = _grid_axes(len(ORDERED), size=(4.4, 3.4))
    for ax, m in zip(axes, ORDERED):
        null = perm["null"][m]
        ax.hist(null, bins=15, color=MODEL_COLORS[m], alpha=0.55, edgecolor="white", linewidth=0.8)
        score, p = table.loc[m, "score_f1_macro"], table.loc[m, "p_value"]
        ax.axvline(score, color=INK_SECONDARY, lw=2, label=f"真实标签 {score:.3f}")
        ax.axvline(np.percentile(null, 95), color=INK_MUTED, lw=1, ls="--", label="零分布 95% 分位")
        ax.set_title(f"{MODEL_SHORT[m]}（p = {p:.3f}）")
        ax.set_xlabel("macro-F1（分层 5 折均值）")
        ax.set_ylabel("置换次数")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7.5, loc="upper right")
    n = int(table["n_permutations"].iloc[0])
    fig.suptitle(f"标签置换检验（默认超参数，{n} 次置换；p 越小越说明分数不是随机水平）", y=1.01)
    _save(fig, outpath)
