"""图表绘制（全部使用 OOF 分数，Agg 后端，中文字体）。"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import interpolate_curve, pr_curves_per_class, roc_curves_per_class

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# 模型 → 颜色（统一使用，保证图表一致）
MODEL_COLORS = {
    "logistic": "#1f77b4",
    "svm": "#ff7f0e",
    "gnb": "#2ca02c",
    "knn": "#d62728",
    "random_forest": "#9467bd",
    "xgboost": "#8c564b",
    "dummy": "#7f7f7f",
}

_GRID = np.linspace(0.0, 1.0, 201)


def plot_class_counts(y: np.ndarray, class_names: list[str], outpath: str):
    counts = [int((np.asarray(y) == c).sum()) for c in range(3)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(class_names, counts, color=[MODEL_COLORS["gnb"], MODEL_COLORS["svm"], MODEL_COLORS["knn"]])
    ax.set_title("类别样本数（共 27 条）")
    ax.set_ylabel("样本数")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(c), ha="center")
    ax.set_ylim(0, max(counts) + 2)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


_METRIC_TITLES = {
    "accuracy": "Accuracy",
    "f1_macro": "macro F1",
    "roc_auc_macro": "macro ROC-AUC (OvR)",
    "mAP": "mAP (逐类 AP 平均)",
    "pr_auc_macro": "macro PR-AUC (梯形积分)",
}


def plot_model_metrics_bar(oof_metrics, class_names: list[str], outpath: str, metric_keys=None):
    """模型指标对比：区分调参与未调参。默认 5 个子图，含 mAP 与梯形 PR-AUC 两者。

    mAP 与 macro PR-AUC 的算法不同（逐类 AP 算术平均 vs PR 曲线梯形积分），
    简报要求分开报告，故两个都要出现在图上，不能只留其中一列。
    """
    from .models import MODEL_SHORT, SIMPLICITY_ORDER

    metric_keys = metric_keys or ["accuracy", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]
    titles = [_METRIC_TITLES.get(k, k) for k in metric_keys]
    real = [m for m in oof_metrics["model"].unique() if m != "dummy"]
    real = sorted(real, key=lambda m: SIMPLICITY_ORDER.index(m))

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4.6 * len(metric_keys), 4.4), squeeze=False)
    axes = axes[0]
    for ax, key, title in zip(axes, metric_keys, titles):
        x = np.arange(len(real))
        w = 0.38
        for vi, version in enumerate(["untuned", "tuned"]):
            vals = []
            for m in real:
                row = oof_metrics[(oof_metrics["model"] == m) & (oof_metrics["version"] == version)]
                vals.append(row[key].iloc[0] if len(row) else np.nan)
            ax.bar(x + (vi - 0.5) * w, vals, w, label="未调参" if version == "untuned" else "调参",
                   color="#9ecae1" if version == "untuned" else "#3182bd")
        # Dummy 基线虚线
        dummy = oof_metrics[oof_metrics["model"] == "dummy"]
        if len(dummy):
            dv = dummy[key].iloc[0]
            ax.axhline(dv, ls="--", color=MODEL_COLORS["dummy"], lw=1, label="Dummy 基线")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in real], rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("六模型 OOF 指标对比（种子 42，27 条样本）", y=1.12)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_uncertainty(ci_table, outpath: str, metrics=("accuracy", "f1_macro"), baseline=None):
    """OOF 指标的 bootstrap 95% 区间（误差棒）——直观显示各模型区间高度重叠。

    ci_table 为 results/uncertainty.csv 的原始列：model, model_label, metric, point, ci_lo, ci_hi, ...
    baseline：可选，形如 {"accuracy": 0.259, "f1_macro": 0.223}，用虚线标出，便于看「区间是否盖住基线」。
    """
    metrics = list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2 * len(metrics), 4.8), squeeze=False)
    axes = axes[0]
    for ax, metric in zip(axes, metrics):
        sub = ci_table[ci_table["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(sub))
        point = sub["point"].to_numpy(dtype=float)
        lo = sub["ci_lo"].to_numpy(dtype=float)
        hi = sub["ci_hi"].to_numpy(dtype=float)
        err = np.vstack([point - lo, hi - point])
        err = np.nan_to_num(err, nan=0.0)
        ax.errorbar(point, y, xerr=err, fmt="o", color="#3182bd", ecolor="#9ecae1",
                    elinewidth=2, capsize=4, markersize=6)
        if baseline is not None and metric in baseline:
            ax.axvline(float(baseline[metric]), ls="--", color=MODEL_COLORS["dummy"], lw=1.2,
                       label=f"Dummy 基线 {float(baseline[metric]):.3f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model_label"].tolist(), fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(_METRIC_TITLES.get(metric, metric))
        ax.set_title(f"{_METRIC_TITLES.get(metric, metric)} 的 bootstrap 95% 区间")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("OOF 指标的不确定性：对 27 条样本做样本级重采样（2000 次）", y=1.03)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _macro_roc(y_true, y_score):
    tprs = []
    curves = roc_curves_per_class(y_true, y_score)
    for c in range(3):
        fpr, tpr = curves[c]
        tprs.append(interpolate_curve(fpr, tpr, _GRID))
    return _GRID, np.mean(tprs, axis=0)


def _macro_pr(y_true, y_score):
    precs = []
    curves = pr_curves_per_class(y_true, y_score)
    for c in range(3):
        precision, recall = curves[c]
        precs.append(interpolate_curve(recall, precision, _GRID))
    return _GRID, np.mean(precs, axis=0)


def plot_macro_roc_comparison(oof, metric_by_model: dict, outpath: str):
    """六模型（调参）+ Dummy 的 macro ROC 对比（公共 FPR 网格插值取均值）。"""
    from .models import MODEL_SHORT, SIMPLICITY_ORDER

    fig, ax = plt.subplots(figsize=(7, 6))
    order = [m for m in SIMPLICITY_ORDER] + ["dummy"]
    for m in order:
        if (m, "tuned") in oof and m != "dummy":
            key = (m, "tuned")
            lab = MODEL_SHORT[m]
        elif (m, "untrained") in oof:
            key = (m, "untrained")
            lab = "Dummy"
        else:
            continue
        df = oof[key]
        y_true = df["y_true"].to_numpy()
        score_cols = [c for c in df.columns if c.startswith("score_")]
        y_score = df[score_cols].to_numpy(dtype=float)
        fpr, tpr = _macro_roc(y_true, y_score)
        auc_val = metric_by_model.get(m, np.nan)
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=1.8,
                label=f"{lab} (macro-AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="随机")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_title("六模型 macro ROC 对比（跨折拼接 OOF；图例 macro-AUC 为逐类 AUC 平均）")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_macro_pr_comparison(oof, metric_by_model: dict, outpath: str):
    """六模型（调参）+ Dummy 的 macro PR 对比（公共 recall 网格插值取均值）。"""
    from .models import MODEL_SHORT, SIMPLICITY_ORDER

    fig, ax = plt.subplots(figsize=(7, 6))
    order = [m for m in SIMPLICITY_ORDER] + ["dummy"]
    for m in order:
        if (m, "tuned") in oof and m != "dummy":
            key = (m, "tuned")
            lab = MODEL_SHORT[m]
        elif (m, "untrained") in oof:
            key = (m, "untrained")
            lab = "Dummy"
        else:
            continue
        df = oof[key]
        y_true = df["y_true"].to_numpy()
        score_cols = [c for c in df.columns if c.startswith("score_")]
        y_score = df[score_cols].to_numpy(dtype=float)
        recall, prec = _macro_pr(y_true, y_score)
        ap_val = metric_by_model.get(m, np.nan)
        ax.plot(recall, prec, color=MODEL_COLORS[m], lw=1.8,
                label=f"{lab} (mAP={ap_val:.3f})")
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("六模型 macro PR 对比（相同召回率取最大精确率后插值；图例 mAP 为逐类 AP 平均）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_per_class_curves(model_label: str, y_true, y_score, class_names: list[str], outpath: str):
    """单模型逐类 ROC 与 PR 曲线（每类一条）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    roc = roc_curves_per_class(y_true, y_score)
    pr = pr_curves_per_class(y_true, y_score)
    for c in range(3):
        fpr, tpr = roc[c]
        axes[0].plot(fpr, tpr, lw=1.8, label=f"{class_names[c]} (class {c})")
        precision, recall = pr[c]
        axes[1].plot(recall, precision, lw=1.8, label=f"{class_names[c]} (class {c})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("逐类 OvR ROC")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("逐类 OvR PR")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{model_label} 逐类曲线（OOF 分数）")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], outpath: str, normalize: str | None = None, title: str = ""):
    """混淆矩阵：横轴预测、纵轴真实；统一低危/中危/高危顺序。normalize='true' 时按真实类别行归一化。"""
    cm = np.asarray(cm, dtype=float)
    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = cm / row_sums
        fmt = ".2f"
        cbarlabel = "行归一化占比"
    else:
        display = cm
        fmt = ".0f"
        cbarlabel = "样本数"
    fig, ax = plt.subplots(figsize=(5, 4.6))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=(1 if normalize else display.max()))
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title or ("混淆矩阵" + ("（行归一化）" if normalize else "")))
    thresh = display.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(display[i, j], fmt), ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, label=cbarlabel)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
