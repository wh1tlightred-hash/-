"""多分类评价指标与曲线计算。

约定：y_true 为一维整数标签（0/1/2，固定顺序 低危/中危/高危）；
y_score 为 (n, 3) 分数矩阵，列序严格对齐类别 [0, 1, 2]；
OvR ROC/PR/AP 使用三列二值标签（one-vs-rest），不是把 one-hot y 当作训练输入。

指标清单（PROJECT_BRIEF.md 第 7 节）：
- Accuracy、macro / weighted / 逐类 Precision、Recall、F1（zero_division 策略明确）。
- ROC-AUC：三分类 OvR，逐类 AUC + macro 平均。
- mAP：每类 one-vs-rest 的 average_precision_score 取算术平均。
- macro PR-AUC：PR 曲线梯形积分（np.trapz），与 AP 算法不同，分开报告。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _check_inputs(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim != 2 or y_score.shape[1] != 3:
        raise ValueError(f"y_score 应为 (n, 3) 分数矩阵，实际 {y_score.shape}。")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true 与 y_score 行数不一致。")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score 含非有限值（NaN/inf）。")
    return y_true, y_score


def _auc_ovr(y_true: np.ndarray, y_score: np.ndarray):
    """逐类 OvR ROC-AUC。某类缺失时返回 NaN 并给出原因标记。"""
    aucs = []
    for c in range(3):
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(yb, y_score[:, c]))
    return aucs


def _ap_ovr(y_true: np.ndarray, y_score: np.ndarray):
    """逐类 average_precision_score（AP）。"""
    aps = []
    for c in range(3):
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) < 2:
            aps.append(np.nan)
        else:
            aps.append(average_precision_score(yb, y_score[:, c]))
    return aps


def pr_auc_trapezoid(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """PR 曲线梯形积分（trapezoidal PR-AUC），区别于 average_precision_score。

    沿 precision_recall_curve 原始阈值顺序积分，保留重复 recall 的垂直段。
    不能按相同 recall 的最大 precision 去重，否则会改变曲线面积。
    """
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
    if len(recall) < 2:
        return float(np.nan)
    return float(auc(recall, precision))


def _pr_auc_ovr(y_true: np.ndarray, y_score: np.ndarray):
    """逐类梯形积分 PR-AUC。"""
    aucs = []
    for c in range(3):
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(pr_auc_trapezoid(yb, y_score[:, c]))
    return aucs


def compute_metrics(y_true, y_score, y_pred, zero_division: int = 0) -> dict:
    """计算全部必需指标，返回扁平 dict（含逐类列表）。

    返回键：
      accuracy, precision_macro, recall_macro, f1_macro,
      precision_weighted, recall_weighted, f1_weighted,
      per_class_precision/recall/f1/support (list, 顺序 0/1/2),
      roc_auc_ovr (list), roc_auc_macro, ap_ovr (list), mAP,
      pr_auc_ovr (list), pr_auc_macro, confusion_matrix (3x3 list)
    """
    y_true, y_score = _check_inputs(y_true, y_score)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = [0, 1, 2]

    acc = float(accuracy_score(y_true, y_pred))
    # 固定 labels=[0,1,2]：若某类在真实与预测中都不出现，sklearn 默认会把 macro 平均的
    # 分母变成「实际出现的类别数」，导致不同折/不同重采样之间 macro 维度不一致。
    # 显式给 labels 后，缺失类按 zero_division 策略计入，macro 恒为三类平均。
    p_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=zero_division))
    r_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=zero_division))
    f_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=zero_division))
    p_w = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=zero_division))
    r_w = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=zero_division))
    f_w = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=zero_division))

    p_cls, r_cls, f_cls, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=zero_division
    )

    roc_ovr = _auc_ovr(y_true, y_score)
    ap_ovr = _ap_ovr(y_true, y_score)
    pr_ovr = _pr_auc_ovr(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
        "per_class_precision": [float(x) for x in p_cls],
        "per_class_recall": [float(x) for x in r_cls],
        "per_class_f1": [float(x) for x in f_cls],
        "support": [int(x) for x in sup],
        "roc_auc_ovr": [float(x) if np.isfinite(x) else None for x in roc_ovr],
        "roc_auc_macro": float(np.nanmean(roc_ovr)) if np.any(np.isfinite(roc_ovr)) else None,
        "ap_ovr": [float(x) if np.isfinite(x) else None for x in ap_ovr],
        "mAP": float(np.nanmean(ap_ovr)) if np.any(np.isfinite(ap_ovr)) else None,
        "pr_auc_ovr": [float(x) if np.isfinite(x) else None for x in pr_ovr],
        "pr_auc_macro": float(np.nanmean(pr_ovr)) if np.any(np.isfinite(pr_ovr)) else None,
        "confusion_matrix": cm,
    }


def class_scores(estimator, X: np.ndarray, class_order=(0, 1, 2)) -> np.ndarray:
    """提取对齐固定类别顺序 [0,1,2] 的分数矩阵。

    优先 predict_proba（概率），SVM 等无概率模型回退 decision_function（决策分数）。
    按 estimator.classes_ 显式重排，绝不假设列序。
    """
    class_order = np.array(class_order)
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(estimator.classes_)
    elif hasattr(estimator, "decision_function"):
        proba = np.asarray(estimator.decision_function(X), dtype=float)
        classes = np.asarray(estimator.classes_)
    else:
        raise ValueError("估计器既无 predict_proba 也无 decision_function。")
    # 将列重排到 class_order 顺序
    out = np.zeros((proba.shape[0], len(class_order)), dtype=float)
    for j, c in enumerate(class_order):
        idx = np.where(classes == c)[0]
        if idx.size == 0:
            raise ValueError(f"估计器 classes_ {classes.tolist()} 缺少类别 {c}。")
        out[:, j] = proba[:, idx[0]]
    return out


def interpolate_curve(x, y, grid) -> np.ndarray:
    """将曲线 (x 单调升, y) 线性插值到公共 grid。x 升序排序后逐点插值。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # 去重 x，取最大值对应 y（对 ROC 的 fpr 和 PR 的 recall 均适用）
    uniq: dict[float, float] = {}
    for xi, yi in zip(x, y):
        uniq[float(xi)] = max(uniq.get(float(xi), -np.inf), float(yi))
    xs = np.array(sorted(uniq))
    ys = np.array([uniq[float(v)] for v in xs])
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])


def roc_curves_per_class(y_true, y_score) -> dict:
    """逐类 OvR ROC 曲线点。返回 {class_idx: (fpr, tpr)}。"""
    out = {}
    for c in range(3):
        yb = (np.asarray(y_true) == c).astype(int)
        fpr, tpr, _ = roc_curve(yb, np.asarray(y_score)[:, c])
        out[c] = (fpr, tpr)
    return out


def pr_curves_per_class(y_true, y_score) -> dict:
    """逐类 OvR PR 曲线点。返回 {class_idx: (precision, recall)}。"""
    out = {}
    for c in range(3):
        yb = (np.asarray(y_true) == c).astype(int)
        precision, recall, _ = precision_recall_curve(yb, np.asarray(y_score)[:, c])
        out[c] = (precision, recall)
    return out
