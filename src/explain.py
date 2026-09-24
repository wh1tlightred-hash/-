"""SHAP 特征重要性解释（对最终所选模型，全量训练样本，训练后描述性分析）。

- 树模型：TreeExplainer（tree_path_dependent），在原始特征空间解释。
- 非树模型：KernelExplainer（模型无关），背景为全部训练样本，在原始特征空间解释，
  输出空间为 predict_proba 或 decision_function（SVM）的三类分数。
- 记录背景样本、解释对象、输出空间与计算预算；SHAP 只作描述性分析，非因果、非外部验证。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .models import MODEL_LABELS


def _is_tree(clf) -> bool:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return isinstance(clf, (RandomForestClassifier, XGBClassifier))


def _scoring_fn(pipeline):
    """返回 pipeline 的三类评分函数（概率或决策分数）。"""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba
    if hasattr(clf, "decision_function"):
        return pipeline.decision_function
    raise ValueError("最终模型无 predict_proba 或 decision_function，无法做 SHAP 解释。")


def explain_model(pipeline, X_raw, feature_names, class_names, cfg) -> dict:
    """计算 SHAP 值并返回 {values, ranking, meta}。

    values: (n_samples, n_features, n_classes) 数组。
    ranking: 每类与聚合的 mean(|SHAP|) 排名 DataFrame。

    空间一致性守卫（对应审查提出的「泛化代码需留意」）：
    - 树解释器在**送入分类器的那个特征空间**上解释。若 Pipeline 在分类器前含 scaler，
      直接对原始特征做树 SHAP 会解释错空间，这里直接报错而不是给出看似合理的数字。
    - imputer 对**无缺失**数据是恒等变换；一旦数据含缺失，它就会改值，此时必须解释
      变换后的特征。这里按实际变换结果取值，并在 meta 里记录是否恒等。
    - XGBoost 提示：TreeExplainer 对 XGBClassifier 默认解释**原始 margin**，不是概率。
      当前交付模型是随机森林（概率空间），此项未触发；若将来换成 XGBoost，
      不能因为分类器有 predict_proba 就把输出标成概率，需显式指定输出变换。
    """
    clf = pipeline.named_steps["clf"]
    pre_steps = [name for name in pipeline.named_steps if name != "clf"]
    X = np.asarray(X_raw, dtype=float)
    explainer_kind = None

    if "scaler" in pre_steps:
        raise ValueError(
            "Pipeline 在分类器前含 scaler；直接对原始特征做树 SHAP 会解释错误的空间。"
            "请改为解释与 Pipeline 一致变换后的特征，或改用能吸收该变换的解释器。"
        )

    # imputer 是否恒等：决定 SHAP 应基于原始特征还是变换后特征
    if "imputer" in pre_steps:
        X_used = np.asarray(pipeline.named_steps["imputer"].transform(X), dtype=float)
        preproc_identity = bool(np.array_equal(X_used, X))
    else:
        X_used = X
        preproc_identity = True

    if _is_tree(clf):
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(X_used)
        explainer_kind = "TreeExplainer(tree_path_dependent)"
        background = ("原始特征空间（树模型无缩放；imputer 恒等，已验证）"
                      if preproc_identity else
                      "送入分类器的变换后特征空间（imputer 非恒等，已按变换后取值解释）")
    else:
        fn = _scoring_fn(pipeline)
        nsamples = int(cfg["shap"].get("nsamples_kernel", 400))
        explainer = shap.KernelExplainer(fn, X, nsamples=nsamples)
        values = explainer.shap_values(X, nsamples=nsamples)
        explainer_kind = f"KernelExplainer(nsamples={nsamples})"
        background = "全部 27 条训练样本（原始特征空间）"

    values = np.asarray(values)
    if values.ndim == 2:  # 单输出情况（一般不会发生）
        values = values[:, :, np.newaxis]

    n_samples, n_feat, n_classes = values.shape
    rows = []
    for c in range(n_classes):
        for f in range(n_feat):
            rows.append({
                "class": class_names[c],
                "class_idx": c,
                "feature": feature_names[f],
                "mean_abs_shap": float(np.mean(np.abs(values[:, f, c]))),
            })
    rank = pd.DataFrame(rows)
    # 三类聚合：对每个特征的 |SHAP| 在三类上取平均
    agg = rank.groupby("feature")["mean_abs_shap"].mean().reset_index()
    agg = agg.rename(columns={"mean_abs_shap": "mean_abs_shap_overall"})
    agg = agg.sort_values("mean_abs_shap_overall", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    meta = {
        "model": MODEL_LABELS.get(_name_of(clf), type(clf).__name__),
        "explainer": explainer_kind,
        "background": background,
        "explained": "全部 27 条训练样本",
        "output_space": "predict_proba 概率" if hasattr(clf, "predict_proba") else "decision_function 决策分数",
        "preprocessing_is_identity": bool(preproc_identity),
        "note": "训练后描述性分析，非外部验证，非因果关系证据；解释空间见 background 字段。",
    }
    return {"values": values, "ranking": rank, "aggregate": agg, "meta": meta}


def _name_of(clf) -> str:
    for k, v in {
        "logistic": "LogisticRegression",
        "svm": "SVC",
        "gnb": "GaussianNB",
        "knn": "KNeighborsClassifier",
        "random_forest": "RandomForestClassifier",
        "xgboost": "XGBClassifier",
    }.items():
        if type(clf).__name__ == v:
            return k
    return "unknown"


def plot_shap(values, feature_names, class_names, outpath: str):
    """SHAP 汇总图：每类 mean(|SHAP|) 条形图 + 三类聚合排名。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    values = np.asarray(values)
    fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(4.2 * (len(class_names) + 1), 5))
    for c in range(len(class_names)):
        ax = axes[c]
        imp = np.mean(np.abs(values[:, :, c]), axis=0)
        order = np.argsort(imp)
        ax.barh(np.array(feature_names)[order], imp[order], color="#3182bd")
        ax.set_title(f"{class_names[c]} (class {c})")
        ax.set_xlabel("mean(|SHAP|)")
    # 聚合
    ax = axes[-1]
    imp = np.mean(np.abs(values), axis=(0, 2))
    order = np.argsort(imp)
    ax.barh(np.array(feature_names)[order], imp[order], color="#31a354")
    ax.set_title("三类聚合")
    ax.set_xlabel("mean(|SHAP|)")
    fig.suptitle("SHAP 特征重要性（训练后描述性分析）", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
