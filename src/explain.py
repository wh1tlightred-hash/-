"""SHAP 特征重要性解释（对最终所选模型，全量训练样本，训练后描述性分析）。

- 树模型：TreeExplainer（tree_path_dependent），在原始特征空间解释。
- 非树模型：KernelExplainer（模型无关），背景为全部训练样本，在原始特征空间解释，
  输出空间为 predict_proba 或 decision_function（SVM）的三类分数。
- 记录背景样本、解释对象、输出空间与计算预算；SHAP 只作描述性分析，非因果、非外部验证。
- 扩展分析：折外 SHAP（每条样本由没见过它的折模型解释）与六模型 SHAP 重要性对比。
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


def _output_space(clf) -> str:
    """解释器输出空间的中文描述（报告与 meta 共用）。"""
    from xgboost import XGBClassifier

    if isinstance(clf, XGBClassifier):
        return "原始 margin（softmax 前的 log-odds，非概率）"
    if hasattr(clf, "predict_proba"):
        return "predict_proba 概率"
    return "decision_function 决策分数"


def shap_values_for(pipeline, X_background, X_explain, cfg) -> dict:
    """对已拟合 Pipeline 计算 SHAP 值，返回 {values (n,f,c), base_values (c,), explainer, output_space, ...}。

    空间一致性守卫（对应审查提出的「泛化代码需留意」）：
    - 树解释器在**送入分类器的那个特征空间**上解释。若 Pipeline 在分类器前含 scaler，
      直接对原始特征做树 SHAP 会解释错空间，这里直接报错而不是给出看似合理的数字。
    - imputer 对**无缺失**数据是恒等变换；一旦数据含缺失，它就会改值，此时必须解释
      变换后的特征。这里按实际变换结果取值，并在返回值里记录是否恒等。
    - XGBoost：TreeExplainer 对 XGBClassifier 解释的是**原始 margin**，不是概率，
      输出空间如实标注为 margin，不因分类器有 predict_proba 就标成概率。
    - 非树模型：KernelExplainer 包住**整个 Pipeline**（含缩放），在原始特征空间解释，
      背景为 X_background（全量分析为全部训练样本，折外分析为该折训练集）。
    """
    clf = pipeline.named_steps["clf"]
    pre_steps = [name for name in pipeline.named_steps if name != "clf"]
    Xb = np.asarray(X_background, dtype=float)
    Xe = np.asarray(X_explain, dtype=float)

    if _is_tree(clf):
        if "scaler" in pre_steps:
            raise ValueError(
                "Pipeline 在分类器前含 scaler；直接对原始特征做树 SHAP 会解释错误的空间。"
                "请改为解释与 Pipeline 一致变换后的特征，或改用能吸收该变换的解释器。"
            )
        if "imputer" in pre_steps:
            X_used = np.asarray(pipeline.named_steps["imputer"].transform(Xe), dtype=float)
            identity = bool(np.array_equal(X_used, Xe))
        else:
            X_used, identity = Xe, True
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(X_used)
        base = explainer.expected_value
        kind = "TreeExplainer(tree_path_dependent)"
        background = ("原始特征空间（树模型无缩放；imputer 恒等，已验证）" if identity else
                      "送入分类器的变换后特征空间（imputer 非恒等，已按变换后取值解释）")
    else:
        fn = _scoring_fn(pipeline)
        nsamples = int(cfg.get("shap", {}).get("nsamples_kernel", 400))
        explainer = shap.KernelExplainer(fn, Xb)
        values = explainer.shap_values(Xe, nsamples=nsamples, silent=True)
        base = explainer.expected_value
        kind = f"KernelExplainer(nsamples={nsamples})"
        background = f"{len(Xb)} 条训练样本（原始特征空间，解释整个 Pipeline）"
        identity = True

    values = np.asarray(values, dtype=float)
    if values.ndim == 2:  # 单输出情况（一般不会发生）
        values = values[:, :, np.newaxis]
    base = np.atleast_1d(np.asarray(base, dtype=float))
    return {"values": values, "base_values": base, "explainer": kind, "background": background,
            "output_space": _output_space(clf), "preprocessing_is_identity": identity}


def _rank_tables(values, feature_names, class_names):
    """每类 mean(|SHAP|) 明细 + 三类聚合排名。"""
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
    return rank, agg


def _ranks_from_importance(imp: np.ndarray) -> np.ndarray:
    order = np.argsort(-imp, kind="stable")
    ranks = np.empty(len(imp), dtype=int)
    ranks[order] = np.arange(1, len(imp) + 1)
    return ranks


def explain_model(pipeline, X_raw, feature_names, class_names, cfg) -> dict:
    """计算 SHAP 值并返回 {values, base_values, ranking, aggregate, meta}。

    values: (n_samples, n_features, n_classes) 数组。
    ranking: 每类与聚合的 mean(|SHAP|) 排名 DataFrame。
    空间一致性守卫见 ``shap_values_for``。
    """
    clf = pipeline.named_steps["clf"]
    r = shap_values_for(pipeline, X_raw, X_raw, cfg)
    rank, agg = _rank_tables(r["values"], feature_names, class_names)
    n = len(np.asarray(X_raw))
    meta = {
        "model": MODEL_LABELS.get(_name_of(clf), type(clf).__name__),
        "explainer": r["explainer"],
        "background": r["background"],
        "explained": f"全部 {n} 条训练样本",
        "output_space": r["output_space"],
        "preprocessing_is_identity": bool(r["preprocessing_is_identity"]),
        "note": "训练后描述性分析，非外部验证，非因果关系证据；解释空间见 background 字段。",
    }
    return {"values": r["values"], "base_values": r["base_values"], "ranking": rank,
            "aggregate": agg, "meta": meta}


def cv_shap(fitted_per_fold: list, splits: list, X, feature_names, class_names, cfg) -> dict:
    """折外 SHAP：用每个外层折**只见过训练折**的模型，解释该折测试样本。

    拼起来得到全部样本的折外 SHAP 值——每条样本都由没见过它的模型解释，
    与全量拟合模型的「训练后」SHAP 互为对照。另给出逐折特征排名，用于看排名稳定性。
    逐折排名用该折**训练集**上的 mean(|SHAP|)：测试折只有 5–6 条，直接排名噪声过大。
    """
    X = np.asarray(X, dtype=float)
    values = np.full((len(X), len(feature_names), len(class_names)), np.nan)
    fold_rows = []
    kinds = set()
    for fold_i, ((tr, te), est) in enumerate(zip(splits, fitted_per_fold)):
        r = shap_values_for(est, X[tr], X[te], cfg)
        values[te] = r["values"]
        kinds.add((r["explainer"], r["output_space"]))
        r_tr = shap_values_for(est, X[tr], X[tr], cfg)
        imp = np.mean(np.abs(r_tr["values"]), axis=(0, 2))
        ranks = _ranks_from_importance(imp)
        for f, name in enumerate(feature_names):
            fold_rows.append({"fold": fold_i, "feature": name, "mean_abs_shap": float(imp[f]),
                              "rank": int(ranks[f])})
    if np.isnan(values).any():
        raise ValueError("折外 SHAP 未覆盖全部样本。")
    _, agg = _rank_tables(values, feature_names, class_names)
    explainer, output_space = sorted(kinds)[0]
    return {"values": values, "aggregate": agg, "fold_ranks": pd.DataFrame(fold_rows),
            "meta": {"explainer": explainer, "output_space": output_space,
                     "explained": "每条样本由其所在外层折的调参模型（未见过该样本）解释",
                     "fold_rank_basis": "逐折排名用该折训练集上的 mean(|SHAP|)"}}


def multi_model_shap(fitted: dict, X, feature_names, cfg) -> dict:
    """六模型 SHAP 重要性对比。fitted = {model: 全量拟合的 Pipeline}。

    不同模型的输出空间不同（概率 / 决策分数 / margin），绝对数值不可比，
    因此每个模型的 mean(|SHAP|) 归一化为占比（和为 1）后再比较，并给出排名。
    """
    rows, meta = [], {}
    for name, pipe in fitted.items():
        r = shap_values_for(pipe, X, X, cfg)
        imp = np.mean(np.abs(r["values"]), axis=(0, 2))
        share = imp / imp.sum() if imp.sum() > 0 else imp
        ranks = _ranks_from_importance(imp)
        for f, feat in enumerate(feature_names):
            rows.append({"model": name, "feature": feat, "mean_abs_shap": float(imp[f]),
                         "share": float(share[f]), "rank": int(ranks[f])})
        meta[name] = {"explainer": r["explainer"], "output_space": r["output_space"]}
    return {"table": pd.DataFrame(rows), "meta": meta}


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

    from .plot import STYLE

    plt.rcParams.update(STYLE)
    values = np.asarray(values)
    fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(4.2 * (len(class_names) + 1), 5))
    for c in range(len(class_names)):
        ax = axes[c]
        imp = np.mean(np.abs(values[:, :, c]), axis=0)
        order = np.argsort(imp)
        ax.barh(np.array(feature_names)[order], imp[order], color="#2a78d6", height=0.7)
        ax.set_title(f"{class_names[c]} (class {c})")
        ax.set_xlabel("mean(|SHAP|)")
        ax.grid(axis="y", visible=False)
    # 聚合
    ax = axes[-1]
    imp = np.mean(np.abs(values), axis=(0, 2))
    order = np.argsort(imp)
    ax.barh(np.array(feature_names)[order], imp[order], color="#1c5cab", height=0.7)
    ax.set_title("三类聚合")
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("SHAP 特征重要性（训练后描述性分析）", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
