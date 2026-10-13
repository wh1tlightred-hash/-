"""交叉验证补充分析（不改动主实验的嵌套 CV 规则与胜者判定）。

主实验是种子 42 的外层 5 折 / 内层 3 折嵌套 CV。本模块回答它回答不了的几个问题：

1. ``cv_scheme_comparison`` —— 换一种切法（分层 3/5/9 折、留一法、重复分层 5 折）结论还一样吗？
2. ``permutation_tests``    —— 打乱标签后模型还能得到同样的分数吗？（置换检验，判断是否优于随机）
3. ``learning_curves``      —— 训练样本再多一点，性能还会涨吗？（学习曲线）
4. ``validation_curves``    —— 关键超参数变化时训练/验证分数如何变化？（验证曲线，观察欠/过拟合）
5. ``calibration_summary``  —— 概率模型给出的概率是否可信？（Brier、log loss、可靠性曲线）

统一约定：1–4 使用**默认超参数**的完整 Pipeline（不做调参），所以不存在「先在全数据上调参
再交叉验证」的泄漏；它们刻画的是模型家族在本数据上的行为，不替代嵌套 CV 主结果。
所有拟合型预处理都在 Pipeline 内，只在训练折内拟合。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    permutation_test_score,
    validation_curve,
)

from .metrics import class_scores, compute_metrics
from .models import ALL_MODELS, REAL_MODELS, build_pipeline, get_validation_param


def _fit_predict(name: str, seed: int, n_jobs: int, X_tr, y_tr, X_te):
    pipe = build_pipeline(name, seed=seed, n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te), class_scores(pipe, X_te), pipe


def pooled_cv_predictions(name: str, X, y, splitter, seed: int, n_jobs: int = 1, zero_division: int = 0):
    """按给定切分器做一次完整 CV，返回拼接后的折外预测与逐折指标。

    切分器必须让每条样本恰好出现在一个测试折中（KFold/LOO 类），否则拒绝拼接。
    逐折指标只在测试折含三类时计算（LOO 每折 1 条，不计算逐折指标）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    y_pred = np.full(len(y), -1, dtype=int)
    y_score = np.full((len(y), 3), np.nan)
    per_fold = []
    for fold_i, (tr, te) in enumerate(splitter.split(X, y)):
        if np.any(y_pred[te] != -1):
            raise ValueError("切分器的测试折存在重叠，不能拼接为一次折外预测。")
        pred, score, _ = _fit_predict(name, seed, n_jobs, X[tr], y[tr], X[te])
        y_pred[te] = pred
        y_score[te] = score
        if len(np.unique(y[te])) == 3:
            m = compute_metrics(y[te], score, pred, zero_division=zero_division)
            per_fold.append({"fold": fold_i, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                             "roc_auc_macro": m["roc_auc_macro"]})
    if np.any(y_pred == -1):
        raise ValueError("切分器没有覆盖全部样本。")
    pooled = compute_metrics(y, y_score, y_pred, zero_division=zero_division)
    return {"y_pred": y_pred, "y_score": y_score, "per_fold": per_fold, "pooled": pooled}


def _schemes(cfg: dict, seed: int) -> list[tuple[str, str, object]]:
    """返回 [(方案键, 中文名, 切分器)]。重复 CV 单独处理（每次重复都是一次完整拼接）。"""
    sc = cfg["analysis"]["cv_schemes"]
    out = []
    for k in sc["kfold"]:
        out.append((f"skf{k}", f"分层 {k} 折",
                    StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)))
    if sc.get("loo", True):
        out.append(("loo", "留一法 LOOCV", LeaveOneOut()))
    return out


def cv_scheme_comparison(X, y, cfg: dict, seed: int, models: list[str] | None = None) -> dict:
    """比较多种 CV 方案下（默认超参数）各模型的折外指标。

    返回：
      summary  —— 每个 (模型, 方案) 一行：拼接折外 Accuracy / macro-F1 / macro-AUC，
                  以及「单元」分数的均值±标准差（k 折为逐折，重复 CV 为逐次重复的拼接分数）。
      units    —— 逐单元明细（用于箱线图）。
    """
    models = models or ALL_MODELS
    n_jobs = int(cfg["n_jobs"])
    zd = cfg["zero_division"]
    rep_cfg = cfg["analysis"]["cv_schemes"]["repeated"]
    rep_splits, n_rep = int(rep_cfg["n_splits"]), int(rep_cfg["n_repeats"])
    rep_key, rep_label = f"rskf{rep_splits}x{n_rep}", f"重复分层 {rep_splits} 折 ×{n_rep}"

    summary, units = [], []
    for name in models:
        for key, label, splitter in _schemes(cfg, seed):
            r = pooled_cv_predictions(name, X, y, splitter, seed, n_jobs, zd)
            f1s = [f["f1_macro"] for f in r["per_fold"]]
            for f in r["per_fold"]:
                units.append({"model": name, "scheme": key, "scheme_label": label, "unit": "fold",
                              "unit_id": f["fold"], "accuracy": f["accuracy"], "f1_macro": f["f1_macro"],
                              "roc_auc_macro": f["roc_auc_macro"]})
            summary.append({
                "model": name, "scheme": key, "scheme_label": label,
                "n_fits": int(splitter.get_n_splits(np.asarray(X), y)),
                "accuracy": r["pooled"]["accuracy"], "f1_macro": r["pooled"]["f1_macro"],
                "roc_auc_macro": r["pooled"]["roc_auc_macro"],
                "unit_kind": "逐折" if f1s else "—",
                "unit_f1_mean": float(np.mean(f1s)) if f1s else np.nan,
                "unit_f1_std": float(np.std(f1s)) if f1s else np.nan,
            })
        # 重复分层 k 折：每次重复用不同随机种子，得到一次完整的 27 条拼接预测
        rep_rows = []
        for r_i in range(n_rep):
            skf = StratifiedKFold(n_splits=rep_splits, shuffle=True, random_state=seed + r_i)
            r = pooled_cv_predictions(name, X, y, skf, seed, n_jobs, zd)
            row = {"accuracy": r["pooled"]["accuracy"], "f1_macro": r["pooled"]["f1_macro"],
                   "roc_auc_macro": r["pooled"]["roc_auc_macro"]}
            rep_rows.append(row)
            units.append({"model": name, "scheme": rep_key, "scheme_label": rep_label, "unit": "repeat",
                          "unit_id": r_i, **row})
        rep_df = pd.DataFrame(rep_rows)
        summary.append({
            "model": name, "scheme": rep_key, "scheme_label": rep_label,
            "n_fits": rep_splits * n_rep,
            "accuracy": float(rep_df["accuracy"].mean()), "f1_macro": float(rep_df["f1_macro"].mean()),
            "roc_auc_macro": float(rep_df["roc_auc_macro"].mean()),
            "unit_kind": "逐次重复",
            "unit_f1_mean": float(rep_df["f1_macro"].mean()),
            "unit_f1_std": float(rep_df["f1_macro"].std(ddof=0)),
        })
    return {"summary": pd.DataFrame(summary), "units": pd.DataFrame(units)}


def permutation_tests(X, y, cfg: dict, seed: int, models: list[str] | None = None) -> dict:
    """标签置换检验（sklearn.permutation_test_score，默认超参数，分层 5 折，macro-F1）。

    p 值 = (置换分数 ≥ 真实分数的次数 + 1) / (置换次数 + 1)。它检验的是
    「模型分数是否只是标签与特征无关时也能得到的水平」，不检验模型之间谁更好。
    """
    models = models or REAL_MODELS
    pc = cfg["analysis"]["permutation"]
    n_perm = int(pc["n_permutations"])
    cv = StratifiedKFold(n_splits=int(pc["cv_splits"]), shuffle=True, random_state=seed)
    rows, null = [], {}
    for name in models:
        pipe = build_pipeline(name, seed=seed, n_jobs=int(cfg["n_jobs"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, perm_scores, pvalue = permutation_test_score(
                pipe, np.asarray(X, dtype=float), y, cv=cv, n_permutations=n_perm,
                scoring="f1_macro", random_state=seed, n_jobs=int(cfg["n_jobs"]))
        null[name] = np.asarray(perm_scores, dtype=float)
        rows.append({"model": name, "score_f1_macro": float(score),
                     "null_mean": float(np.mean(perm_scores)), "null_p95": float(np.percentile(perm_scores, 95)),
                     "p_value": float(pvalue), "n_permutations": n_perm})
    return {"table": pd.DataFrame(rows), "null": null}


def learning_curves(X, y, cfg: dict, seed: int, models: list[str] | None = None) -> pd.DataFrame:
    """分层学习曲线：在每个外层训练折内按「每类 k 条」分层子采样，评估测试折 macro-F1。

    不用 sklearn.learning_curve 的随机前 n 条——27 条样本下小子集常缺类，
    会让 XGBoost 等直接报错或让 macro 指标维度不一致。这里每类等量抽取，保证三类齐全。
    """
    models = models or REAL_MODELS
    lc = cfg["analysis"]["learning_curve"]
    sizes = [int(k) for k in lc["per_class_sizes"]]
    n_splits, n_rep = int(lc["n_splits"]), int(lc["n_repeats"])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
    for r_i in range(n_rep):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r_i)
        for fold_i, (tr, te) in enumerate(skf.split(X, y)):
            per_class = [tr[y[tr] == c] for c in range(3)]
            cap = min(len(pc) for pc in per_class)
            for k in sizes:
                if k > cap:
                    continue
                sub = np.concatenate([rng.choice(pc, size=k, replace=False) for pc in per_class])
                for name in models:
                    pred_te, score_te, pipe = _fit_predict(name, seed, int(cfg["n_jobs"]), X[sub], y[sub], X[te])
                    pred_tr = pipe.predict(X[sub])
                    m_tr = compute_metrics(y[sub], class_scores(pipe, X[sub]), pred_tr, cfg["zero_division"])
                    m_te = compute_metrics(y[te], score_te, pred_te, cfg["zero_division"])
                    rows.append({"model": name, "repeat": r_i, "fold": fold_i, "per_class": k,
                                 "n_train": 3 * k, "train_f1_macro": m_tr["f1_macro"],
                                 "test_f1_macro": m_te["f1_macro"]})
    return pd.DataFrame(rows)


def validation_curves(X, y, cfg: dict, seed: int, models: list[str] | None = None) -> pd.DataFrame:
    """单个关键超参数的验证曲线（重复分层 5 折，macro-F1）。

    这是对「超参数敏感性」的描述性观察，不是调参过程；全部样本都参与了 CV，
    所以曲线最高点不能当作泛化性能，也不能拿来改主实验的搜索空间。
    """
    models = models or REAL_MODELS
    vc = cfg["analysis"]["validation_curve"]
    cv = RepeatedStratifiedKFold(n_splits=int(vc["n_splits"]), n_repeats=int(vc["n_repeats"]),
                                 random_state=seed)
    rows = []
    for name in models:
        param, values = get_validation_param(name)
        pipe = build_pipeline(name, seed=seed, n_jobs=int(cfg["n_jobs"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr, te = validation_curve(pipe, np.asarray(X, dtype=float), y, param_name=param,
                                      param_range=values, cv=cv, scoring="f1_macro",
                                      n_jobs=int(cfg["n_jobs"]), error_score=np.nan)
        for i, v in enumerate(values):
            rows.append({"model": name, "param": param.removeprefix("clf__"), "value": "None" if v is None else str(v),
                         "order": i, "train_mean": float(np.nanmean(tr[i])), "train_std": float(np.nanstd(tr[i])),
                         "test_mean": float(np.nanmean(te[i])), "test_std": float(np.nanstd(te[i]))})
    return pd.DataFrame(rows)


def multiclass_brier(y_true, proba) -> float:
    """多分类 Brier 分数：mean_i Σ_c (p_ic − 1[y_i=c])²，范围 [0, 2]，越低越好。"""
    y_true = np.asarray(y_true, dtype=int)
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1)))


def calibration_summary(oof: dict, cfg: dict, models: list[str]) -> dict:
    """对概率模型的折外概率做校准评估。SVM 输出决策分数（非概率），不参与。

    返回 table（Brier、log loss、ECE）与 curves（逐类可靠性曲线点）。
    27 条样本、每类 9 个正例：分箱很粗，可靠性曲线只作方向性参考。
    """
    n_bins = int(cfg["analysis"]["calibration_bins"])
    rows, curves = [], {}
    for name in models:
        key = (name, "untrained") if name == "dummy" else (name, "tuned")
        df = oof[key]
        y_true = df["y_true"].to_numpy(dtype=int)
        proba = df[[c for c in df.columns if c.startswith("score_")]].to_numpy(dtype=float)
        p = np.clip(proba, 1e-15, 1.0)
        logloss = float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))
        conf = proba.max(axis=1)
        correct = (proba.argmax(axis=1) == y_true).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        rows.append({"model": name, "brier": multiclass_brier(y_true, proba), "log_loss": logloss,
                     "ece_top1": float(ece), "mean_confidence": float(conf.mean()),
                     "accuracy": float(correct.mean())})
        curves[name] = {}
        for c in range(3):
            yb = (y_true == c).astype(int)
            frac_pos, mean_pred = calibration_curve(yb, proba[:, c], n_bins=n_bins, strategy="quantile")
            curves[name][c] = (mean_pred, frac_pos)
    return {"table": pd.DataFrame(rows), "curves": curves}
