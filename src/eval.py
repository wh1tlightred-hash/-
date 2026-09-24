"""嵌套交叉验证（外层 5 折 / 内层 3 折）、OOF 收集与调参执行。

- 所有模型、所有版本共享同一组外层样本索引（同一 StratifiedKFold 对象）。
- 内层只在每个外层训练折上独立调参，内层主评分固定为 macro-F1。
- 未调参（默认参数）与调参版本使用完全相同的外层切分。
- 每条样本对每个模型版本恰好有一次外层折外预测（OOF）。
"""
from __future__ import annotations

import time
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from .metrics import class_scores, compute_metrics
from .models import (
    ALL_MODELS,
    REAL_MODELS,
    build_pipeline,
    get_search_space,
    grid_n_candidates,
)

VERSIONS = ("untuned", "tuned")


def _min_inner_train_size(X_train, y_train, inner_cv):
    return min(len(tr) for tr, _ in inner_cv.split(X_train, y_train))


def _make_search(pipe, grid, search_type, inner_cv, scoring, seed, n_iter, n_jobs):
    common = dict(cv=inner_cv, scoring=scoring, refit=True, n_jobs=n_jobs)
    if search_type == "grid":
        return GridSearchCV(pipe, grid, **common)
    return RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, random_state=seed, **common
    )


def _capture_warnings(caught):
    """汇总一次 fit 捕获的警告：计数与唯一消息。"""
    msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    from collections import Counter

    return {"n_warnings": len(msgs), "unique": dict(Counter(msgs))}


def validate_candidate_budget(budget: dict, cfg: dict) -> None:
    """校验每折候选数不超过 config.json 的 max_candidates_per_model。

    审查指出该配置原先只用于描述、并非硬上限。这里让它在运行时真正生效：
    超限直接报错，**不做静默截断**（静默截断会让落盘的预算表与实际拟合数不符）。
    """
    cap = int(cfg.get("max_candidates_per_model") or 0)
    if cap <= 0:
        return
    over = {
        m: b.get("candidates_per_outer_fold")
        for m, b in budget.items()
        if b.get("candidates_per_outer_fold") and b["candidates_per_outer_fold"] > cap
    }
    if over:
        raise ValueError(
            f"以下模型的每折候选数超过 config.json 的 max_candidates_per_model={cap}：{over}。"
            f"请缩小搜索空间，或显式调大该上限。"
        )


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    cfg: dict,
    seed: int,
    model_names: list[str] | None = None,
) -> dict:
    """执行一轮嵌套交叉验证。返回 {'oof', 'folds', 'budget'}。"""
    models = model_names or ALL_MODELS
    outer = StratifiedKFold(
        n_splits=cfg["outer_cv"]["n_splits"],
        shuffle=cfg["outer_cv"]["shuffle"],
        random_state=seed,
    )
    inner = StratifiedKFold(
        n_splits=cfg["inner_cv"]["n_splits"],
        shuffle=cfg["inner_cv"]["shuffle"],
        random_state=seed,
    )
    scoring = cfg["inner_scoring"]
    n_iter = int(cfg["n_iter_random"])
    n_jobs = int(cfg["n_jobs"])

    Xn = np.asarray(X, dtype=float)
    splits = list(outer.split(Xn, y))  # 固定同一组外层索引，供所有模型复用
    audit_dir = Path(cfg["results_dir"]) / "search_audit" / f"seed_{seed}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    split_audit = []
    for fold_i, (tr, te) in enumerate(splits):
        inner_splits = [{"train": tr[a].tolist(), "validation": tr[b].tolist()}
                        for a, b in inner.split(Xn[tr], y[tr])]
        split_audit.append({"fold": fold_i, "train": tr.tolist(), "test": te.tolist(),
                            "inner": inner_splits})
    (audit_dir / "splits.json").write_text(json.dumps({
        "sample_ids": meta["sample_id"].tolist(), "folds": split_audit
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    oof: dict[tuple[str, str], pd.DataFrame] = {}
    folds: list[dict] = []
    budget: dict[str, dict] = {}

    for name in models:
        is_real = name in REAL_MODELS
        versions = VERSIONS if is_real else ("untrained",)  # Dummy 仅基线（prior 无参数）
        # 预算记录（真实模型）：搜索空间大小与每折拟合数
        if is_real:
            space_size = None
            search_type = None
            # 用第一个外层训练折估算最小内层训练集大小（各折差异极小）
            tr0, _ = splits[0]
            min_inner = _min_inner_train_size(Xn[tr0], y[tr0], inner)
            grid, search_type = get_search_space(name, min_inner)
            space_size = grid_n_candidates(grid)
            cand_per_fold = space_size if search_type == "grid" else min(n_iter, space_size)
            budget[name] = {
                "search_type": search_type,
                "space_size": space_size,
                "candidates_per_outer_fold": cand_per_fold,
                "max_candidates_cap": int(cfg.get("max_candidates_per_model") or 0),
                "inner_folds": cfg["inner_cv"]["n_splits"],
                "fits_per_outer_fold": cand_per_fold * cfg["inner_cv"]["n_splits"],
                "total_tuned_fits": cand_per_fold * cfg["inner_cv"]["n_splits"] * cfg["outer_cv"]["n_splits"],
            }
            # 硬上限校验：一旦某模型超限，在它开始拟合之前就报错
            validate_candidate_budget(budget, cfg)
        else:
            budget[name] = {"search_type": None, "space_size": None,
                            "candidates_per_outer_fold": 0, "inner_folds": 0,
                            "fits_per_outer_fold": 0, "total_tuned_fits": 0}

        for version in versions:
            # 用于 OOF 汇总的缓存
            yt_list, yp_list, ys_list, fold_list, idx_list = [], [], [], [], []
            train_accs, train_f1s = [], []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = Xn[train_idx], Xn[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                if not is_real:  # Dummy
                    from .models import make_estimator

                    est = make_estimator(name, seed=seed, n_jobs=n_jobs)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        est.fit(X_tr, y_tr)
                    elapsed = time.perf_counter() - t0
                    pred_start = time.perf_counter()
                    y_pred = est.predict(X_te)
                    y_score = class_scores(est, X_te)
                    prediction_time = time.perf_counter() - pred_start
                    best_params = {}
                    best_inner_score = None
                    warn_sum = _capture_warnings(caught)
                elif version == "untuned":
                    pipe = build_pipeline(name, seed=seed, n_jobs=n_jobs, params=None)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pipe.fit(X_tr, y_tr)
                    elapsed = time.perf_counter() - t0
                    pred_start = time.perf_counter()
                    y_pred = pipe.predict(X_te)
                    y_score = class_scores(pipe, X_te)
                    prediction_time = time.perf_counter() - pred_start
                    best_params = {}
                    best_inner_score = None
                    warn_sum = _capture_warnings(caught)
                else:  # tuned
                    min_inner = _min_inner_train_size(X_tr, y_tr, inner)
                    grid, search_type = get_search_space(name, min_inner)
                    pipe = build_pipeline(name, seed=seed, n_jobs=n_jobs, params=None)
                    search = _make_search(pipe, grid, search_type, inner, scoring, seed, n_iter, n_jobs)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        search.fit(X_tr, y_tr)
                    elapsed = time.perf_counter() - t0
                    pd.DataFrame(search.cv_results_).to_csv(
                        audit_dir / f"{name}_fold_{fold_i}_candidates.csv", index=False,
                        encoding="utf-8-sig")
                    pred_start = time.perf_counter()
                    y_pred = search.predict(X_te)
                    y_score = class_scores(search, X_te)
                    prediction_time = time.perf_counter() - pred_start
                    best_params = {
                        k.removeprefix("clf__"): v for k, v in search.best_params_.items()
                    }
                    best_inner_score = float(search.best_score_)
                    warn_sum = _capture_warnings(caught)

                # 训练集指标（观察过拟合的辅助）
                if not is_real:
                    tr_pred = est.predict(X_tr)
                    tr_score = class_scores(est, X_tr)
                elif version == "untuned":
                    tr_pred = pipe.predict(X_tr)
                    tr_score = class_scores(pipe, X_tr)
                else:
                    tr_pred = search.predict(X_tr)
                    tr_score = class_scores(search, X_tr)
                tr_metrics = compute_metrics(y_tr, tr_score, tr_pred, zero_division=cfg["zero_division"])
                train_accs.append(tr_metrics["accuracy"])
                train_f1s.append(tr_metrics["f1_macro"])

                te_metrics = compute_metrics(y_te, y_score, y_pred, zero_division=cfg["zero_division"])

                folds.append({
                    "seed": seed, "model": name, "version": version, "fold": fold_i,
                    "best_params": best_params, "best_inner_score": best_inner_score,
                    "train_accuracy": tr_metrics["accuracy"], "train_f1_macro": tr_metrics["f1_macro"],
                    "train_time_s": elapsed, "prediction_time_s": prediction_time,
                    "warnings": warn_sum,
                    "metrics": te_metrics,
                })

                yt_list.append(y_te)
                yp_list.append(y_pred)
                ys_list.append(y_score)
                fold_list.append(np.full(len(y_te), fold_i, dtype=int))
                idx_list.append(test_idx)

            # 组装 OOF
            oof_df = pd.DataFrame({
                "sample_id": meta["sample_id"].iloc[np.concatenate(idx_list)].values,
                "label_name": meta["label_name"].iloc[np.concatenate(idx_list)].values,
                "y_true": np.concatenate(yt_list),
                "y_pred": np.concatenate(yp_list),
                "fold": np.concatenate(fold_list),
            })
            score_mat = np.vstack(ys_list)
            for c, cn in enumerate(cfg["class_names"]):
                oof_df[f"score_{c}_{cn}"] = score_mat[:, c]
            oof[(name, version)] = oof_df.reset_index(drop=True)

    return {"oof": oof, "folds": folds, "budget": budget}


def aggregate_fold_metrics(folds: list[dict], metric_keys: list[str]) -> pd.DataFrame:
    """将逐折指标聚合为均值±标准差表（跨外层折）。"""
    rows = []
    for key in metric_keys:
        for model in sorted({f["model"] for f in folds}):
            for version in sorted({f["version"] for f in folds if f["model"] == model}):
                vals = [f["metrics"][key] for f in folds if f["model"] == model and f["version"] == version]
                vals = [v for v in vals if v is not None]
                if vals:
                    rows.append({
                        "model": model, "version": version, "metric": key,
                        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    })
    return pd.DataFrame(rows)


def oof_metrics(oof: dict[tuple[str, str], pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """从 OOF 预测重新计算汇总指标（与折均值不同的完整 OOF 汇总值）。"""
    rows = []
    for (model, version), df in oof.items():
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        y_score = df[[f"score_{c}_{cn}" for c, cn in enumerate(cfg["class_names"])]].to_numpy(dtype=float)
        m = compute_metrics(y_true, y_score, y_pred, zero_division=cfg["zero_division"])
        row = {"model": model, "version": version}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)
