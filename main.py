"""中医声诊三分类项目一键实验入口。

流程：环境核验 → 数据加载/校验/落盘 → 种子 42 主嵌套 CV → 种子 43/44 稳定性 →
胜者选择 → 全量最终模型（训练内调参 + 拟合 + 保存）→ SHAP → 图表 → 中文报告。

运行：.venv/Scripts/python.exe main.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# 必须在导入 matplotlib 前设置本地缓存目录（避免写入用户目录失败）
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    file_sha256,
    load_dataset,
    split_xy,
    validate_dataset,
)
from src.eval import oof_metrics, run_nested_cv  # noqa: E402
from src.explain import explain_model, plot_shap  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_SHORT,
    REAL_MODELS,
    SIMPLICITY_ORDER,
    VERSION_SHORT,
    build_pipeline,
    get_search_space,
)
from src.plot import (  # noqa: E402
    plot_class_counts,
    plot_confusion_matrix,
    plot_macro_pr_comparison,
    plot_macro_roc_comparison,
    plot_model_metrics_bar,
    plot_per_class_curves,
    plot_uncertainty,
)
from src.report import build_report, write_reports  # noqa: E402
from src.uncertainty import (  # noqa: E402
    bootstrap_ci,
    mcnemar_exact,
    nested_selection_estimate,
    score_matrix_from_oof,
)

KEY_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]


def _env_info(cfg: dict) -> dict:
    import platform
    import subprocess

    from sklearn import __version__ as skv
    import xgboost, shap, scipy, matplotlib, seaborn

    pip_check = "见 logs/pip_check.txt"
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
        pip_check = r.stdout.strip() or r.stderr.strip() or "(ok)"
    except Exception as e:  # noqa: BLE001
        pip_check = f"无法运行 pip check：{e}"

    return {
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pip_check": pip_check,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": skv,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
    }


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _run_one_seed(cfg, X, y, meta, seed: int, results_dir: Path) -> dict:
    """运行一轮嵌套 CV 并落盘 OOF、逐折记录、预算、OOF 指标。"""
    seed_dir = results_dir / f"seed{seed}"
    res = run_nested_cv(X, y, meta, cfg, seed=seed)

    oof_dir = seed_dir / "oof"
    for (model, version), df in res["oof"].items():
        p = oof_dir / f"{model}_{version}_oof.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    folds_df = pd.DataFrame(res["folds"])
    # metrics / warnings 是嵌套 dict，这里原样落盘为字符串；逐列指标请用 oof_metrics.csv，
    # 完整嵌套结构请用同目录的 OOF 文件重算（见 verify_acceptance.py 第 6 项）。
    folds_df.to_csv(seed_dir / "folds.csv", index=False)
    _save_json(res["budget"], seed_dir / "budget.json")

    om = oof_metrics(res["oof"], cfg)
    om.to_csv(seed_dir / "oof_metrics.csv", index=False)
    return {"res": res, "oof_metrics": om, "seed_dir": seed_dir}


def _winner(oof_metrics: pd.DataFrame, cfg: dict) -> dict:
    """按预先确定的 macro-F1 选胜者；并列依次比较 Accuracy、模型简洁性。"""
    tuned = oof_metrics[(oof_metrics["version"] == "tuned") & (oof_metrics["model"] != "dummy")].copy()
    tuned = tuned.sort_values(
        ["f1_macro", "accuracy"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    best_f1 = tuned["f1_macro"].iloc[0]
    # 并列组内按简洁性
    tie = tuned[tuned["f1_macro"] >= best_f1 - 1e-12]
    tie = tie.sort_values("accuracy", ascending=False, kind="stable")
    tie = tie[tie["accuracy"] >= tie["accuracy"].max() - 1e-12]
    tie = tie.assign(_simp=tie["model"].map(lambda m: SIMPLICITY_ORDER.index(m)))
    tie = tie.sort_values("_simp", kind="stable")
    win_model = tie["model"].iloc[0]

    acc_row = tuned.sort_values("accuracy", ascending=False, kind="stable").iloc[0]
    acc_model = acc_row["model"]

    win_row = tuned[tuned["model"] == win_model].iloc[0]
    return {
        "f1_best": win_model,
        "f1_best_score": float(win_row["f1_macro"]),
        "f1_best_accuracy": float(win_row["accuracy"]),
        "accuracy_best": acc_model,
        "accuracy_best_score": float(acc_row["accuracy"]),
        "tie_break_note": (
            f"胜者按 macro-F1 判定；若 {win_model} 与 {acc_model} 不同，则分别给出两者混淆矩阵。"
            "并列时依次比较 Accuracy 与简洁性（预先固定顺序）。"
        ),
    }


def _uncertainty(cfg, oof, folds_df, win, results_dir: Path) -> dict:
    """补充分析：bootstrap 区间、胜者 vs Dummy 的配对检验、含模型选择的诚实估计。

    只用已算好的 OOF 与逐折记录，不改动主实验的任何规则、不重新训练。
    """
    rows = []
    for m in REAL_MODELS + ["dummy"]:
        version = "untrained" if m == "dummy" else "tuned"
        df = oof[(m, version)]
        ci = bootstrap_ci(
            df["y_true"].to_numpy(dtype=int),
            df["y_pred"].to_numpy(dtype=int),
            score_matrix_from_oof(df),
            seed=int(cfg["seed"]),
            zero_division=cfg["zero_division"],
        )
        for k, v in ci.items():
            rows.append({
                "model": m, "model_label": MODEL_SHORT[m], "metric": k,
                "point": v["point"], "ci_lo": v["lo"], "ci_hi": v["hi"],
                "n_boot": v["n_boot"], "n_valid": v["n_valid"], "n_invalid": v["n_invalid"],
            })
    ci_table = pd.DataFrame(rows)
    ci_table.to_csv(results_dir / "uncertainty.csv", index=False)

    winner_df = oof[(win["f1_best"], "tuned")]
    dummy_df = oof[("dummy", "untrained")]
    mc = mcnemar_exact(
        winner_df["y_true"].to_numpy(dtype=int),
        winner_df["y_pred"].to_numpy(dtype=int),
        dummy_df["y_pred"].to_numpy(dtype=int),
    )
    mc["model_a"] = win["f1_best"]
    mc["model_b"] = "dummy"
    _save_json(mc, results_dir / "mcnemar.json")

    sel = nested_selection_estimate(folds_df, oof, zero_division=cfg["zero_division"])
    sel_out = {
        "picked_per_fold": sel["picked_per_fold"],
        "n_samples": sel["n_samples"],
        "metrics": {k: sel["metrics"][k] for k in KEY_METRICS},
        "posthoc_winner": {
            "model": win["f1_best"],
            "accuracy": win["f1_best_accuracy"],
            "f1_macro": win["f1_best_score"],
        },
        "note": "每折按内层 macro-F1 选模型后拼接折外预测；模型选择不含事后挑选，仍非独立外部测试集。",
    }
    _save_json(sel_out, results_dir / "selection_estimate.json")
    return {"ci_table": ci_table, "mcnemar": mc, "selection": sel_out}


def _refit_final(cfg, X, y, model_name: str, seed: int, n_jobs: int):
    """在全部 27 条样本上重新做训练内调参并拟合最终模型（仅用于演示预测）。"""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

    inner = StratifiedKFold(n_splits=cfg["inner_cv"]["n_splits"], shuffle=cfg["inner_cv"]["shuffle"], random_state=seed)
    Xn = np.asarray(X, dtype=float)
    min_inner = min(len(tr) for tr, _ in inner.split(Xn, y))
    grid, search_type = get_search_space(model_name, min_inner)
    pipe = build_pipeline(model_name, seed=seed, n_jobs=n_jobs, params=None)
    common = dict(cv=inner, scoring=cfg["inner_scoring"], refit=True, n_jobs=n_jobs)
    if search_type == "grid":
        search = GridSearchCV(pipe, grid, **common)
    else:
        search = RandomizedSearchCV(pipe, grid, n_iter=int(cfg["n_iter_random"]), random_state=seed, **common)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(Xn, y)
    best_params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    final_pipe = search.best_estimator_
    train_metrics = compute_metrics(y, _score(final_pipe, Xn), final_pipe.predict(Xn), cfg["zero_division"])
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    return final_pipe, best_params, search.best_score_, train_metrics, warn_msgs


def _score(pipeline, X):
    from src.metrics import class_scores

    return class_scores(pipeline, X)


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    results_dir = ROOT / cfg["results_dir"]
    models_dir = ROOT / cfg["models_dir"]
    data_dir = ROOT / cfg["data_processed_dir"]
    report_dir = ROOT / cfg["report_dir"]
    fig_dir = results_dir / "figures"
    logs_dir = ROOT / "logs"
    for d in (results_dir, models_dir, data_dir, report_dir, fig_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[1/8] 数据加载与校验")
    df = load_dataset(cfg["input_xlsx"], cfg["feature_columns"])
    val = validate_dataset(df, cfg["feature_columns"])
    if not val["ok"]:
        raise ValueError("数据契约检查失败：" + "; ".join(val["errors"]))
    X, y, meta = split_xy(df, cfg["feature_columns"])
    df.to_csv(data_dir / "merged.csv", index=False)
    meta.to_csv(data_dir / "source_mapping.csv", index=False)
    _save_json({"label_map": cfg["label_map"], "class_names": cfg["class_names"],
                "feature_order": cfg["feature_columns"]}, data_dir / "label_map.json")
    _save_json(val, data_dir / "validation_report.json")
    hashes = {"xlsx": file_sha256(cfg["input_xlsx"]), "docx": file_sha256(cfg["input_docx"])}
    _save_json(hashes, results_dir / "input_hashes.json")
    print(f"   数据 {df.shape}，三类计数 {dict(Counter(y.tolist()))}，校验通过={val['ok']}")
    for w in val["warnings"]:
        print(f"   警告：{w}")

    print("[2/8] 环境核验")
    env = _env_info(cfg)
    _save_json(env, logs_dir / "environment_check.json")
    with open(logs_dir / "pip_check.txt", "w", encoding="utf-8") as f:
        f.write(env["pip_check"] + "\n")
    print(f"   Python {env['python']} | sklearn {env['packages']['scikit-learn']} | xgboost {env['packages']['xgboost']}")

    print("[3/8] 种子 42 主嵌套交叉验证（外层5折/内层3折）")
    main_run = _run_one_seed(cfg, X, y, meta, cfg["seed"], results_dir)
    om42 = main_run["oof_metrics"]

    print("[4/8] 种子 43/44 切分稳定性")
    stability_rows = []
    for s in cfg["stability_seeds"]:
        r = _run_one_seed(cfg, X, y, meta, s, results_dir)
        om = r["oof_metrics"].copy()
        om["seed"] = s
        stability_rows.append(om)
    stability = pd.concat([om42.copy().assign(seed=cfg["seed"])] + stability_rows, ignore_index=True)
    stability.to_csv(results_dir / "stability.csv", index=False)

    print("[5/8] 胜者选择与图表")
    win = _winner(om42, cfg)
    _save_json(win, results_dir / "winner.json")

    # 逐折均值±标准差
    folds_df = pd.DataFrame(main_run["res"]["folds"])
    fold_agg_rows = []
    for model in sorted(folds_df["model"].unique()):
        for version in sorted(folds_df[folds_df["model"] == model]["version"].unique()):
            sub = folds_df[(folds_df["model"] == model) & (folds_df["version"] == version)]
            row = {"model": model, "version": version}
            for k in KEY_METRICS:
                vals = [f["metrics"][k] for f in sub.to_dict("records")]
                vals = [v for v in vals if v is not None]
                row[k] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "NA"
            fold_agg_rows.append(row)
    fold_agg = pd.DataFrame(fold_agg_rows)
    fold_agg.to_csv(results_dir / "fold_agg.csv", index=False)

    # 补充分析：不确定性量化 + 含模型选择的诚实估计（只读已有 OOF，不重训）
    unc = _uncertainty(cfg, main_run["res"]["oof"], folds_df, win, results_dir)
    print(f"   胜者 vs Dummy 配对检验 p={unc['mcnemar']['p_value']:.4f}；"
          f"含选择估计 macro-F1={unc['selection']['metrics']['f1_macro']:.3f}（事后挑选为 {win['f1_best_score']:.3f}）")

    # 图表
    fig_paths = {}
    plot_class_counts(y, cfg["class_names"], str(fig_dir / "class_counts.png"))
    fig_paths["class_counts"] = "../results/figures/class_counts.png"
    plot_model_metrics_bar(om42, cfg["class_names"], str(fig_dir / "model_metrics.png"))
    fig_paths["model_metrics"] = "../results/figures/model_metrics.png"

    oof = main_run["res"]["oof"]
    # 逐类曲线：六模型调参版
    per_class_figs = []
    for m in REAL_MODELS:
        key = (m, "tuned")
        o = oof[key]
        y_true = o["y_true"].to_numpy()
        score_cols = [c for c in o.columns if c.startswith("score_")]
        y_score = o[score_cols].to_numpy(dtype=float)
        fn = f"per_class_{m}.png"
        plot_per_class_curves(MODEL_SHORT[m], y_true, y_score, cfg["class_names"], str(fig_dir / fn))
        per_class_figs.append({"title": f"{MODEL_LABELS[m]} 逐类 ROC/PR", "path": f"../results/figures/{fn}"})

    # macro ROC / PR 对比（六模型调参 + dummy）
    roc_metric = {m: om42[(om42["model"] == m) & (om42["version"] == "tuned")]["roc_auc_macro"].iloc[0] for m in REAL_MODELS}
    roc_metric["dummy"] = om42[om42["model"] == "dummy"]["roc_auc_macro"].iloc[0]
    map_metric = {m: om42[(om42["model"] == m) & (om42["version"] == "tuned")]["mAP"].iloc[0] for m in REAL_MODELS}
    map_metric["dummy"] = om42[om42["model"] == "dummy"]["mAP"].iloc[0]
    plot_macro_roc_comparison(oof, roc_metric, str(fig_dir / "macro_roc.png"))
    plot_macro_pr_comparison(oof, map_metric, str(fig_dir / "macro_pr.png"))
    fig_paths["macro_roc"] = "../results/figures/macro_roc.png"
    fig_paths["macro_pr"] = "../results/figures/macro_pr.png"

    # 不确定性区间图（bootstrap 95%），Dummy 基线用虚线对比
    dummy_row = {k: float(om42[om42["model"] == "dummy"][k].iloc[0]) for k in ("accuracy", "f1_macro")}
    plot_uncertainty(unc["ci_table"], str(fig_dir / "uncertainty.png"),
                     metrics=("accuracy", "f1_macro"), baseline=dummy_row)
    fig_paths["uncertainty"] = "../results/figures/uncertainty.png"

    # 混淆矩阵：macro-F1 胜者 + accuracy 最高者
    conf_figs = []
    for m in dict.fromkeys([win["f1_best"], win["accuracy_best"]]):
        o = oof[(m, "tuned")]
        cm = compute_metrics(o["y_true"].to_numpy(),
                             o[[c for c in o.columns if c.startswith("score_")]].to_numpy(dtype=float),
                             o["y_pred"].to_numpy(), cfg["zero_division"])["confusion_matrix"]
        cm = np.array(cm)
        base = f"cm_{m}"
        plot_confusion_matrix(cm, cfg["class_names"], str(fig_dir / f"{base}_count.png"), title=f"{MODEL_LABELS[m]} 混淆矩阵（计数）")
        plot_confusion_matrix(cm, cfg["class_names"], str(fig_dir / f"{base}_norm.png"), normalize="true", title=f"{MODEL_LABELS[m]} 混淆矩阵（行归一化）")
        conf_figs.append({"title": f"{MODEL_LABELS[m]} 混淆矩阵（计数，合计 {int(cm.sum())}）", "path": f"../results/figures/{base}_count.png"})
        conf_figs.append({"title": f"{MODEL_LABELS[m]} 混淆矩阵（行归一化）", "path": f"../results/figures/{base}_norm.png"})

    print("[6/8] 最终模型重拟合 + 保存")
    final_model_name = win["f1_best"]
    final_pipe, best_params, best_inner_score, train_metrics, warn_msgs = _refit_final(
        cfg, X, y, final_model_name, cfg["seed"], int(cfg["n_jobs"])
    )
    import joblib

    final_model_dir = results_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, final_model_dir / "pipeline.joblib")
    joblib.dump(final_pipe, models_dir / "final_pipeline.joblib")
    final_meta = {
        "model": final_model_name,
        "model_label": MODEL_LABELS[final_model_name],
        "best_params": best_params,
        "inner_cv_best_f1_macro": float(best_inner_score),
        "train_metrics": train_metrics,
        "feature_order": cfg["feature_columns"],
        "label_map": cfg["label_map"],
        "class_names": cfg["class_names"],
        "score_kind": "decision_function" if final_model_name == "svm" else "probability",
        "fit_on": "全部 27 条样本",
        "note": "训练内调参后的最终演示模型；训练集得分不替代 OOF 性能。",
        "warnings": warn_msgs[:10],
    }
    _save_json(final_meta, final_model_dir / "metadata.json")
    _save_json(final_meta, models_dir / "final_metadata.json")

    print("[7/8] SHAP 解释")
    shap_result = explain_model(final_pipe, X, cfg["feature_columns"], cfg["class_names"], cfg)
    np.savez(final_model_dir / "shap_values.npz", values=shap_result["values"])
    shap_result["aggregate"].to_csv(final_model_dir / "shap_ranking.csv", index=False)
    _save_json(shap_result["meta"], final_model_dir / "shap_meta.json")
    plot_shap(shap_result["values"], cfg["feature_columns"], cfg["class_names"], str(fig_dir / "shap.png"))
    fig_paths["shap"] = "../results/figures/shap.png"

    print("[8/8] 生成报告")
    report_md = _build_report_dict_and_md(
        cfg, env, hashes, val, om42, fold_agg, main_run["res"]["budget"],
        folds_df, stability, win, fig_paths, per_class_figs, conf_figs,
        final_meta, shap_result, unc,
    )
    write_reports(report_md, report_dir / "实验报告.md", report_dir / "实验报告.html")

    _save_json({
        "winner": win,
        "final_model": final_model_name,
        "final_best_params": best_params,
        "elapsed_s": round(time.time() - t_start, 1),
    }, results_dir / "summary.json")

    print("=" * 70)
    print(f"完成。总耗时 {time.time() - t_start:.1f} s")
    print(f"胜者（macro-F1）：{MODEL_LABELS[final_model_name]}，OOF F1={win['f1_best_score']:.3f}")
    print(f"报告：{report_dir / '实验报告.md'} 与 {report_dir / '实验报告.html'}")


def _build_report_dict_and_md(cfg, env, hashes, val, om42, fold_agg, budget, folds_df,
                              stability, win, fig_paths, per_class_figs, conf_figs,
                              final_meta, shap_result, unc) -> str:
    from datetime import date

    class_names = cfg["class_names"]

    # OOF 汇总表（精简列）
    om_table = om42[["model", "version", "accuracy", "precision_macro", "recall_macro",
                     "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]].copy()
    om_table["model"] = om_table["model"].map(lambda m: MODEL_LABELS.get(m, m))
    om_table["version"] = om_table["version"].map(lambda v: VERSION_SHORT.get(v, v))
    om_table = om_table.rename(columns={"model": "模型", "version": "版本"})

    # 预算表
    bud_rows = []
    for m in REAL_MODELS:
        b = budget[m]
        bud_rows.append({"模型": MODEL_SHORT[m], "搜索类型": b["search_type"],
                         "空间大小": b["space_size"], "每折候选": b["candidates_per_outer_fold"],
                         "每折拟合数": b["fits_per_outer_fold"], "调参总拟合数": b["total_tuned_fits"]})
    budget_table = pd.DataFrame(bud_rows)

    # 最优参数（多数）
    best_param_rows = []
    for m in REAL_MODELS:
        subs = folds_df[(folds_df["model"] == m) & (folds_df["version"] == "tuned")]
        params = [json.dumps(f["best_params"], ensure_ascii=False, sort_keys=True) for f in subs.to_dict("records")]
        if params:
            most, cnt = Counter(params).most_common(1)[0]
            best_param_rows.append({"模型": MODEL_SHORT[m],
                                    "最优参数（多数）": json.loads(most),
                                    "出现折数": f"{cnt}/{len(params)}"})
    best_params_table = pd.DataFrame(best_param_rows) if best_param_rows else None

    # 训练 vs OOF
    tr_rows = []
    for m in REAL_MODELS:
        for version in ["untuned", "tuned"]:
            sub = folds_df[(folds_df["model"] == m) & (folds_df["version"] == version)]
            o = om42[(om42["model"] == m) & (om42["version"] == version)].iloc[0]
            tr_rows.append({"模型": MODEL_SHORT[m], "版本": VERSION_SHORT[version],
                            "train_f1_macro": np.mean(sub["train_f1_macro"]),
                            "train_accuracy": np.mean(sub["train_accuracy"]),
                            "oof_f1_macro": o["f1_macro"], "oof_accuracy": o["accuracy"]})
    train_oof_table = pd.DataFrame(tr_rows)

    # 调参变化文本
    delta_lines = []
    for m in REAL_MODELS:
        u = om42[(om42["model"] == m) & (om42["version"] == "untuned")].iloc[0]
        t = om42[(om42["model"] == m) & (om42["version"] == "tuned")].iloc[0]
        d = t["f1_macro"] - u["f1_macro"]
        direction = "提升" if d > 0 else ("下降" if d < 0 else "不变")
        delta_lines.append(f"- {MODEL_SHORT[m]}：macro-F1 {u['f1_macro']:.3f} → {t['f1_macro']:.3f}（{direction} {d:+.3f}）。")
    tuning_change = "\n".join(delta_lines)

    # 稳定性
    stab_table = stability[stability["version"] == "tuned"][["seed", "model", "accuracy", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]].copy()
    stab_table["model"] = stab_table["model"].map(lambda m: MODEL_SHORT.get(m, m))
    stab_table = stab_table.rename(columns={"seed": "种子", "model": "模型"})
    stability_text = "以下为各模型（调参版）在三个种子下的 OOF 指标，用于观察表现对切分的敏感性。重复切分不是新增独立样本，不能把 81 次预测当作 81 名患者来缩小置信区间。"

    # 胜者 / 混淆矩阵文本
    win_model_label = MODEL_SHORT[win["f1_best"]]
    acc_model_label = MODEL_SHORT[win["accuracy_best"]]
    confusion_text = (
        f"按预先确定的 macro-F1，胜者为 **{MODEL_LABELS[win['f1_best']]}**"
        f"（OOF macro-F1={win['f1_best_score']:.3f}，Accuracy={win['f1_best_accuracy']:.3f}）。"
        f"单独标记的 Accuracy 最高模型为 **{MODEL_LABELS[win['accuracy_best']]}**（Accuracy={win['accuracy_best_score']:.3f}）。"
    )
    if win["f1_best"] != win["accuracy_best"]:
        confusion_text += f" 两者不同：{win_model_label} 胜在 macro-F1 均衡，{acc_model_label} 胜在整体 Accuracy，已分别给出两者混淆矩阵。"

    # 逐类弱点：Codex 审查指出报告正文未充分解释「高危几乎识别不出」这一最要紧的事实
    _cm = np.array(
        om42[(om42["model"] == win["f1_best"]) & (om42["version"] == "tuned")].iloc[0]["confusion_matrix"],
        dtype=int,
    )
    _rec = [_cm[i, i] / _cm[i].sum() if _cm[i].sum() else 0.0 for i in range(3)]
    per_class_text = (
        f"按真实类别拆开看，胜者的逐类召回率为：低危 **{_rec[0]:.1%}**（{_cm[0, 0]}/{_cm[0].sum()}）、"
        f"中危 **{_rec[1]:.1%}**（{_cm[1, 1]}/{_cm[1].sum()}）、"
        f"高危 **仅 {_rec[2]:.1%}**（{_cm[2, 2]}/{_cm[2].sum()}）。\n\n"
        f"也就是说，{_cm[2].sum() - _cm[2, 2]} 条真实高危样本没有被识别为高危。"
        f"在三分类风险预警里，这比「哪个模型是冠军」重要得多：**模型对最需要预警的类别几乎失效**，"
        f"高分主要来自低危与中危两类。任何关于本实验「可用性」的表述都必须先说明这一点。"
        f"（这里评价的是给定标签下的实验表现，不推断真实疾病结局。）"
    )

    # 最终模型文本
    score_kind = final_meta["score_kind"]
    final_model_text = (
        f"最终演示模型为 **{final_meta['model_label']}**，在全部 27 条样本上重新做训练内 3 折调参后拟合。\n"
        f"- 最终超参数：`{final_meta['best_params']}`\n"
        f"- 训练内 CV 最优 macro-F1：{final_meta['inner_cv_best_f1_macro']:.3f}\n"
        f"- 训练集（全 27 条）Accuracy={final_meta['train_metrics']['accuracy']:.3f}，macro-F1={final_meta['train_metrics']['f1_macro']:.3f}"
        f"（训练集得分不得替代上面的 OOF 性能）。\n"
        f"- 分数类型：{'decision_function 决策分数（非概率）' if score_kind == 'decision_function' else 'predict_proba 概率'}。\n"
        f"- 已保存：`models/final_pipeline.joblib`（含预处理+模型），`models/final_metadata.json`（特征顺序与标签映射）。"
    )

    # 结论文本（关键问题）
    best_f1_sorted = om42[om42["version"] == "tuned"].sort_values("f1_macro", ascending=False)
    # 结论：明确「胜者只在部分指标胜出」+ 高危弱点 + 不把两策略之差当偏差估计
    _tuned = om42[om42["version"] == "tuned"]
    auc_best = _tuned.sort_values("roc_auc_macro", ascending=False, kind="stable").iloc[0]
    map_best = _tuned.sort_values("mAP", ascending=False, kind="stable").iloc[0]
    conclusion = (
        f"在当前 27 条样本、外层 5 折内层 3 折嵌套 CV 方案下，按**预先固定**的 macro-F1 规则，"
        f"胜者为 **{MODEL_LABELS[win['f1_best']]}**（OOF macro-F1={win['f1_best_score']:.4f}、"
        f"Accuracy={win['f1_best_accuracy']:.4f}）。\n\n"
        f"但必须同时说明三点：\n\n"
        f"1. **它只在部分指标上胜出。** macro ROC-AUC 与 mAP 最好的调参模型分别是 "
        f"{MODEL_SHORT[auc_best['model']]}（AUC={auc_best['roc_auc_macro']:.4f}）与 "
        f"{MODEL_SHORT[map_best['model']]}（mAP={map_best['mAP']:.4f}），"
        f"不能说随机森林在所有指标上都最优。\n"
        f"2. **胜者的分数含事后挑选的乐观性。** 它是在同一批折外预测上挑出的最高分；"
        f"各模型的嵌套评估不能自动成为「获胜模型选择过程」的无偏估计。"
        f"（报告 §4.11 另给出一个不含事后挑选的内层选择策略结果作为对照，"
        f"但那是**另一套策略**的表现，两者之差不可解释为偏差大小。）\n"
        f"3. **逐类看，最要紧的问题是高危类几乎识别不出**（召回率见 §4.8）。"
        f"整体 Accuracy 主要由低危与中危两类支撑。\n\n"
        f"调参对个别模型有提升（如随机森林、XGBoost），对部分模型反而下降，"
        f"反映出小样本下内层选择噪声大。受样本量限制，所有模型 OOF 指标波动较大，"
        f"不能据此声称临床部署有效或外部人群泛化良好。"
    )

    limitations = [
        "仅 27 条样本（每类 9 条），外层每折仅 5–6 条，OOF 指标方差大、置信范围宽；任何结论都是低统计功效下的初步观察。",
        "没有患者 ID，无法核实同一人是否存在重复录音跨行；本实验把‘每行独立’作为暂定假设，未声称已排除患者级泄漏。",
        "标准差是跨折变动，不是置信区间，也不是临床不确定性范围。",
        "获胜模型得分存在选择乐观性（在多个模型中挑最高分本身会高估该模型）。",
        "SHAP 为全量训练后描述性分析，不是外部验证解释，也不是因果关系证据。",
        "样本来源、特征单位与采集协议未提供，结果不可外推到其他人群或设备。",
    ]

    # 不确定性量化与选择乐观性（补充分析，只读已有 OOF，不改动主实验规则）
    ci_fmt = unc["ci_table"][unc["ci_table"]["metric"].isin(
        ["accuracy", "f1_macro", "roc_auc_macro", "mAP"])].copy()
    ci_fmt["OOF 点估计"] = ci_fmt["point"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    ci_fmt["bootstrap 95% 区间"] = ci_fmt.apply(
        lambda r: f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if pd.notna(r["ci_lo"]) else "NA", axis=1)
    ci_fmt = ci_fmt.rename(columns={"model_label": "模型", "metric": "指标",
                                    "n_boot": "重采样次数", "n_valid": "有效重采样数",
                                    "n_invalid": "缺类作废数"})
    ci_fmt = ci_fmt[["模型", "指标", "OOF 点估计", "bootstrap 95% 区间", "重采样次数",
                     "有效重采样数", "缺类作废数"]]

    def _ci_of(model: str, metric: str):
        r = unc["ci_table"][(unc["ci_table"]["model"] == model) & (unc["ci_table"]["metric"] == metric)]
        if r.empty:
            return None
        return float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0])

    ci_text = (
        "下表是对 27 条折外预测做**样本级 bootstrap 重采样（2000 次）**得到的百分位区间。"
        "它刻画的是「这 27 条样本」内部的抽样波动，**不是**人群抽样分布，也不能把种子 42/43/44 的 81 次重复预测当成 81 名患者。\n\n"
        "**有效抽样规则**：一次抽样只有**三类都出现**时才计入；缺任一类则该次对**全部指标**作废"
        "（计入「缺类作废数」）。不按剩余类别凑一个 macro 值、也不用 0 顶替——否则 macro 平均的类别数会在各次之间变化。"
        "本轮实际未出现缺类抽样（各模型「缺类作废数」均为 0），但规则与代码、测试一致。\n\n"
        "**适用范围（重要）**：本区间固定了已经拟合好的折外预测，只重采样行；它**没有**重新执行训练、调参、"
        "切分与胜者选择，也未建模交叉验证各折训练集之间的重叠。因此它**不是**「完整建模流程泛化性能」的已证 95% 置信区间。"
    )

    mc = unc["mcnemar"]
    if mc["p_value"] is not None:
        sig = "在 0.05 水平下**不显著**：以本样本量尚不能认为胜者优于先验基线。" if mc["p_value"] > 0.05 \
            else "在 0.05 水平下显著。"
        mc_p = f"精确 McNemar 双尾 p = **{mc['p_value']:.4f}**，{sig}"
    else:
        mc_p = "两者不一致对为 0，无法进行检验。"
    mcnemar_text = (
        f"胜者 **{MODEL_SHORT[mc['model_a']]}** 与 Dummy 先验基线使用同一组 27 条折外预测，可作配对比较："
        f"两者都判对 {mc['n_both_correct']} 条、都判错 {mc['n_both_wrong']} 条，"
        f"胜者判对而基线判错 {mc['n_a_only_correct']} 条、反之 {mc['n_b_only_correct']} 条。{mc_p} "
        f"不一致对仅 {mc['n_discordant']} 条，检验功效很低；p 不显著只表示「未观察到显著差异」，不等于「两者相同」。\n\n"
        f"**局限**：应把本检验视为**探索性比较**。「精确」仅指二项尾概率的算法精确——胜者本身来自同一批折外"
        f"预测的筛选、各折模型又共享训练数据，因此它不构成对整个「选模流程」的精确检验；且它只比较**错误率**，"
        f"不涉及 macro-F1、ROC-AUC 或排序质量。不宜把该 p 值当作强证据。"
    )

    sel = unc["selection"]
    picked_txt = "、".join(f"折 {p['fold']}→{MODEL_SHORT[p['model']]}" for p in sel["picked_per_fold"])
    sel_text = (
        f"这里换一种**不含事后挑选**的做法：每个外层折只按**该折的内层 macro-F1** 决定用哪个模型"
        f"（{picked_txt}），再拼接各折折外预测。结果为 Accuracy={sel['metrics']['accuracy']:.3f}、"
        f"macro-F1={sel['metrics']['f1_macro']:.3f}；作为对照，事后在同一批折外预测上挑出的 "
        f"**{MODEL_SHORT[sel['posthoc_winner']['model']]}** 为 Accuracy={sel['posthoc_winner']['accuracy']:.3f}、"
        f"macro-F1={sel['posthoc_winner']['f1_macro']:.3f}。\n\n"
        f"**这两个数值不能相减来解释为偏差的大小。** 它们来自**两套不同的选模策略**：前者外层实际用的是 "
        f"{'/'.join(sorted({MODEL_SHORT[p['model']] for p in sel['picked_per_fold']}))}，后者用的是 "
        f"{MODEL_SHORT[sel['posthoc_winner']['model']]}。"
        f"两者之差同时混合了(a)策略不同、(b)有限样本波动、(c)事后挑选的乐观性三种来源，无法单独识别 (c)，"
        f"更不能当作{MODEL_SHORT[sel['posthoc_winner']['model']]}经偏差校正后的成绩。\n\n"
        f"由此能得到的结论只有一条：**本次结果对「用哪种方式选模型」是敏感的**——内层选择策略在外层评估中"
        f"表现更差，且各折选出的模型并不固定在{MODEL_SHORT[sel['posthoc_winner']['model']]}上，说明当前样本量下模型排名本身不稳定。"
        f"该估计衡量的是「模型选择这个过程」的表现，仍不是独立外部测试集。"
    )

    # 把「统计功效低」这类定性说法换成具体数字，避免只声明不量化
    win_ci_acc = _ci_of(sel["posthoc_winner"]["model"], "accuracy")
    win_ci_f1 = _ci_of(sel["posthoc_winner"]["model"], "f1_macro")
    if win_ci_acc and win_ci_f1:
        limitations.append(
            f"胜者的 bootstrap 95% 区间很宽：Accuracy [{win_ci_acc[0]:.3f}, {win_ci_acc[1]:.3f}]、"
            f"macro-F1 [{win_ci_f1[0]:.3f}, {win_ci_f1[1]:.3f}]，各模型区间大量重叠，"
            f"不能按点估计大小排出可信的名次。"
        )
    if mc["p_value"] is not None:
        limitations.append(
            f"胜者相对 Dummy 先验基线的精确 McNemar 检验 p={mc['p_value']:.4f}（>0.05），"
            f"以本样本量尚不足以证明「有真实优于基线的判别能力」。"
        )
    limitations.append(
        f"内层选择策略（每折按内层 macro-F1 选模型）在外层评估中得 Accuracy={sel['metrics']['accuracy']:.3f}、"
        f"macro-F1={sel['metrics']['f1_macro']:.3f}，低于事后挑出的胜者分数。但这是**两套不同策略**的结果，"
        f"差值同时混合策略差异、有限样本波动与选择效应，**不能解释为选择偏差的大小**，"
        f"也不是胜者的偏差校正成绩；能说的只是结论对「用哪种方式选模型」敏感。"
    )
    limitations.append(
        "bootstrap 区间固定了已经拟合好的折外预测，只对 27 行做样本级重采样；它**没有**重新执行训练、调参、"
        "切分与胜者选择，也未建模交叉验证各折训练集之间的重叠。因此不能把它当作「完整建模流程泛化性能」"
        "的已证 95% 置信区间。本轮 2000 次重采样中未出现缺类抽样（见 results/uncertainty.csv 的 n_invalid 列），"
        "故区间数值未受缺类处理规则影响。"
    )
    limitations.append(
        "McNemar 检验的「精确」仅指二项尾概率的算法精确。由于胜者本身来自同一批折外预测的筛选、"
        "各折模型又共享训练数据，它不构成对整个选模流程的精确检验；它也比较错误率，"
        "不检验 macro-F1、AUC 或排序质量。应按**探索性比较**理解，其 p 值不宜作为强证据。"
    )

    R = {
        "config": cfg, "env": env, "input_hashes": hashes, "figures": fig_paths,
        "date": date.today().isoformat(),
        "oof_metrics_table": om_table,
        "fold_agg_table": fold_agg[["model", "version"] + KEY_METRICS].rename(
            columns={"model": "模型", "version": "版本"}
        ).assign(
            模型=lambda d: d["模型"].map(lambda m: MODEL_SHORT.get(m, m)),
            版本=lambda d: d["版本"].map(lambda v: VERSION_SHORT.get(v, v)),
        ),
        "budget_table": budget_table,
        "best_params_table": best_params_table,
        "train_oof_table": train_oof_table,
        "tuning_change_text": tuning_change,
        "stability_table": stab_table,
        "stability_text": stability_text,
        "unc_ci_table": ci_fmt,
        "unc_ci_text": ci_text,
        "unc_mcnemar_text": mcnemar_text,
        "unc_selection_text": sel_text,
        "per_class_figs": per_class_figs,
        "confusion_figs": conf_figs,
        "confusion_text": confusion_text,
        "per_class_text": per_class_text,
        "shap_meta": shap_result["meta"],
        "shap_agg_table": shap_result["aggregate"].rename(columns={
            "feature": "特征", "mean_abs_shap_overall": "mean(|SHAP|)", "rank": "排名"}),
        "final_model": final_meta,
        "final_model_text": final_model_text,
        "conclusion_text": conclusion,
        "limitations": limitations,
        "summary_text": conclusion,
    }
    return build_report(R)


if __name__ == "__main__":
    main()
