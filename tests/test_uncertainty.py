"""补充分析模块 test_uncertainty：bootstrap 区间、配对检验、含模型选择的估计。

部分用例只用合成数据（不依赖 main.py 产物），因此未跑主实验也能全量执行。
"""
import numpy as np
import pandas as pd
import pytest

from src.uncertainty import (
    bootstrap_ci,
    mcnemar_exact,
    nested_selection_estimate,
    score_matrix_from_oof,
)

RNG = np.random.default_rng(7)


def _toy(n=30):
    """构造三分类合成 OOF：y_true 与分数列序一致，预测较好但非完美。"""
    y = np.repeat([0, 1, 2], n // 3)
    score = np.full((len(y), 3), 0.1)
    score[np.arange(len(y)), y] = 0.8
    # 让其中约 1/4 的样本被预测错
    flip = RNG.choice(len(y), size=len(y) // 4, replace=False)
    score[flip] = np.roll(score[flip], 1, axis=1)
    y_pred = score.argmax(axis=1)
    return y, y_pred, score


# ---------- bootstrap_ci ----------

def test_bootstrap_point_matches_plain_metrics():
    y, p, s = _toy()
    ci = bootstrap_ci(y, p, s, seed=42, n_boot=200)
    from src.metrics import compute_metrics

    m = compute_metrics(y, s, p, zero_division=0)
    assert ci["accuracy"]["point"] == pytest.approx(m["accuracy"])
    assert ci["f1_macro"]["point"] == pytest.approx(m["f1_macro"])


def test_bootstrap_is_deterministic_for_same_seed():
    y, p, s = _toy()
    a = bootstrap_ci(y, p, s, seed=123, n_boot=200)
    b = bootstrap_ci(y, p, s, seed=123, n_boot=200)
    assert a == b


def test_bootstrap_interval_brackets_point_estimate():
    y, p, s = _toy()
    ci = bootstrap_ci(y, p, s, seed=42, n_boot=500)
    for key in ("accuracy", "f1_macro"):
        assert ci[key]["lo"] <= ci[key]["point"] <= ci[key]["hi"], key


def test_bootstrap_counts_valid_resamples():
    y, p, s = _toy()
    ci = bootstrap_ci(y, p, s, seed=42, n_boot=300)
    # 合成数据每类都有 10 条，重采样几乎不会缺类，但仍必须报出有效次数
    assert 0 < ci["roc_auc_macro"]["n_valid"] <= 300
    assert ci["accuracy"]["n_valid"] == 300  # 准确率任何子集都能算


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError, match="至少"):
        bootstrap_ci(np.array([]), np.array([]), np.zeros((0, 3)))


# ---------- mcnemar_exact ----------

def test_mcnemar_known_case():
    """A 全对、B 全错，4 条样本 -> 不一致对 4，精确双尾 p = 2 * 0.5^4 = 0.125。"""
    y = np.array([0, 0, 1, 1])
    res = mcnemar_exact(y, np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    assert res["n_a_only_correct"] == 4
    assert res["n_b_only_correct"] == 0
    assert res["n_discordant"] == 4
    assert res["p_value"] == pytest.approx(0.125)


def test_mcnemar_identical_predictions_gives_na():
    y = np.array([0, 1, 2, 0])
    pred = np.array([0, 1, 2, 1])
    res = mcnemar_exact(y, pred, pred)
    assert res["n_discordant"] == 0
    assert res["p_value"] is None


def test_mcnemar_counts_partition_all_samples():
    y, p, _ = _toy()
    other = np.roll(p, 1)
    res = mcnemar_exact(y, p, other)
    total = (res["n_both_correct"] + res["n_a_only_correct"]
             + res["n_b_only_correct"] + res["n_both_wrong"])
    assert total == len(y)


# ---------- nested_selection_estimate ----------

def _toy_folds_and_oof():
    """两折、两个模型；折 0 的 inner 分数 gnb 更高，折 1 的 logistic 更高。"""
    folds = pd.DataFrame([
        {"fold": 0, "model": "gnb", "version": "tuned", "best_inner_score": 0.9},
        {"fold": 0, "model": "logistic", "version": "tuned", "best_inner_score": 0.2},
        {"fold": 1, "model": "gnb", "version": "tuned", "best_inner_score": 0.1},
        {"fold": 1, "model": "logistic", "version": "tuned", "best_inner_score": 0.8},
    ])
    oof = {}
    for model in ("gnb", "logistic"):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        fold = np.array([0, 0, 0, 1, 1, 1])
        # gnb 折 0 全对、折 1 全错；logistic 相反
        if model == "gnb":
            y_pred = np.array([0, 1, 2, 1, 2, 0])
        else:
            y_pred = np.array([1, 2, 0, 0, 1, 2])
        score = np.zeros((6, 3))
        score[np.arange(6), y_pred] = 0.9
        score[np.arange(6), y_true] = np.where(y_pred == y_true, 0.9, 0.05)
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(6)],
            "y_true": y_true, "y_pred": y_pred, "fold": fold,
        })
        for c in range(3):
            df[f"score_{c}_c{c}"] = score[:, c]
        oof[(model, "tuned")] = df
    return folds, oof


def test_nested_selection_picks_by_inner_score_per_fold():
    folds, oof = _toy_folds_and_oof()
    res = nested_selection_estimate(folds, oof, model_names=["gnb", "logistic"])
    picked = {p["fold"]: p["model"] for p in res["picked_per_fold"]}
    assert picked == {0: "gnb", 1: "logistic"}
    # 每折都取了「该折内层分数更高」的模型，而它在该折恰好全对
    assert res["n_samples"] == 6
    assert res["metrics"]["accuracy"] == pytest.approx(1.0)


def test_nested_selection_raises_without_inner_scores():
    folds, oof = _toy_folds_and_oof()
    folds = folds.assign(best_inner_score=np.nan)
    with pytest.raises(ValueError, match="内层分数"):
        nested_selection_estimate(folds, oof, model_names=["gnb", "logistic"])


def test_nested_selection_raises_on_incomplete_coverage():
    """若某折没有任何 tuned 记录，拼接数量会少，必须报错而不是静默给出偏小样本。"""
    folds, oof = _toy_folds_and_oof()
    folds = folds[folds["fold"] == 0]
    with pytest.raises(ValueError, match="不一致"):
        nested_selection_estimate(folds, oof, model_names=["gnb", "logistic"])


# ---------- score_matrix_from_oof ----------

def test_score_matrix_keeps_column_order():
    df = pd.DataFrame({
        "sample_id": ["a", "b"],
        "score_0_低危": [0.1, 0.2], "score_1_中危": [0.2, 0.3], "score_2_高危": [0.7, 0.5],
    })
    m = score_matrix_from_oof(df)
    assert m.shape == (2, 3)
    assert m[0].tolist() == [0.1, 0.2, 0.7]


# ---------- 与绘图/表格的字段契约 ----------

def _ci_table_like_main(n_models=3):
    """复刻 main._uncertainty 生成的 uncertainty.csv 列结构（这是与绘图的接口契约）。"""
    rows = []
    for i in range(n_models):
        y, p, s = _toy()
        ci = bootstrap_ci(y, p, s, seed=42 + i, n_boot=50)
        for k, v in ci.items():
            rows.append({
                "model": f"m{i}", "model_label": f"模型{i}", "metric": k,
                "point": v["point"], "ci_lo": v["lo"], "ci_hi": v["hi"],
                "n_boot": v["n_boot"], "n_valid": v["n_valid"],
            })
    return pd.DataFrame(rows)


def test_ci_table_columns_match_plot_contract(tmp_path):
    """plot_uncertainty 依赖 point/ci_lo/ci_hi 列名；列名改动必须在这里失败。"""
    from src.plot import plot_uncertainty

    tbl = _ci_table_like_main()
    for col in ("model", "model_label", "metric", "point", "ci_lo", "ci_hi"):
        assert col in tbl.columns, f"缺少绘图所需列：{col}"
    out = tmp_path / "uncertainty.png"
    plot_uncertainty(tbl, str(out), metrics=("accuracy", "f1_macro"),
                     baseline={"accuracy": 0.25, "f1_macro": 0.22})
    assert out.exists() and out.stat().st_size > 0


def test_plot_uncertainty_tolerates_nan_interval(tmp_path):
    """某指标区间可能为 NA（重采样子集全缺类），绘图不应崩。"""
    from src.plot import plot_uncertainty

    tbl = _ci_table_like_main(n_models=2)
    tbl.loc[0, "ci_lo"] = np.nan
    tbl.loc[0, "ci_hi"] = np.nan
    out = tmp_path / "u2.png"
    plot_uncertainty(tbl, str(out), metrics=("accuracy",))
    assert out.exists()


def test_plot_uncertainty_skips_absent_metric(tmp_path):
    from src.plot import plot_uncertainty

    tbl = _ci_table_like_main(n_models=2)
    out = tmp_path / "u3.png"
    plot_uncertainty(tbl, str(out), metrics=("accuracy", "not_a_metric"))
    assert out.exists()


# ---------- 与 main 的接线一致性 ----------

def test_uncertainty_table_helpers_accept_real_artifacts():
    """若主实验产物已存在，用真实 OOF 跑一遍三个函数，确保接口没被改坏。"""
    from pathlib import Path

    from src.config import load_config

    cfg = load_config()
    oof_dir = Path(cfg["results_dir"]) / f"seed{cfg['seed']}" / "oof"
    if not oof_dir.exists():
        pytest.skip("未找到 OOF 产物（需先运行 main.py）")

    rf = pd.read_csv(oof_dir / "random_forest_tuned_oof.csv")
    dummy = pd.read_csv(oof_dir / "dummy_untrained_oof.csv")
    ci = bootstrap_ci(rf["y_true"].to_numpy(), rf["y_pred"].to_numpy(),
                      score_matrix_from_oof(rf), n_boot=100)
    assert ci["accuracy"]["point"] == pytest.approx((rf["y_pred"] == rf["y_true"]).mean())
    mc = mcnemar_exact(rf["y_true"].to_numpy(), rf["y_pred"].to_numpy(), dummy["y_pred"].to_numpy())
    assert mc["n_discordant"] == mc["n_a_only_correct"] + mc["n_b_only_correct"]
