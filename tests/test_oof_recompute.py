"""OOF 重算一致性：从保存的 OOF 文件重算指标与报告 OOF 汇总一致；27 条；混淆矩阵合计 27；分数有限且概率行和≈1。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, ROOT
from src.metrics import compute_metrics

cfg = load_config()
SEED_DIR = Path(cfg["results_dir"]) / f"seed{cfg['seed']}"


@pytest.fixture(scope="module")
def oof_metrics_csv():
    p = SEED_DIR / "oof_metrics.csv"
    if not p.exists():
        pytest.skip("未找到 oof_metrics.csv（需先运行 main.py）")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def oof_files():
    oof_dir = SEED_DIR / "oof"
    if not oof_dir.exists():
        pytest.skip("未找到 OOF 目录")
    return oof_dir


def _load_oof(oof_dir, model, version):
    p = oof_dir / f"{model}_{version}_oof.csv"
    assert p.exists(), f"缺少 OOF 文件 {p}"
    return pd.read_csv(p)


def test_all_models_present(oof_files):
    from src.models import REAL_MODELS

    for m in REAL_MODELS:
        for v in ["untuned", "tuned"]:
            assert (oof_files / f"{m}_{v}_oof.csv").exists()
    assert (oof_files / "dummy_untrained_oof.csv").exists()


def test_oof_count_27(oof_metrics_csv, oof_files):
    for _, row in oof_metrics_csv.iterrows():
        df = _load_oof(oof_files, row["model"], row["version"])
        assert len(df) == 27, f"{row['model']}/{row['version']} OOF 数量 {len(df)} != 27"


def test_recompute_matches_summary(oof_metrics_csv, oof_files):
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "roc_auc_macro", "mAP", "pr_auc_macro"]
    for _, row in oof_metrics_csv.iterrows():
        df = _load_oof(oof_files, row["model"], row["version"])
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        score_cols = [c for c in df.columns if c.startswith("score_")]
        y_score = df[score_cols].to_numpy(dtype=float)
        m = compute_metrics(y_true, y_score, y_pred, zero_division=cfg["zero_division"])
        for k in keys:
            expected = row[k]
            if pd.isna(expected):
                continue
            assert np.isclose(m[k], expected, atol=1e-9), \
                f"{row['model']}/{row['version']} 指标 {k} 重算 {m[k]:.6f} != 报告 {expected:.6f}"


def test_confusion_matrix_sums_to_27(oof_metrics_csv, oof_files):
    for _, row in oof_metrics_csv.iterrows():
        df = _load_oof(oof_files, row["model"], row["version"])
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        score_cols = [c for c in df.columns if c.startswith("score_")]
        m = compute_metrics(y_true, df[score_cols].to_numpy(dtype=float), y_pred, cfg["zero_division"])
        assert int(np.sum(m["confusion_matrix"])) == 27


def test_scores_finite_and_rowsums(oof_metrics_csv, oof_files):
    for _, row in oof_metrics_csv.iterrows():
        df = _load_oof(oof_files, row["model"], row["version"])
        score_cols = [c for c in df.columns if c.startswith("score_")]
        y_score = df[score_cols].to_numpy(dtype=float)
        assert np.isfinite(y_score).all(), f"{row['model']}/{row['version']} 分数非有限"
        # 概率模型行和≈1；SVM 为决策分数，跳过
        if row["model"] != "svm":
            rowsums = y_score.sum(axis=1)
            assert np.allclose(rowsums, 1.0, atol=1e-6), \
                f"{row['model']}/{row['version']} 概率行和不接近 1"


def test_score_column_order(oof_files):
    df = _load_oof(oof_files, "logistic", "tuned")
    score_cols = [c for c in df.columns if c.startswith("score_")]
    assert score_cols[0].startswith("score_0_")
    assert score_cols[1].startswith("score_1_")
    assert score_cols[2].startswith("score_2_")
