"""不确定性量化与「含模型选择」的诚实估计（补充分析，不替代预先固定的主结果）。

原报告只**定性**声明「27 条样本统计功效低」，没有给出任何数值化的不确定性。
本模块用已有的 OOF 产物补上三件可核验的事：

1. ``bootstrap_ci``            —— 对 OOF 预测做**样本级** bootstrap 重采样，给出各模型指标的百分位区间。
                                  注意适用范围：这是对「这 27 条样本」的重采样，不是人群抽样，
                                  也不能把种子 42/43/44 的 81 次重复预测当成 81 名患者。
2. ``mcnemar_exact``          —— 胜者 vs Dummy 基线的**配对**比较（精确二项检验）。
                                  回答「排名第一是否只是噪声」这一报告未回答的问题。
3. ``nested_selection_estimate`` —— 每个外层折按**内层分数**选模型，再拼接各折折外预测。
                                  得到「含模型选择」这一**策略**的表现，用来与事后挑选的结果做对照。
                                  注意：它不是「选择偏差」的估计值——两套策略之差同时混合了
                                  策略差异、样本波动与选择效应，无法分离出偏差本身。

注意：本模块**不**声称解决了模型选择偏差。它只把报告里那句「存在选择乐观性」
限定到有据可依的范围内（见报告 §4.11 与 §6.1）。

全部计算只读已落盘的 OOF / folds 产物，不重新训练、不改动主实验的任何规则。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .models import REAL_MODELS

# 参与不确定性量化的指标（与报告主表一致，便于对照）
CI_METRICS = ("accuracy", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro")


def score_matrix_from_oof(df: pd.DataFrame) -> np.ndarray:
    """从 OOF DataFrame 取出 (n, 3) 分数矩阵，列序即 score_0/1/2（低危/中危/高危）。"""
    return df[[c for c in df.columns if c.startswith("score_")]].to_numpy(dtype=float)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metric_keys: tuple[str, ...] = CI_METRICS,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    zero_division: int = 0,
) -> dict[str, dict]:
    """对 OOF 预测做样本级 bootstrap，返回每个指标的 {point, lo, hi, n_valid, n_invalid}。

    point 是原始 27 条上的取值（与报告主表一致，不因重采样而变化）；
    lo/hi 是 bootstrap 分布的 alpha/2 与 1-alpha/2 百分位。

    有效抽样的定义（代码与报告、测试必须一致）：**一次抽样只有在三类都出现时才计入**。
    理由：macro 指标是「三类一起平均」，若某类缺失，该类 AUC/AP/PR-AUC 无从计算，
    `nanmean` 会在更少的类别上取均值（维度不一致）；若只让部分指标作废，
    各指标的 n_valid 又会各不相同、无法对齐。因此缺类抽样对**全部指标**一律作废，
    计入 n_invalid，不用 0 顶替，也不用「剩余类别」凑一个数。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("bootstrap 需要至少 1 条 OOF 预测。")

    point = compute_metrics(y_true, y_score, y_pred, zero_division=zero_division)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    draws: dict[str, list[float]] = {k: [] for k in metric_keys}
    n_invalid = 0
    for row in idx:
        yt_row = y_true[row]
        if len(np.unique(yt_row)) < 3:  # 缺任一类 -> 整次抽样作废
            n_invalid += 1
            continue
        try:
            m = compute_metrics(yt_row, y_score[row], y_pred[row], zero_division=zero_division)
        except ValueError:  # 例如子集内分数全同导致曲线无法计算
            n_invalid += 1
            continue
        for k in metric_keys:
            v = m.get(k)
            draws[k].append(np.nan if v is None else float(v))

    out: dict[str, dict] = {}
    for k in metric_keys:
        arr = np.asarray(draws[k], dtype=float)
        valid = arr[np.isfinite(arr)]
        lo = hi = np.nan
        if valid.size:
            lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        pv = point.get(k)
        out[k] = {
            "point": None if pv is None else float(pv),
            "lo": None if not np.isfinite(lo) else float(lo),
            "hi": None if not np.isfinite(hi) else float(hi),
            "n_valid": int(valid.size),
            "n_boot": int(n_boot),
            "n_invalid": int(n_invalid),
        }
    return out


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """配对 McNemar 精确检验：模型 A（胜者）相对模型 B（Dummy 基线）是否正确得更频繁。

    只看两者**判断不一致**的样本对（b = A 对/B 错，c = A 错/B 对），
    在「两者等同」的原假设下 b 服从 Binomial(b+c, 0.5)。样本量小，用精确检验而非卡方。
    返回配对表与双尾 p 值；不一致对为 0 时 p 记为 NA。
    """
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int) == y_true
    b = np.asarray(pred_b, dtype=int) == y_true
    n_both = int((a & b).sum())
    n_a_only = int((a & ~b).sum())
    n_b_only = int((~a & b).sum())
    n_neither = int((~a & ~b).sum())
    n_disc = n_a_only + n_b_only
    p = None
    if n_disc > 0:
        from scipy.stats import binomtest

        p = float(binomtest(n_a_only, n_disc, 0.5).pvalue)
    return {
        "n_both_correct": n_both,
        "n_a_only_correct": n_a_only,
        "n_b_only_correct": n_b_only,
        "n_both_wrong": n_neither,
        "n_discordant": n_disc,
        "p_value": p,
        "note": "精确二项检验（双尾）；不一致对很少时检验功效很低，p 不显著只说明「未观察到显著差异」。",
    }


def nested_selection_estimate(
    folds: pd.DataFrame,
    oof: dict[tuple[str, str], pd.DataFrame],
    zero_division: int = 0,
    model_names: list[str] | None = None,
) -> dict:
    """「含模型选择」的诚实估计：每折按内层 macro-F1 选模型，拼接各折折外预测。

    主实验的胜者是在**同一批 OOF 预测**上挑出来的最高分，因此该分数含选择乐观性。
    这里换一种做法：外层第 i 折只看第 i 折的**内层分数**决定用哪个模型，
    再用该模型在第 i 折的折外预测，拼成 27 条完整预测后计算指标。
    模型选择完全发生在折外预测之前，不含事后挑选。

    注意：这仍不是独立外部测试集；它衡量的是「模型选择这个过程」的表现，而非某个模型的泛化上限。
    """
    models = model_names or REAL_MODELS
    tuned = folds[(folds["version"] == "tuned") & (folds["model"].isin(models))]
    if tuned.empty:
        raise ValueError("folds 中没有可用的 tuned 记录，无法做含选择的估计。")

    picked: list[dict] = []
    parts: list[pd.DataFrame] = []
    for fold_i, sub in tuned.groupby("fold"):
        if sub["best_inner_score"].isna().all():
            raise ValueError(f"第 {fold_i} 折缺少内层分数，无法按内层分数选模型。")
        best = sub.loc[sub["best_inner_score"].idxmax()]
        model = str(best["model"])
        o = oof[(model, "tuned")]
        parts.append(o[o["fold"] == fold_i])
        picked.append({
            "fold": int(fold_i),
            "model": model,
            "best_inner_score": float(best["best_inner_score"]),
        })

    spliced = pd.concat(parts).sort_values("sample_id").reset_index(drop=True)
    if len(spliced) != len(oof[(models[0], "tuned")]):
        raise ValueError(
            f"拼接后的折外预测数量 {len(spliced)} 与单模型 OOF 数量 "
            f"{len(oof[(models[0], 'tuned')])} 不一致。"
        )
    m = compute_metrics(
        spliced["y_true"].to_numpy(dtype=int),
        score_matrix_from_oof(spliced),
        spliced["y_pred"].to_numpy(dtype=int),
        zero_division=zero_division,
    )
    return {"picked_per_fold": picked, "n_samples": int(len(spliced)), "metrics": m}
