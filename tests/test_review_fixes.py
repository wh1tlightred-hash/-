"""针对 Codex 审查意见的修复回归测试。

每个测试对应审查报告里的具体一条，便于把「改了什么」和「怎么验证」绑在一起：

- bootstrap 缺类抽样：审查指出文档承诺「缺类作废」但实现按 nanmean 继续计入。
- macro 指标维度：审查指出缺类时 macro 平均的类别数会变。
- SHAP 解释空间守卫：审查指出换模型/加变换后 TreeExplainer 的空间假设会失效。
- 搜索预算硬上限：审查指出 max_candidates_per_model 原先只是描述性字段。
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval import validate_candidate_budget
from src.explain import explain_model
from src.metrics import compute_metrics
from src.uncertainty import bootstrap_ci


# ---------- bootstrap 缺类抽样 ----------

def test_bootstrap_rejects_draws_missing_a_class():
    """审查的边界例：三类各 2 条、seed 42、100 次抽样 -> 21 次缺类，应只有 79 次有效。

    修复前实际记为 100 次有效（nanmean 在剩余类别上凑出了 macro 值）。
    """
    y = np.array([0, 0, 1, 1, 2, 2])
    score = np.eye(3)[y]
    rng = np.random.default_rng(42)
    missing = int(sum(len(np.unique(y[r])) < 3 for r in rng.integers(0, 6, size=(100, 6))))

    res = bootstrap_ci(y, y, score, n_boot=100, seed=42)

    assert missing == 21, "边界例本身变了，请重新核对审查数据"
    assert res["roc_auc_macro"]["n_valid"] == 100 - missing == 79
    assert res["roc_auc_macro"]["n_invalid"] == 21
    # 有效抽样数必须对所有指标一致，否则各指标的区间不可比
    assert len({v["n_valid"] for v in res.values()}) == 1
    assert len({v["n_invalid"] for v in res.values()}) == 1


def test_bootstrap_counts_add_up():
    y = np.repeat([0, 1, 2], 9)
    score = np.eye(3)[y] + 0.01
    res = bootstrap_ci(y, y, score, n_boot=300, seed=7)
    for v in res.values():
        assert v["n_valid"] + v["n_invalid"] == v["n_boot"]


# ---------- macro 指标维度固定 ----------

def test_macro_metrics_keep_three_class_dimension():
    """缺类时 macro 平均仍按三类平均（zero_division=0），不因缺类改变分母。"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # 类别 2 完全没出现
    m = compute_metrics(y_true, np.eye(3)[y_true], y_pred, zero_division=0)
    # 逐类 precision: 类0=1.0、类1=1.0、类2=0.0 -> macro=(1+1+0)/3
    assert m["precision_macro"] == pytest.approx(2 / 3)
    assert m["recall_macro"] == pytest.approx(2 / 3)
    assert len(m["per_class_precision"]) == 3
    assert len(m["support"]) == 3


# ---------- SHAP 解释空间守卫 ----------

def _pipe(with_scaler: bool) -> Pipeline:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 4))
    y = np.tile([0, 1, 2], 8)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", RandomForestClassifier(n_estimators=5, random_state=0)))
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def test_explain_model_rejects_scaler_before_classifier():
    """分类器前有 scaler 时，对原始特征做树 SHAP 会解释错空间，必须直接报错。"""
    pipe = _pipe(with_scaler=True)
    X = pd.DataFrame(np.zeros((2, 4)), columns=list("abcd"))
    with pytest.raises(ValueError, match="scaler"):
        explain_model(pipe, X, list("abcd"), ["低危", "中危", "高危"], {"shap": {}})


def test_explain_model_records_identity_preprocessing():
    pipe = _pipe(with_scaler=False)
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(6, 4)), columns=list("abcd"))
    res = explain_model(pipe, X, list("abcd"), ["低危", "中危", "高危"], {"shap": {}})
    assert res["meta"]["preprocessing_is_identity"] is True
    assert "恒等" in res["meta"]["background"]
    assert res["values"].shape == (6, 4, 3)


# ---------- 搜索预算硬上限 ----------

def test_candidate_budget_cap_enforced():
    cap = {"max_candidates_per_model": 20}
    validate_candidate_budget({"svm": {"candidates_per_outer_fold": 20}}, cap)  # 等于上限，允许
    with pytest.raises(ValueError, match="max_candidates_per_model"):
        validate_candidate_budget({"svm": {"candidates_per_outer_fold": 25}}, cap)


def test_candidate_budget_cap_disabled_when_zero():
    validate_candidate_budget({"svm": {"candidates_per_outer_fold": 999}},
                              {"max_candidates_per_model": 0})


def test_candidate_budget_ignores_dummy_entry():
    """Dummy 基线没有搜索预算（candidates=0），不应被当作超限。"""
    budget = {"dummy": {"candidates_per_outer_fold": 0},
              "gnb": {"candidates_per_outer_fold": 5}}
    with pytest.raises(ValueError, match="gnb"):
        validate_candidate_budget(budget, {"max_candidates_per_model": 4})
    validate_candidate_budget({"dummy": {"candidates_per_outer_fold": 0}},
                              {"max_candidates_per_model": 4})  # 只有 dummy -> 通过
