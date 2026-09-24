"""错误输入测试：predict.validate_and_reorder 对缺列、多列、非数值明确报错；乱序列安全重排。"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from predict import validate_and_reorder

cfg = load_config()
FEATURES = cfg["feature_columns"]


def _df(cols, values=None):
    values = values or [[float(i + j) for j in range(len(cols))] for i in range(2)]
    return pd.DataFrame(values, columns=cols)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="缺少特征列"):
        validate_and_reorder(_df(FEATURES[:-1]), FEATURES)


def test_extra_column_raises():
    cols = FEATURES + ["extra"]
    with pytest.raises(ValueError, match="多余列"):
        validate_and_reorder(_df(cols), FEATURES)


def test_non_numeric_raises():
    df = pd.DataFrame([{f: (1.0 if f != "FO" else "abc") for f in FEATURES}])
    with pytest.raises(ValueError, match="非数值"):
        validate_and_reorder(df, FEATURES)


def test_infinite_raises():
    df = pd.DataFrame([{f: (1.0 if f != "FO" else float("inf")) for f in FEATURES}])
    with pytest.raises(ValueError, match="无穷"):
        validate_and_reorder(df, FEATURES)


def test_reorder_shuffled_columns():
    import random

    shuffled = FEATURES.copy()
    random.Random(0).shuffle(shuffled)
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], columns=shuffled)
    X = validate_and_reorder(df, FEATURES)
    assert X.shape == (1, 10)
    # 重排后列序与 FEATURES 一致
    assert list(df[FEATURES].iloc[0].values) == list(X[0])


def test_empty_raises():
    with pytest.raises(ValueError):
        validate_and_reorder(pd.DataFrame(), FEATURES)
