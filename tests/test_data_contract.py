"""数据契约测试：27×10、三类各 9、表头顺序、数值性、无重复、样本 ID 不进 X。"""
import numpy as np
import pytest

from src.config import load_config
from src.data import load_dataset, split_xy, validate_dataset

cfg = load_config()


@pytest.fixture(scope="module")
def loaded():
    df = load_dataset(cfg["input_xlsx"], cfg["feature_columns"])
    X, y, meta = split_xy(df, cfg["feature_columns"])
    return df, X, y, meta


def test_shape_27x10(loaded):
    df, X, y, meta = loaded
    assert df.shape[0] == 27
    assert X.shape == (27, 10)


def test_class_counts(loaded):
    df, X, y, meta = loaded
    assert sorted(np.bincount(y).tolist()) == [9, 9, 9]


def test_feature_header_exact(loaded):
    df, X, y, meta = loaded
    assert list(X.columns) == cfg["feature_columns"]
    assert "FO" in cfg["feature_columns"]  # 字母 O 保留，未改成 F0
    assert "F0" not in cfg["feature_columns"]


def test_all_numeric_no_missing(loaded):
    df, X, y, meta = loaded
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_no_duplicate_feature_rows(loaded):
    df, X, y, meta = loaded
    assert X.duplicated().sum() == 0


def test_sample_id_not_in_X(loaded):
    df, X, y, meta = loaded
    assert "sample_id" not in X.columns
    assert "label" not in X.columns
    assert "source_sheet" not in X.columns
    assert "excel_row" not in X.columns


def test_validate_report_ok(loaded):
    df, X, y, meta = loaded
    rep = validate_dataset(df, cfg["feature_columns"])
    assert rep["ok"] is True, rep["errors"]
    assert rep["checks"]["n_samples_27"] is True
    assert rep["checks"]["feature_columns_present"] is True


def test_label_map_fixed():
    assert cfg["label_map"] == {"低危": 0, "中危": 1, "高危": 2}


def test_unknown_sheet_name_raises():
    from src.data import _label_from_sheet_name

    with pytest.raises(ValueError):
        _label_from_sheet_name("汇总  某某  声  ")
    with pytest.raises(ValueError):
        _label_from_sheet_name("高危中危")
