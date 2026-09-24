"""切分泄漏与索引完整性测试：外层测试折互不重叠、覆盖全部样本、每折含三类；内层训练/验证不交叉。"""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import load_config, ROOT
from src.data import load_dataset, split_xy

cfg = load_config()


@pytest.fixture(scope="module")
def data():
    df = load_dataset(cfg["input_xlsx"], cfg["feature_columns"])
    return split_xy(df, cfg["feature_columns"])


def test_outer_folds_disjoint_and_cover(data):
    X, y, meta = data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.asarray(X, dtype=float), y))
    assert len(splits) == 5
    all_test = np.concatenate([te for _, te in splits])
    assert len(all_test) == 27
    assert len(np.unique(all_test)) == 27  # 互不重叠且覆盖全部
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0  # 训练/测试不交叉


def test_each_outer_fold_has_three_classes(data):
    X, y, meta = data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(np.asarray(X, dtype=float), y):
        assert set(np.unique(y[te])) == {0, 1, 2}


def test_inner_train_validation_disjoint(data):
    X, y, meta = data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, te in skf.split(np.asarray(X, dtype=float), y):
        for a, b in inner.split(np.asarray(X)[tr], y[tr]):
            assert len(np.intersect1d(tr[a], tr[b])) == 0
            assert set(tr[a]).issubset(set(tr)) and set(tr[b]).issubset(set(tr))


def test_saved_splits_audit(data):
    """若 main.py 已运行，核对落盘的 splits.json 与外层/内层切分一致。"""
    X, y, meta = data
    audit = Path(cfg["results_dir"]) / "search_audit" / f"seed_{cfg['seed']}" / "splits.json"
    if not audit.exists():
        pytest.skip("未找到 splits.json（需先运行 main.py）")
    rec = json.loads(audit.read_text(encoding="utf-8"))
    assert rec["sample_ids"] == meta["sample_id"].tolist()
    folds = rec["folds"]
    assert len(folds) == 5
    tests = [set(f["test"]) for f in folds]
    assert all(len(t & s) == 0 for i, t in enumerate(tests) for s in tests[i + 1:])
    assert set().union(*tests) == set(range(27))
    for f in folds:
        assert set(np.unique(y[list(f["test"])])) == {0, 1, 2}
        for inn in f["inner"]:
            assert set(inn["train"]).isdisjoint(set(inn["validation"]))
            assert set(inn["train"]).issubset(set(f["train"]))
            assert set(inn["validation"]).issubset(set(f["train"]))
