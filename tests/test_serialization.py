"""序列化与预测一致性：保存后重载预测一致；最终模型可加载并输出有限分数。"""
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.config import load_config, ROOT
from src.data import load_dataset, split_xy
from src.models import build_pipeline

cfg = load_config()


def test_pipeline_roundtrip_consistent():
    df = load_dataset(cfg["input_xlsx"], cfg["feature_columns"])
    X, y, meta = split_xy(df, cfg["feature_columns"])
    pipe = build_pipeline("logistic", seed=42, n_jobs=1)
    pipe.fit(X.to_numpy(dtype=float), y)
    pred_before = pipe.predict(X.to_numpy(dtype=float))
    proba_before = pipe.predict_proba(X.to_numpy(dtype=float))
    # 用项目内自管临时目录，避开 pytest tmp_path 在 Windows + Py3.14 下的清理问题
    tmpdir = Path(tempfile.mkdtemp(prefix="roundtrip_", dir=str(ROOT / "results")))
    try:
        p = tmpdir / "pipe.joblib"
        joblib.dump(pipe, p)
        pipe2 = joblib.load(p)
        assert np.array_equal(pipe2.predict(X.to_numpy(dtype=float)), pred_before)
        assert np.allclose(pipe2.predict_proba(X.to_numpy(dtype=float)), proba_before)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_final_model_loads_and_predicts():
    model_path = ROOT / cfg["models_dir"] / "final_pipeline.joblib"
    meta_path = ROOT / cfg["models_dir"] / "final_metadata.json"
    if not model_path.exists():
        pytest.skip("未找到最终模型（需先运行 main.py）")
    df = load_dataset(cfg["input_xlsx"], cfg["feature_columns"])
    X, y, meta = split_xy(df, cfg["feature_columns"])
    pipe = joblib.load(model_path)
    y_pred = pipe.predict(X.to_numpy(dtype=float))
    assert y_pred.shape == (27,)
    assert set(np.unique(y_pred)).issubset({0, 1, 2})
    from src.metrics import class_scores

    scores = class_scores(pipe, X.to_numpy(dtype=float))
    assert scores.shape == (27, 3)
    assert np.isfinite(scores).all()
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    assert m["feature_order"] == cfg["feature_columns"]
    assert m["label_map"] == cfg["label_map"]
    if m["score_kind"] == "probability":
        assert np.allclose(scores.sum(axis=1), 1.0, atol=1e-6)
