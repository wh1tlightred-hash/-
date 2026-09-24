"""配置加载与路径工具。所有可调参数集中在项目根目录 config.json，不在代码中重复硬编码。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

_DEFAULTS: dict = {
    "seed": 42,
    "stability_seeds": [43, 44],
    "outer_cv": {"n_splits": 5, "shuffle": True},
    "inner_cv": {"n_splits": 3, "shuffle": True},
    "inner_scoring": "f1_macro",
    "max_candidates_per_model": 20,
    "n_iter_random": 20,
    "n_jobs": 1,
    "feature_columns": ["FO", "I", "F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4"],
    "label_map": {"低危": 0, "中危": 1, "高危": 2},
    "class_names": ["低危", "中危", "高危"],
    "input_xlsx": "inputs/声诊练习.xlsx",
    "input_docx": "inputs/中医声诊实验.docx",
    "data_processed_dir": "data/processed",
    "results_dir": "results",
    "models_dir": "models",
    "report_dir": "report",
    "shap": {"background": "all_train", "nsamples_kernel": 400, "explain_all_train": True},
    "zero_division": 0,
}


def load_config(path: Path | str | None = None) -> dict:
    """读取配置；文件缺失时回退到默认值。返回合并后的 dict。"""
    cfg = json.loads(json.dumps(_DEFAULTS))  # 深拷贝默认值
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user = json.load(f)
        _deep_update(cfg, user)
    return cfg


def _deep_update(base: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def resolve(cfg: dict, key: str) -> Path:
    """将相对路径解析到项目根目录。"""
    return ROOT / cfg[key]
