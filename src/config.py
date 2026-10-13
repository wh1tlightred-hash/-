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
    # 补充分析（交叉验证方案对比、置换检验、学习/验证曲线、校准、扩展 SHAP）
    "analysis": {"cv_schemes": {"kfold": [3, 5, 9], "loo": True, "repeated": {"n_splits": 5, "n_repeats": 10}}, "permutation": {"n_permutations": 100, "cv_splits": 5}, "learning_curve": {"per_class_sizes": [2, 3, 4, 5, 6, 7], "n_splits": 5, "n_repeats": 3}, "validation_curve": {"n_splits": 5, "n_repeats": 3}, "calibration_bins": 3, "shap_all_models": True, "shap_cv": True, "shap_dependence_top_k": 4},
    "report": {"formats": ["md", "html", "docx"], "html_embed_images": True},
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
