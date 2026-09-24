"""新样本预测入口：加载已保存的最终模型管线，对输入样本输出类别与分数。

用法：
    .venv/Scripts/python.exe predict.py --csv 新样本.csv
    .venv/Scripts/python.exe predict.py --json '[{"FO":358.6,"I":46.2,"F1":897.7,"F2":1737.7,"F3":3025.7,"F4":4095.2,"B1":458.9,"B2":256.0,"B3":484.5,"B4":603.0}]'

输入只含 10 个同名特征（FO, I, F1..F4, B1..B4）；缺列、多列、非数值会明确报错；
列顺序不同会按名称安全重排。输出类别名/编码与三类分数（概率或 SVM 决策分数，非疾病发生概率）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import class_scores  # noqa: E402

MODELS_DIR = ROOT / "models"


def load_model(models_dir: Path = MODELS_DIR) -> tuple:
    pipe_path = models_dir / "final_pipeline.joblib"
    meta_path = models_dir / "final_metadata.json"
    if not pipe_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{pipe_path}（请先运行 main.py 完成训练）。")
    pipeline = joblib.load(pipe_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return pipeline, meta


def validate_and_reorder(df: pd.DataFrame, feature_order: list[str]) -> np.ndarray:
    """校验 10 个特征列并安全重排。缺列/多列/非数值均报错。"""
    cols = list(df.columns)
    if df.empty:
        raise ValueError("预测输入不能为空。")
    if df.columns.duplicated().any():
        raise ValueError("预测输入含重复特征列名。")
    missing = [c for c in feature_order if c not in cols]
    extra = [c for c in cols if c not in feature_order]
    if missing:
        raise ValueError(f"缺少特征列：{missing}。需要 {feature_order}。")
    if extra:
        raise ValueError(f"存在多余列：{extra}。只允许 {feature_order}。")
    X = df[feature_order].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"特征 {bad} 含非数值或缺失值。")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("预测输入包含无穷值。")
    return X.to_numpy(dtype=float)


def predict_new(pipeline, X: np.ndarray, meta: dict) -> pd.DataFrame:
    y_pred = pipeline.predict(X)
    scores = class_scores(pipeline, X)
    class_names = meta.get("class_names", ["低危", "中危", "高危"])
    label_map = meta.get("label_map", {"低危": 0, "中危": 1, "高危": 2})
    score_kind = meta.get("score_kind", "probability")
    rows = []
    for i in range(len(X)):
        code = int(y_pred[i])
        name = class_names[code] if 0 <= code < len(class_names) else str(code)
        rec = {"predicted_label": name, "predicted_code": code}
        for c, cn in enumerate(class_names):
            rec[f"score_{cn}"] = float(scores[i, c])
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(f"# 分数类型：{'decision_function 决策分数（非概率，未校准）' if score_kind == 'decision_function' else 'predict_proba 概率'}")
    return out


def main(argv=None) -> int:
    """返回退出码：0 正常，2 输入/模型文件错误（给出可读提示，不抛裸 traceback）。"""
    p = argparse.ArgumentParser(description="中医声诊三分类新样本预测")
    p.add_argument("--csv", help="输入 CSV 路径（含 10 个特征列）")
    p.add_argument("--json", help="内联 JSON 样本列表")
    p.add_argument("--out", help="结果输出 CSV 路径（可选）")
    args = p.parse_args(argv)

    try:
        pipeline, meta = load_model()
        feature_order = meta.get(
            "feature_order",
            ["FO", "I", "F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4"],
        )

        if args.csv:
            df = pd.read_csv(args.csv)
        elif args.json:
            df = pd.DataFrame(json.loads(args.json))
        else:
            print("请用 --csv 或 --json 提供输入。示例见文件顶部 docstring。")
            return 0

        X = validate_and_reorder(df, feature_order)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as e:
        print(f"输入错误：{e}", file=sys.stderr)
        print("提示：输入必须只含同名的 10 个特征列，且全为有限数值。", file=sys.stderr)
        return 2

    out = predict_new(pipeline, X, meta)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"已保存到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
