"""数据加载与契约校验。

输入附件 inputs/声诊练习.xlsx 含三个工作表，表名含「高危 / 中危 / 低危」关键词，
每表在 C1:L1 有 10 个特征表头（FO, I, F1..F4, B1..B4），C2:L10 为 9 行数值样本。

要点（见 PROJECT_BRIEF.md 第 2 节）：
- 原表头写的是字母 O 的 ``FO``，不能静默改成数字 0 的 ``F0``。
- 类别来自工作表名称关键词，不是原始特征列；未知或歧义表名必须报错，不按表序猜标签。
- 固定标签映射 低危=0，中危=1，高危=2；训练用一维整数 y。
- 保留原表名与原始 Excel 行号以便追溯；样本 ID、来源、行号绝不进入特征矩阵 X。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

# 数据区：表头在 C1:L1，数据在 C2:L10（openpyxl 的列索引从 1 开始）
_FIRST_COL = 3  # C
_LAST_COL = 12  # L
_HEADER_ROW = 1
_FIRST_DATA_ROW = 2
_LAST_DATA_ROW = 10

_LABEL_KEYWORDS = ("高危", "中危", "低危")
_LABEL_CODE = {"低危": 0, "中危": 1, "高危": 2}


def _label_from_sheet_name(sheet_name: str) -> tuple[str, int]:
    """从工作表名唯一匹配类别关键词。歧义或缺失时抛错，绝不按表序猜测。"""
    normalized = re.sub(r"\s+", "", sheet_name or "")
    hits = [k for k in _LABEL_KEYWORDS if k in normalized]
    if len(hits) == 0:
        raise ValueError(
            f"工作表名 {sheet_name!r} 未命中任何类别关键词 {_LABEL_KEYWORDS}，请人工核对。"
        )
    if len(hits) > 1:
        raise ValueError(
            f"工作表名 {sheet_name!r} 同时命中多个类别关键词 {hits}，存在歧义，请人工核对。"
        )
    label_name = hits[0]
    return label_name, _LABEL_CODE[label_name]


def _read_sheet(ws, sheet_name: str) -> tuple[str, int, list[list[float]], list[int]]:
    """读取单个工作表的数据区，返回 (类别名, 类别码, 特征行列表, 原始行号列表)。"""
    header = [ws.cell(row=_HEADER_ROW, column=c).value for c in range(_FIRST_COL, _LAST_COL + 1)]
    label_name, label_code = _label_from_sheet_name(sheet_name)

    rows: list[list[float]] = []
    row_numbers: list[int] = []
    for r in range(_FIRST_DATA_ROW, ws.max_row + 1):
        values = [ws.cell(row=r, column=c).value for c in range(_FIRST_COL, _LAST_COL + 1)]
        # 若该行 10 个格子全为空，视为空行跳过（高危表第 11 行为空行，不读取也不报错）。
        if all(v is None or (isinstance(v, str) and v.strip() == "") for v in values):
            continue
        rows.append(values)
        row_numbers.append(r)
    return label_name, label_code, rows, row_numbers, header


def load_dataset(
    xlsx_path: str | Path,
    feature_columns: list[str] | None = None,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """读取 xlsx，返回合并后的长表 DataFrame。

    列：sample_id, label_name, label, source_sheet, excel_row, 10 个特征列。
    特征列使用特征原始名 FO, I, F1..F4, B1..B4。
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")

    wb = load_workbook(path, data_only=True, read_only=True)
    expected_features = feature_columns or ["FO", "I", "F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4"]
    label_codes = label_map or _LABEL_CODE
    if label_codes != _LABEL_CODE:
        raise ValueError(f"标签映射必须固定为 {_LABEL_CODE}，实际为 {label_codes}。")

    records: list[dict] = []
    seen_labels: list[str] = []
    for sheet_name in wb.sheetnames:
        label_name, label_code, rows, row_numbers, header = _read_sheet(wb[sheet_name], sheet_name)
        if label_name not in label_codes:
            raise ValueError(f"表名 {sheet_name!r} 对应的类别 {label_name!r} 不在配置的 label_map 中。")
        # 校验表头与期望特征列一致（按顺序）
        if header != expected_features:
            raise ValueError(
                f"表 {sheet_name!r} 表头 {header} 与期望特征列 {expected_features} 不一致。"
            )
        seen_labels.append(label_name)
        for row_values, excel_row in zip(rows, row_numbers):
            rec = {
                "label_name": label_name,
                "label": label_code,
                "source_sheet": sheet_name,
                "excel_row": excel_row,
            }
            rec.update({f: v for f, v in zip(expected_features, row_values)})
            records.append(rec)

    df = pd.DataFrame(records)
    wb.close()
    if df.empty:
        raise ValueError("输入工作簿没有有效样本。")
    # 样本 ID：类别名 + 原始 Excel 行号，稳定且可追溯；仅存元数据，不进特征矩阵。
    df.insert(0, "sample_id", df["label_name"].astype(str) + "_r" + df["excel_row"].astype(str))
    return df


def split_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """拆分为特征矩阵 X、标签 y（一维整数）与元数据 meta。"""
    meta = df[["sample_id", "label_name", "label", "source_sheet", "excel_row"]].reset_index(drop=True)
    X = df[feature_columns].copy()
    y = df["label"].to_numpy(dtype=int)
    return X, y, meta


def validate_dataset(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """数据契约校验，返回结构化报告 dict。

    校验项：27×10 形状、三类各 9 条、特征表头顺序、无非数值、无缺失、无完全重复的十维特征行。
    """
    report: dict = {"ok": True, "errors": [], "warnings": [], "checks": {}}
    n_rows, n_cols = df.shape

    # 特征列数 = 10
    feat_present = [c for c in feature_columns if c in df.columns]
    report["checks"]["feature_columns_present"] = feat_present == feature_columns
    if feat_present != feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        report["ok"] = False
        report["errors"].append(f"缺少特征列：{missing}")

    # 有效样本 27 条
    report["checks"]["n_samples_27"] = n_rows == 27
    if n_rows != 27:
        report["ok"] = False
        report["errors"].append(f"期望 27 条样本，实际 {n_rows} 条。")

    # 三类各 9 条
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index().to_dict()
        report["checks"]["class_counts"] = counts
        expected = {0: 9, 1: 9, 2: 9}
        if counts != expected:
            report["ok"] = False
            report["errors"].append(f"类别计数 {counts} 与期望 {expected} 不一致。")
    else:
        report["ok"] = False
        report["errors"].append("缺少 label 列。")

    # 数值性与缺失
    if feat_present:
        X = df[feature_columns]
        numeric = X.apply(pd.to_numeric, errors="coerce")
        is_all_numeric = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
        report["checks"]["all_numeric"] = bool(is_all_numeric)
        if not is_all_numeric:
            report["ok"] = False
            report["errors"].append("特征区存在非数值、缺失值或无穷值。")

        # 完全重复的十维特征行
        dup = X.duplicated().sum()
        report["checks"]["duplicate_feature_rows"] = int(dup)
        if dup > 0:
            report["warnings"].append(f"存在 {dup} 行完全重复的十维特征行，需人工核对是否同一患者重复录音。")

    # 极值提示（不做删除，只报告）
    if feat_present:
        X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        zscore = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
        extreme = (zscore.abs() > 3.0).any(axis=1).sum()
        report["checks"]["n_extreme_rows_z3"] = int(extreme)
        if extreme:
            report["warnings"].append(
                f"有 {extreme} 行样本存在 |z|>3 的极值特征，保留样本并核对，不因影响得分而删除。"
            )

    report["ok"] = bool(report["ok"]) and not report["errors"]
    return report


def file_sha256(path: str | Path) -> str:
    """计算文件 SHA-256，用于核对原始附件未被修改。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
