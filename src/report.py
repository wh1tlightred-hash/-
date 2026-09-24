"""生成中文实验报告（Markdown 与 HTML），图表用相对路径引用，便于本机打开。"""
from __future__ import annotations

from pathlib import Path

import markdown as md_lib


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _df_md(df, float_cols=None, nd=3):
    """将 DataFrame 转为 Markdown 表格字符串。"""
    if df is None or len(df) == 0:
        return "（无数据）\n"
    float_cols = set(float_cols or [])
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols:
                cells.append(_fmt(v, nd))
            else:
                cells.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _list_md(items):
    if not items:
        return "（无）\n"
    return "\n".join(f"- {i}" for i in items) + "\n"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>中医声诊三分类实验报告</title>
<style>
body{{font-family:"Microsoft YaHei","SimHei",sans-serif;max-width:1000px;margin:24px auto;padding:0 20px;line-height:1.7;color:#222;}}
h1{{border-bottom:3px solid #3182bd;padding-bottom:8px;}}
h2{{border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:32px;color:#1a4a73;}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px;}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:center;}}
th{{background:#eef3f8;}}
img{{max-width:100%;border:1px solid #eee;margin:8px 0;}}
code{{background:#f4f4f4;padding:1px 5px;border-radius:3px;}}
pre{{background:#f4f4f4;padding:10px;border-radius:5px;overflow-x:auto;}}
blockquote{{border-left:4px solid #3182bd;margin:12px 0;padding:4px 16px;color:#555;background:#f7fafc;}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def write_reports(report_md: str, md_path: str | Path, html_path: str | Path) -> None:
    md_path = Path(md_path)
    html_path = Path(html_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_md, encoding="utf-8")
    body = md_lib.markdown(report_md, extensions=["tables", "fenced_code", "sane_lists"])
    html_path.write_text(_HTML_TEMPLATE.format(body=body), encoding="utf-8")


def build_report(R: dict) -> str:
    """根据汇总字典 R 生成完整中文报告 Markdown 字符串。"""
    cfg = R["config"]
    env = R["env"]
    fig = R.get("figures", {})
    L = []
    A = L.append

    A("# 中医声诊样本风险三分类实验报告")
    A("")
    A(f"> 生成日期：{R.get('date', '')}。本报告全部指标来自真实运行（嵌套交叉验证外层折外预测）。")
    A("")

    # ---------- 摘要 ----------
    A("## 1 摘要")
    A("")
    A(R.get("summary_text", ""))
    A("")

    # ---------- 数据与设置 ----------
    A("## 2 数据与实验设置")
    A("")
    A(f"- 输入附件：`{cfg['input_xlsx']}`（27 条有效样本 × 10 个数值特征）、`{cfg['input_docx']}`（实验背景参考）。")
    A(f"- 特征列（表头原样保留，`FO` 为字母 O，未改作 `F0`）：{', '.join(cfg['feature_columns'])}。")
    A(f"- 类别映射（固定）：低危=0，中危=1，高危=2；三类各 9 条。")
    A(f"- 切分：外层 5 折、内层 3 折分层嵌套交叉验证，`StratifiedKFold(shuffle=True, random_state=42)`；所有模型共享同一组外层索引。")
    A(f"- 调参：内层主评分固定 macro-F1；LR/SVM/GNB/KNN 用 GridSearchCV，RF/XGBoost 用 RandomizedSearchCV（每折 ≤ {cfg['max_candidates_per_model']} 候选）；所有拟合型预处理均位于 Pipeline 内、只在训练折内拟合。")
    A(f"- 指标：Accuracy、macro/weighted/逐类 Precision、Recall、F1、OvR ROC-AUC（逐类+macro）、mAP（逐类 AP 算术平均）、macro PR-AUC（PR 曲线梯形积分，与 AP 算法不同）。")
    A(f"- 补充：种子 43、44 各完整重复嵌套流程，考察切分稳定性；主结果仍用 42。")
    A("")
    if fig.get("class_counts"):
        A(f"![类别样本数]({fig['class_counts']})")
        A("")
    A(f"原始附件 SHA-256（确认未修改）：")
    A(f"- `{cfg['input_xlsx']}`：`{R['input_hashes'].get('xlsx')}`")
    A(f"- `{cfg['input_docx']}`：`{R['input_hashes'].get('docx')}`")
    A("")

    # ---------- 环境 ----------
    A("## 3 环境核验")
    A("")
    A("- 操作系统 / 架构：" + env.get("platform", "") + " / " + env.get("arch", ""))
    A("- 解释器：" + env.get("python", "") + "，路径 `" + env.get("executable", "") + "`")
    A("- `python -m pip check`：" + env.get("pip_check", ""))
    A("- 关键包版本：")
    A(_list_md([f"{k}=={v}" for k, v in sorted(env.get("packages", {}).items())]))
    A("")

    # ---------- 结果 ----------
    A("## 4 结果")
    A("")
    A("### 4.1 OOF 汇总指标（27 条折外预测，种子 42）")
    A("")
    A("下表为从 OOF 预测重新计算的完整汇总值（区别于逐折均值的平均）。")
    A("")
    A(_df_md(R["oof_metrics_table"],
             float_cols=["accuracy", "precision_macro", "recall_macro", "f1_macro",
                         "roc_auc_macro", "mAP", "pr_auc_macro"]))
    A("")
    A("*注：SVM 的 ROC/PR 使用 decision_function 决策分数（非概率、未做校准），不参与概率行和≈1 的检查；其余模型为 predict_proba 概率。*")
    A("")

    if fig.get("model_metrics"):
        A(f"![模型指标对比]({fig['model_metrics']})")
        A("")

    A("### 4.2 逐折指标（均值 ± 标准差，跨 5 个外层折）")
    A("")
    A("标准差是跨折变动，不是置信区间。")
    A("")
    A(_df_md(R["fold_agg_table"], float_cols=["mean", "std"]))
    A("")

    A("### 4.3 调参搜索预算")
    A("")
    A(_df_md(R["budget_table"], float_cols=[]))
    A("")

    A("### 4.4 调参前后对比")
    A("")
    A(R.get("tuning_change_text", ""))
    A("")
    if R.get("best_params_table") is not None:
        A("各模型外层折最优超参数（取多数出现或逐折记录见 `results/seed42/folds.csv`）：")
        A("")
        A(_df_md(R["best_params_table"]))
        A("")

    A("### 4.5 训练分数与外层分数差距（过拟合观察）")
    A("")
    A(_df_md(R["train_oof_table"], float_cols=["train_f1_macro", "train_accuracy", "oof_f1_macro", "oof_accuracy"]))
    A("")

    A("### 4.6 六模型曲线对比（OOF 分数）")
    A("")
    if fig.get("macro_roc"):
        A(f"![六模型 macro ROC 对比]({fig['macro_roc']})")
        A("")
    if fig.get("macro_pr"):
        A(f"![六模型 macro PR 对比]({fig['macro_pr']})")
        A("")
    A("*读图口径（四点，容易误读）：*")
    A("")
    A("1. 图例中的 macro-AUC 来自**逐类 ROC-AUC 算术平均**、mAP 来自**逐类 AP 算术平均**，"
      "两者都**不是**平均曲线本身的面积，不要互相替代。")
    A("2. macro PR 曲线按「**相同召回率取最大精确率**」去重后再线性插值到公共 recall 网格。"
      "该做法会消除垂直段、使曲线偏高，因此平均曲线下方的面积**大于** mAP 与梯形 PR-AUC"
      "（三者算法不同：mAP=逐类 AP 平均、pr_auc_macro=逐类梯形积分、此图=插值后平均曲线）。"
      "把这里的 max 直接改成 min 也不是通用正确修复；换展示口径需另行说明。")
    A("3. 曲线由**跨折拼接**的 OOF 分数绘制。Dummy 每折 AUC≈0.5，但拼接后约 0.42，"
      "这是不同训练折先验分数之间的排序效应，**不应**把该曲线当作理论随机基线，也不代表 AUC 实现有误。")
    A("4. SVM 的分数是 decision_function 决策分数，跨折可能存在尺度差异，故同时保留逐折指标。")
    A("")

    A("### 4.7 逐类 ROC / PR 曲线")
    A("")
    for entry in R.get("per_class_figs", []):
        A(f"![{entry['title']}]({entry['path']})")
        A("")

    A("### 4.8 混淆矩阵")
    A("")
    A(R.get("confusion_text", ""))
    A("")
    A(R.get("per_class_text", ""))
    A("")
    for entry in R.get("confusion_figs", []):
        A(f"![{entry['title']}]({entry['path']})")
        A("")

    A("### 4.9 SHAP 特征重要性（训练后描述性分析）")
    A("")
    A(f"- 解释对象：最终所选模型（{R['final_model'].get('model_label','')}）。")
    A(f"- 解释器：{R['shap_meta'].get('explainer','')}；背景：{R['shap_meta'].get('background','')}；解释对象：{R['shap_meta'].get('explained','')}；输出空间：{R['shap_meta'].get('output_space','')}。")
    A(f"- 说明：{R['shap_meta'].get('note','')}")
    A("")
    if fig.get("shap"):
        A(f"![SHAP 特征重要性]({fig['shap']})")
        A("")
    A("三类聚合 mean(|SHAP|) 排名：")
    A("")
    A(_df_md(R["shap_agg_table"], float_cols=["mean_abs_shap_overall"]))
    A("")

    A("### 4.10 切分稳定性（种子 43、44 补充）")
    A("")
    A(R.get("stability_text", ""))
    A("")
    A(_df_md(R["stability_table"], float_cols=["accuracy", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]))
    A("")

    A("### 4.11 不确定性量化与选择乐观性（补充分析）")
    A("")
    A("本节为**补充分析**，不改动 §2 预先固定的主实验规则（随机种子、外层/内层切分、内层评分、胜者判定规则全部不变），"
      "只对已经算出的折外预测做进一步的统计刻画。")
    A("")
    A(R.get("unc_ci_text", ""))
    A("")
    A(_df_md(R.get("unc_ci_table")))
    A("")
    if fig.get("uncertainty"):
        A(f"![OOF 指标 bootstrap 95% 区间]({fig['uncertainty']})")
        A("")
    A("**胜者与先验基线的配对检验**")
    A("")
    A(R.get("unc_mcnemar_text", ""))
    A("")
    A("**含模型选择的无事后挑选估计**")
    A("")
    A(R.get("unc_selection_text", ""))
    A("")

    # ---------- 最终模型 ----------
    A("## 5 最终模型")
    A("")
    A(R.get("final_model_text", ""))
    A("")

    # ---------- 结论与限制 ----------
    A("## 6 结论与限制")
    A("")
    A(R.get("conclusion_text", ""))
    A("")
    A("### 6.1 关键限制")
    A("")
    A(_list_md(R.get("limitations", [])))
    A("")

    A("---")
    A("")
    A("报告由 `main.py` 自动生成；复现命令见 `README.md`。所有原始附件保持不变。")
    A("")
    return "\n".join(L)
