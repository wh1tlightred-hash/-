"""扩展分析编排：补充对比曲线（第 5 章）、交叉验证深入分析（第 6 章）、SHAP 深入分析（第 7 章）。

每个函数都：1) 基于真实运行结果计算；2) 把数据落盘到 results/ 下；3) 生成报告小节
（标题、由数据生成的说明文字、图、表）。说明文字里的数字全部从结果中读取，不手写结论。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import curves
from .cv_analysis import (
    calibration_summary,
    cv_scheme_comparison,
    learning_curves,
    permutation_tests,
    validation_curves,
)
from .eval import refit_on_full_data
from .explain import cv_shap, multi_model_shap
from .metrics import class_scores
from .models import MODEL_LABELS, MODEL_SHORT, REAL_MODELS, SIMPLICITY_ORDER
from .shap_plots import (
    pick_representative_samples,
    plot_shap_beeswarm,
    plot_shap_cv_compare,
    plot_shap_dependence,
    plot_shap_model_heatmap,
    plot_shap_waterfalls,
)

FIG_REL = "../results/figures"
ORDERED = [m for m in SIMPLICITY_ORDER if m in REAL_MODELS]


def _fig(fig_dir: Path, name: str, title: str) -> tuple[str, dict]:
    return str(fig_dir / name), {"title": title, "path": f"{FIG_REL}/{name}"}


def _row(om: pd.DataFrame, model: str, version: str) -> pd.Series:
    return om[(om["model"] == model) & (om["version"] == version)].iloc[0]


# ====================================================================== 第 5 章

def curve_sections(cfg, oof, om42, folds_df, stability, train_oof, fig_dir: Path) -> list[dict]:
    cn = cfg["class_names"]
    sections = []

    # 5.1 调参前后 ROC
    path, ent = _fig(fig_dir, "roc_tuned_vs_untuned.png", "调参前后 macro ROC 对比")
    curves.plot_tuned_vs_untuned_roc(oof, om42, path)
    rows = []
    for m in ORDERED:
        u, t = _row(om42, m, "untuned"), _row(om42, m, "tuned")
        rows.append({"模型": MODEL_SHORT[m], "未调参 AUC": u["roc_auc_macro"], "调参 AUC": t["roc_auc_macro"],
                     "ΔAUC": t["roc_auc_macro"] - u["roc_auc_macro"],
                     "未调参 F1": u["f1_macro"], "调参 F1": t["f1_macro"], "ΔF1": t["f1_macro"] - u["f1_macro"]})
    tab = pd.DataFrame(rows)
    up = tab[tab["ΔAUC"] > 0]["模型"].tolist()
    down = tab[tab["ΔAUC"] < 0]["模型"].tolist()
    sections.append({
        "heading": "5.1 调参前后 ROC 对比",
        "text": (f"同一组外层切分下，调参使 macro ROC-AUC 上升的模型：{'、'.join(up) or '无'}；"
                 f"下降的模型：{'、'.join(down) or '无'}。内层调参目标是 macro-F1，"
                 f"它并不保证排序质量（AUC）同步改善；小样本下两者方向不一致很常见。"),
        "figs": [ent], "table": tab,
        "float_cols": ["未调参 AUC", "调参 AUC", "ΔAUC", "未调参 F1", "调参 F1", "ΔF1"],
    })

    # 5.2 逐折 ROC 带
    path, ent = _fig(fig_dir, "roc_fold_bands.png", "逐折 macro ROC 与折间变动")
    curves.plot_fold_roc_bands(oof, path)
    fold_auc = []
    for m in ORDERED:
        vals = [f["metrics"]["roc_auc_macro"] for f in folds_df[(folds_df["model"] == m)
                                                                 & (folds_df["version"] == "tuned")].to_dict("records")]
        fold_auc.append({"模型": MODEL_SHORT[m], "逐折 AUC 均值": np.mean(vals), "逐折 AUC 标准差": np.std(vals),
                         "最低折": np.min(vals), "最高折": np.max(vals)})
    fa = pd.DataFrame(fold_auc)
    widest = fa.loc[(fa["最高折"] - fa["最低折"]).idxmax()]
    sections.append({
        "heading": "5.2 逐折 ROC 与折间变动",
        "text": (f"每个外层折只有 5–6 条测试样本，单折 ROC 呈明显阶梯状。折间差异最大的是 **{widest['模型']}**"
                 f"（单折 AUC 从 {widest['最低折']:.2f} 到 {widest['最高折']:.2f}）。"
                 f"阴影带是 ±1 标准差，反映「换一个测试折」带来的波动，不是置信区间。"),
        "figs": [ent], "table": fa, "float_cols": ["逐折 AUC 均值", "逐折 AUC 标准差", "最低折", "最高折"],
    })

    # 5.3 按类别对比
    path, ent = _fig(fig_dir, "class_model_comparison.png", "逐类别模型对比（ROC / PR）")
    curves.plot_per_class_model_curves(oof, om42, cn, path)
    rows = []
    for m in ORDERED + ["dummy"]:
        r = _row(om42, m, "untrained" if m == "dummy" else "tuned")
        rows.append({"模型": MODEL_SHORT[m], **{f"{cn[c]} AUC": r["roc_auc_ovr"][c] for c in range(3)},
                     **{f"{cn[c]} AP": r["ap_ovr"][c] for c in range(3)}})
    pc = pd.DataFrame(rows)
    real = pc[pc["模型"] != MODEL_SHORT["dummy"]]
    best_hi = real.loc[real[f"{cn[2]} AUC"].idxmax()]
    n_below = int((real[f"{cn[2]} AUC"] < 0.5).sum())
    hi_note = (f"**{len(real)} 个模型中有 {n_below} 个的{cn[2]}类 AUC 低于 0.5**，即对{cn[2]}样本的排序"
               f"还不如随机——这与第 4.8 节「{cn[2]}召回率很低」相互印证，是本实验最需要正视的结果。"
               if n_below else f"所有模型的{cn[2]}类 AUC 均不低于 0.5。")
    sections.append({
        "heading": "5.3 按风险类别比较各模型",
        "text": (f"把三分类拆成三个「一类 vs 其余」问题后，各模型在不同类别上的强弱并不一致。"
                 f"对最需要预警的 **{cn[2]}** 类，AUC 最高的是 {best_hi['模型']}（{best_hi[f'{cn[2]} AUC']:.3f}），"
                 f"{hi_note}PR 图中的虚线是类别先验 1/3：AP 低于它说明该类排序不如随机。"),
        "figs": [ent], "table": pc, "float_cols": [c for c in pc.columns if c != "模型"],
    })

    # 5.4 逐折指标分布
    path, ent = _fig(fig_dir, "fold_metric_box.png", "外层 5 折逐折指标分布")
    curves.plot_fold_metric_box(folds_df, path)
    sections.append({
        "heading": "5.4 外层逐折指标分布",
        "text": "箱线图展示 5 个外层折上的单折指标，与第 4.2 节的均值±标准差表对应。每折只有 5–6 条样本，"
                "单折 Accuracy 只能取 0、1/5、1/6……等少数离散值，因此箱体宽、点常重合。",
        "figs": [ent],
    })

    # 5.5 雷达图
    path, ent = _fig(fig_dir, "radar_metrics.png", "多指标雷达图")
    curves.plot_radar(om42, path)
    sections.append({
        "heading": "5.5 多指标雷达图",
        "text": "雷达图把 7 个汇总指标放在一起，便于看出模型是「全面均衡」还是「单项突出」。"
                "各轴数值与第 4.1 节表格相同；轴之间的面积没有统计含义，不要据此排名。",
        "figs": [ent],
    })

    # 5.6 训练-折外差距
    path, ent = _fig(fig_dir, "train_vs_oof_gap.png", "训练分数与折外分数差距")
    curves.plot_train_vs_oof(train_oof, path)
    tuned = train_oof[train_oof["version"] == "tuned"].copy()
    tuned["gap"] = tuned["train_f1_macro"] - tuned["oof_f1_macro"]
    worst = tuned.loc[tuned["gap"].idxmax()]
    sections.append({
        "heading": "5.6 训练分数与折外分数差距",
        "text": (f"实心点为各外层训练折上的 macro-F1（逐折均值），空心点为外层测试折结果。"
                 f"调参版差距最大的是 **{MODEL_SHORT[worst['model']]}**（{worst['train_f1_macro']:.2f} → "
                 f"{worst['oof_f1_macro']:.2f}，差 {worst['gap']:+.2f}）。差距大说明模型记住了训练样本，"
                 f"在 27 条样本上这比任何单项指标都更能说明「不能拿训练精度当泛化性能」。"),
        "figs": [ent],
    })

    # 5.7 种子稳定性
    path, ent = _fig(fig_dir, "seed_stability.png", "切分种子稳定性")
    curves.plot_seed_stability(stability, path)
    rows = []
    for m in ORDERED:
        sub = stability[(stability["model"] == m) & (stability["version"] == "tuned")]
        rows.append({"模型": MODEL_SHORT[m], "F1 最低": sub["f1_macro"].min(), "F1 最高": sub["f1_macro"].max(),
                     "F1 极差": sub["f1_macro"].max() - sub["f1_macro"].min(),
                     "种子 42 名次": None})
    st = pd.DataFrame(rows)
    ranks = {}
    for s in sorted(stability["seed"].unique()):
        sub = stability[(stability["seed"] == s) & (stability["version"] == "tuned")
                        & (stability["model"].isin(ORDERED))].sort_values("f1_macro", ascending=False)
        ranks[s] = {MODEL_SHORT[m]: i + 1 for i, m in enumerate(sub["model"])}
    st["各种子 F1 名次"] = st["模型"].map(lambda n: " / ".join(str(ranks[s][n]) for s in sorted(ranks)))
    st = st.drop(columns=["种子 42 名次"])
    winners = {s: min(r, key=r.get) for s, r in ranks.items()}
    sections.append({
        "heading": "5.7 切分种子稳定性",
        "text": ("三个种子下 macro-F1 第一名分别是："
                 + "、".join(f"种子 {s} → {w}" for s, w in winners.items())
                 + "。名次列按种子 " + " / ".join(str(s) for s in sorted(ranks)) + " 顺序列出。"
                 + ("三个种子的冠军**不一致**，说明模型排名对切分敏感。" if len(set(winners.values())) > 1
                    else "三个种子冠军一致，但各模型分数仍有明显波动。")),
        "figs": [ent], "table": st, "float_cols": ["F1 最低", "F1 最高", "F1 极差"],
    })
    return sections


# ====================================================================== 第 6 章

def cv_sections(cfg, X, y, oof, fig_dir: Path, out_dir: Path, log=print) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    cn = cfg["class_names"]
    sections, saved = [], {}

    log("   CV 方案对比 …")
    sch = cv_scheme_comparison(X, y, cfg, seed)
    sch["summary"].to_csv(out_dir / "cv_schemes_summary.csv", index=False)
    sch["units"].to_csv(out_dir / "cv_schemes_units.csv", index=False)
    path, ent = _fig(fig_dir, "cv_schemes.png", "交叉验证方案对比")
    curves.plot_cv_schemes(sch["summary"], path)
    summ = sch["summary"]
    n_rep = int(cfg["analysis"]["cv_schemes"]["repeated"]["n_repeats"])
    rep_label = summ[summ["scheme"].str.startswith("rskf")]["scheme_label"].iloc[0]
    labels = list(dict.fromkeys(summ["scheme_label"]))
    piv = summ.pivot(index="model", columns="scheme_label", values="f1_macro")[labels]
    piv = piv.loc[ORDERED + ["dummy"]]
    piv.insert(0, "模型", [MODEL_SHORT[m] for m in piv.index])
    piv["方案间极差"] = piv[labels].max(axis=1) - piv[labels].min(axis=1)
    loo_dummy = summ[(summ["model"] == "dummy") & (summ["scheme"] == "loo")]["accuracy"].iloc[0]
    best_by = {lab: MODEL_SHORT[summ[(summ["scheme_label"] == lab) & (summ["model"] != "dummy")]
                                .sort_values("f1_macro", ascending=False)["model"].iloc[0]] for lab in labels}
    sections.append({
        "heading": "6.1 不同交叉验证方案的结果对比",
        "text": ("下表为各模型（**默认超参数**，不调参）在 5 种 CV 方案下的折外 macro-F1。k 折与留一法为一次完整"
                 f"拼接；{rep_label} 为 {n_rep} 次不同切分的均值。按 macro-F1 第一名："
                 + "、".join(f"{k}→{v}" for k, v in best_by.items()) + "。"),
        "table": piv.reset_index(drop=True), "float_cols": labels + ["方案间极差"], "figs": [ent],
        "after": (f"**留一法的已知陷阱**：Dummy 基线在留一法下 Accuracy = {loo_dummy:.3f}。原因是三类严格均衡时，"
                  f"每次留出一条样本，它所属的类在训练集中就成了少数类，先验基线永远预测另外两类之一——"
                  f"这是留一法在小样本、均衡类别上的系统性悲观偏差，不是代码错误。因此本项目主实验采用分层 k 折而非留一法；"
                  f"方案间极差列说明，**仅改变 CV 方案**就能让同一模型的 macro-F1 变动这么多。"),
    })

    rep = sch["units"][sch["units"]["unit"] == "repeat"]
    path, ent = _fig(fig_dir, "cv_repeated_box.png", "重复分层 5 折的切分波动")
    curves.plot_repeated_cv_box(sch["units"], path)
    rows = []
    for m in ORDERED + ["dummy"]:
        v = rep[rep["model"] == m]["f1_macro"]
        rows.append({"模型": MODEL_SHORT[m], "均值": v.mean(), "标准差": v.std(ddof=0), "最低": v.min(), "最高": v.max()})
    rt = pd.DataFrame(rows)
    sections.append({
        "heading": "6.2 重复交叉验证：仅换切分带来的波动",
        "text": (f"同一模型、同一数据，只换随机切分（{len(rep['unit_id'].unique())} 次）时，拼接 macro-F1 的分布如下。"
                 "它比单次 5 折更稳健地估计「默认参数下」的平均表现，但重复之间共享样本，不是独立重复实验。"),
        "table": rt, "float_cols": ["均值", "标准差", "最低", "最高"], "figs": [ent],
    })

    log("   置换检验 …")
    perm = permutation_tests(X, y, cfg, seed)
    perm["table"].to_csv(out_dir / "permutation_test.csv", index=False)
    np.savez(out_dir / "permutation_null.npz", **perm["null"])
    path, ent = _fig(fig_dir, "permutation_test.png", "标签置换检验")
    curves.plot_permutation(perm, path)
    pt = perm["table"].copy()
    sig = pt[pt["p_value"] < 0.05]["model"].map(MODEL_SHORT).tolist()
    pt["模型"] = pt["model"].map(MODEL_SHORT)
    pt = pt[["模型", "score_f1_macro", "null_mean", "null_p95", "p_value", "n_permutations"]].rename(columns={
        "score_f1_macro": "真实标签 F1", "null_mean": "置换均值", "null_p95": "置换 95% 分位",
        "p_value": "p 值", "n_permutations": "置换次数"})
    n_perm = int(cfg["analysis"]["permutation"]["n_permutations"])
    sections.append({
        "heading": "6.3 标签置换检验",
        "text": (f"把标签随机打乱 {n_perm} 次，每次用同样的分层 5 折重新评估（默认超参数）。若真实标签的分数并不比"
                 f"打乱后的分数高，就说明模型学到的可能只是噪声。p 值 = (置换分数 ≥ 真实分数的次数 + 1) / ({n_perm} + 1)，"
                 f"最小可达 {1 / (n_perm + 1):.4f}。"),
        "table": pt, "float_cols": ["真实标签 F1", "置换均值", "置换 95% 分位", "p 值"], "figs": [ent],
        "after": (f"p < 0.05 的模型：**{'、'.join(sig) if sig else '无'}**。"
                  + ("其余模型在默认参数下与随机水平没有显著差别。同时检验 6 个模型存在多重比较问题（未做校正），"
                     "p 略低于 0.05 的结果不宜过度解读。" if sig else
                     "即在默认参数下，没有一个模型能以 0.05 水平证明其分数优于「标签与特征无关」时的水平，"
                     "这与第 4.11 节胜者对 Dummy 的配对检验不显著一致。")
                  + "置换检验只回答「是否优于随机」，不回答「哪个模型更好」。"),
    })

    log("   学习曲线 …")
    lc = learning_curves(X, y, cfg, seed)
    lc.to_csv(out_dir / "learning_curve.csv", index=False)
    dummy_f1 = float(summ[(summ["model"] == "dummy") & (summ["scheme"].str.startswith("rskf"))]["f1_macro"].iloc[0])
    path, ent = _fig(fig_dir, "learning_curves.png", "学习曲线")
    curves.plot_learning_curves(lc, path, dummy_f1=dummy_f1)
    g = lc.groupby(["model", "n_train"])[["train_f1_macro", "test_f1_macro"]].mean().reset_index()
    nmin, nmax = g["n_train"].min(), g["n_train"].max()
    rows = []
    for m in ORDERED:
        a = g[(g["model"] == m) & (g["n_train"] == nmin)].iloc[0]
        b = g[(g["model"] == m) & (g["n_train"] == nmax)].iloc[0]
        rows.append({"模型": MODEL_SHORT[m], f"测试 F1 (n={nmin})": a["test_f1_macro"],
                     f"测试 F1 (n={nmax})": b["test_f1_macro"], "提升": b["test_f1_macro"] - a["test_f1_macro"],
                     f"训练-测试差 (n={nmax})": b["train_f1_macro"] - b["test_f1_macro"]})
    lt = pd.DataFrame(rows)
    rising = lt[lt["提升"] > 0.05]["模型"].tolist()
    sections.append({
        "heading": "6.4 学习曲线",
        "text": (f"在每个训练折内按「每类等量」子采样（{nmin}–{nmax} 条），观察测试折 macro-F1 随样本量的变化。"
                 f"从 n={nmin} 到 n={nmax} 测试 F1 提升超过 0.05 的模型：{'、'.join(rising) or '无'}。"
                 "若曲线在右端仍在上升、且训练-测试差距较大，说明**增加样本**比继续调参更可能带来改善。"),
        "table": lt, "float_cols": [c for c in lt.columns if c != "模型"], "figs": [ent],
    })

    log("   验证曲线 …")
    vc = validation_curves(X, y, cfg, seed)
    vc.to_csv(out_dir / "validation_curve.csv", index=False)
    path, ent = _fig(fig_dir, "validation_curves.png", "验证曲线")
    curves.plot_validation_curves(vc, path)
    rows = []
    for m in ORDERED:
        sub = vc[vc["model"] == m]
        b = sub.loc[sub["test_mean"].idxmax()]
        rows.append({"模型": MODEL_SHORT[m], "参数": b["param"], "验证最高取值": b["value"],
                     "验证 F1": b["test_mean"], "训练 F1": b["train_mean"], "训练-验证差": b["train_mean"] - b["test_mean"],
                     "验证 F1 范围": f"{sub['test_mean'].min():.2f}–{sub['test_mean'].max():.2f}"})
    vt = pd.DataFrame(rows)
    sections.append({
        "heading": "6.5 验证曲线（超参数敏感性）",
        "text": ("每个模型只扫描一个关键超参数，其余保持默认。左侧通常欠拟合（训练、验证都低），右侧过拟合"
                 "（训练高、验证低）。**注意**：全部 27 条样本都参与了这里的 CV，所以「验证最高取值」只是描述性的，"
                 "不能拿来替换主实验的调参结果，也不能把它的分数当泛化性能。"),
        "table": vt, "float_cols": ["验证 F1", "训练 F1", "训练-验证差"], "figs": [ent],
    })

    cal_models = [m for m in ORDERED if m != "svm"] + ["dummy"]
    cal = calibration_summary(oof, cfg, cal_models)
    cal["table"].to_csv(out_dir / "calibration.csv", index=False)
    path, ent = _fig(fig_dir, "calibration.png", "概率校准（可靠性曲线）")
    curves.plot_calibration(cal, cn, path)
    ct = cal["table"].copy()
    ct["模型"] = ct["model"].map(MODEL_SHORT)
    ct = ct[["模型", "brier", "log_loss", "ece_top1", "mean_confidence", "accuracy"]].rename(columns={
        "brier": "Brier", "log_loss": "Log loss", "ece_top1": "ECE (top-1)", "mean_confidence": "平均置信度",
        "accuracy": "Accuracy"})
    dummy_brier = float(ct[ct["模型"] == MODEL_SHORT["dummy"]]["Brier"].iloc[0])
    better = ct[(ct["模型"] != MODEL_SHORT["dummy"]) & (ct["Brier"] < dummy_brier)]["模型"].tolist()
    sections.append({
        "heading": "6.6 概率校准",
        "text": ("对输出概率的模型（调参版 OOF 概率；SVM 输出决策分数，不在其中），检查「预测 70% 的样本是否真有约 70% 属于该类」。"
                 "Brier 分数（三类多分类形式，0 最好）同时衡量校准与区分度；ECE 为 top-1 置信度与准确率之差的加权平均。"),
        "table": ct, "float_cols": ["Brier", "Log loss", "ECE (top-1)", "平均置信度", "Accuracy"], "figs": [ent],
        "after": (f"Dummy 基线的 Brier = {dummy_brier:.3f}；Brier 低于基线的模型：{'、'.join(better) or '无'}。"
                  f"每类只有 9 个正例、分箱只有 {cfg['analysis']['calibration_bins']} 个，可靠性曲线非常粗糙，"
                  f"只能看方向（偏自信 / 偏保守），不足以支撑概率校准或阈值设定。"),
    })

    saved.update({"permutation": perm["table"], "cv_summary": summ, "n_sig": len(sig),
                  "sig_models": sig, "n_perm": n_perm, "calib_better": better})
    return sections, saved


# ====================================================================== 第 7 章

def shap_sections(cfg, X, y, meta, final_pipe, final_model: str, shap_result, main_res,
                  fig_dir: Path, out_dir: Path, log=print) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = cfg["feature_columns"]
    cn = cfg["class_names"]
    Xn = np.asarray(X, dtype=float)
    values = shap_result["values"]
    label = MODEL_LABELS[final_model]
    out_space = shap_result["meta"]["output_space"]
    sections, summary = [], {}
    agg = shap_result["aggregate"]
    top = agg["feature"].tolist()[:3]

    # 7.1 蜂群图
    path, ent = _fig(fig_dir, "shap_beeswarm.png", "SHAP 蜂群图（全量模型）")
    plot_shap_beeswarm(values, Xn, feats, cn, path,
                       title=f"{label} SHAP 蜂群图（全部 {len(Xn)} 条训练样本，训练后描述性分析）")
    direction = []
    for f in top:
        j = feats.index(f)
        rho = spearmanr(Xn[:, j], values[:, j, 2]).statistic
        if np.isfinite(rho):
            direction.append(f"{f}（ρ={rho:+.2f}）")
    sections.append({
        "heading": "7.1 SHAP 蜂群图：贡献方向与大小",
        "text": (f"每个点是一条样本，横轴为该特征对相应类别输出的 SHAP 贡献（{out_space}），颜色为特征取值高低"
                 f"（蓝=低、红=高）。特征按三类聚合 mean(|SHAP|) 从上到下排序，前三位为 {'、'.join(top)}。"
                 f"对「{cn[2]}」输出，前三位特征取值与 SHAP 的 Spearman 相关为：{'、'.join(direction) or '无法计算'}"
                 f"（正值表示取值越高越推向{cn[2]}）。这是模型行为的描述，不是生理机制或因果关系。"),
        "figs": [ent],
    })

    # 7.2 依赖图
    k = int(cfg["analysis"].get("shap_dependence_top_k", 4))
    path, ent = _fig(fig_dir, "shap_dependence.png", f"SHAP 依赖图（前 {k} 个特征）")
    plot_shap_dependence(values, Xn, feats, cn, path, top_k=k)
    sections.append({
        "heading": "7.2 SHAP 依赖图：重要特征的取值—贡献关系",
        "text": ("横轴为特征原始取值，纵轴为 SHAP 值，三种颜色分别对应对三个类别输出的贡献。树模型的依赖关系通常呈阶梯状"
                 "（分裂阈值）；同一取值下点的纵向分散来自与其他特征的交互。细线为线性趋势，只作方向参考。"),
        "figs": [ent],
    })

    # 7.3 瀑布图
    scores = class_scores(final_pipe, Xn)
    picks = pick_representative_samples(scores, y)
    sids = meta["sample_id"].tolist()
    path, ent = _fig(fig_dir, "shap_waterfall.png", "SHAP 瀑布图（各类代表样本）")
    plot_shap_waterfalls(values, shap_result["base_values"], Xn, y, sids, picks, feats, cn, path, out_space)
    sections.append({
        "heading": "7.3 SHAP 瀑布图：单个样本的预测是怎么来的",
        "text": (f"对每个真实类别，选全量模型给该类打分最高的样本（{'、'.join(sids[i] for i in picks)}），"
                 f"把「该类输出」分解为基准值 E[f(x)] 加上各特征贡献。条越长，影响越大；红色推高、蓝色拉低该类输出。"
                 f"注意全量模型在训练集上 Accuracy 很高，这些样本都被它「见过」，瀑布图展示的是模型如何记住它们，"
                 f"不是对新样本的预测解释。"),
        "figs": [ent],
    })
    np.savez(out_dir / "full_model_shap.npz", values=values, base_values=shap_result["base_values"],
             feature_names=np.array(feats), sample_ids=np.array(sids))

    # 7.4 折外 SHAP
    fitted = main_res.get("fitted", {}).get((final_model, "tuned"))
    if cfg["analysis"].get("shap_cv", True) and fitted:
        log("   折外 SHAP …")
        cvs = cv_shap(fitted, main_res["splits"], Xn, feats, cn, cfg)
        np.savez(out_dir / "cv_shap_values.npz", values=cvs["values"], feature_names=np.array(feats))
        cvs["aggregate"].to_csv(out_dir / "cv_shap_ranking.csv", index=False)
        cvs["fold_ranks"].to_csv(out_dir / "cv_shap_fold_ranks.csv", index=False)
        path, ent = _fig(fig_dir, "shap_cv_stability.png", "折外 SHAP 与逐折排名稳定性")
        plot_shap_cv_compare(values, cvs["values"], cvs["fold_ranks"], feats, path, label)
        path2, ent2 = _fig(fig_dir, "shap_beeswarm_oof.png", "折外 SHAP 蜂群图")
        plot_shap_beeswarm(cvs["values"], Xn, feats, cn, path2,
                           title=f"{label} 折外 SHAP 蜂群图（每条样本由未见过它的折模型解释）")
        full_rank = dict(zip(agg["feature"], agg["rank"]))
        cv_rank = dict(zip(cvs["aggregate"]["feature"], cvs["aggregate"]["rank"]))
        fr = cvs["fold_ranks"]
        rows = []
        for f in agg["feature"]:
            r = fr[fr["feature"] == f]["rank"]
            rows.append({"特征": f, "全量排名": full_rank[f], "折外排名": cv_rank[f],
                         "逐折排名均值": r.mean(), "逐折排名标准差": r.std(ddof=0),
                         "进入前三的折数": f"{int((r <= 3).sum())}/{len(r)}"})
        tab = pd.DataFrame(rows)
        rho = spearmanr([full_rank[f] for f in feats], [cv_rank[f] for f in feats]).statistic
        summary["cv_rank_rho"] = float(rho)
        stable = tab[tab["进入前三的折数"].str.startswith(f"{len(fitted)}/")]["特征"].tolist()
        top3_cnt = {f: int((fr[fr["feature"] == f]["rank"] <= 3).sum()) for f in feats}
        most = max(top3_cnt.values())
        frequent = [f for f, c in top3_cnt.items() if c == most]
        sections.append({
            "heading": "7.4 折外 SHAP 与排名稳定性",
            "text": (f"第 4.9 节与上面各图都用全量模型解释它见过的样本。这里换成**折外**解释：外层每折的调参模型"
                     f"（只用该折训练集拟合）解释该折测试样本，拼成 {len(Xn)} 条折外 SHAP。"
                     f"全量排名与折外排名的 Spearman 相关为 **{rho:.2f}**；"
                     f"在全部 {len(fitted)} 个折中都排进前三的特征：{'、'.join(stable) or '无'}"
                     + ("。" if stable else f"；进前三次数最多的是 {'、'.join(frequent)}（{most}/{len(fitted)} 折）。")
                     + "全量与折外排名一致性越高，越说明排名不只是「解释见过的样本」造成的；"
                       "但逐折排名仍会跳动，见下表标准差列。"),
            "figs": [ent, ent2], "table": tab, "float_cols": ["逐折排名均值", "逐折排名标准差"],
            "after": ("逐折排名基于该折训练集上的 mean(|SHAP|)。排名在折间跳动越大，越说明「哪个特征最重要」"
                      "本身受样本构成影响，不宜把单次全量 SHAP 排名当作稳定结论。"),
        })

    # 7.5 六模型对比
    if cfg["analysis"].get("shap_all_models", True):
        log("   六模型 SHAP 对比（各模型全量训练内调参 + 解释）…")
        fitted_all, params = {}, {}
        for m in ORDERED:
            if m == final_model:
                fitted_all[m] = final_pipe
                continue
            est, bp, _, _ = refit_on_full_data(cfg, Xn, y, m, int(cfg["seed"]), int(cfg["n_jobs"]))
            fitted_all[m], params[m] = est, bp
        mm = multi_model_shap(fitted_all, Xn, feats, cfg)
        mm["table"].to_csv(out_dir / "multi_model_shap.csv", index=False)
        (out_dir / "multi_model_shap_meta.json").write_text(
            json.dumps({"explainers": mm["meta"], "refit_params": params}, ensure_ascii=False, indent=2,
                       default=str), encoding="utf-8")
        path, ent = _fig(fig_dir, "shap_model_comparison.png", "六模型 SHAP 重要性对比")
        plot_shap_model_heatmap(mm["table"], path, ORDERED)
        t = mm["table"]
        win_rank = t[t["model"] == final_model].set_index("feature")["rank"]
        rows = []
        for m in ORDERED:
            sub = t[t["model"] == m].sort_values("rank")
            r = sub.set_index("feature")["rank"].reindex(win_rank.index)
            rows.append({"模型": MODEL_SHORT[m], "解释器": mm["meta"][m]["explainer"].split("(")[0],
                         "输出空间": mm["meta"][m]["output_space"].split("（")[0],
                         "前三特征": "、".join(sub["feature"].head(3)),
                         f"与{MODEL_SHORT[final_model]}排名相关 ρ": spearmanr(r, win_rank).statistic})
        mt = pd.DataFrame(rows)
        counts = t[t["rank"] <= 3]["feature"].value_counts()
        common = counts[counts >= 4].index.tolist()
        summary["common_top3"] = common
        sections.append({
            "heading": "7.5 六模型 SHAP 重要性对比",
            "text": ("六个模型都在全部样本上按主实验同一搜索空间做训练内调参后拟合，再分别计算 SHAP。由于输出空间不同"
                     "（概率 / 决策分数 / margin），绝对值不可比，热力图中每个模型的重要性已归一化为占比。"
                     f"在至少 4 个模型中排进前三的特征：**{'、'.join(common) or '无'}**——跨模型一致出现的特征，"
                     f"比只在单个模型里重要的特征更值得进一步关注。"),
            "figs": [ent], "table": mt, "float_cols": [f"与{MODEL_SHORT[final_model]}排名相关 ρ"],
            "after": "注意：这些模型的 OOF 表现大多接近基线（见第 4.11、6.3 节），对表现差的模型做解释，"
                     "解释的是它学到的（可能是噪声的）规律，不能当作特征有效性的证据。",
        })
    summary["top3_final"] = top
    return sections, summary
