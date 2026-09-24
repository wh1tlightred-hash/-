"""PROJECT_BRIEF.md 第 8 节「运行验收」的端到端核查脚本。

对**磁盘上已生成的真实产物**逐条核验，不重新训练、不推测。任何一条不满足都会
以 FAIL 输出并让脚本以非零码退出。用法：

    .venv/Scripts/python.exe verify_acceptance.py

核验项对应：
 1. 27 × 10 特征、三类各 9 条；原始附件哈希与 Downloads 原件一致（未修改）。
 2. 五个外层测试折互不重叠、合并覆盖全部样本、每折含三类；内外层索引不交叉。
 3. 元数据（来源/标签/行号/样本 ID）没有进入 X；预处理在 Pipeline 内。
 4. 六种模型 + Dummy 基线均真实运行成功，无模拟补齐。
 5. 每个模型版本 OOF 为 27 条，类别分数列序一致，概率有限且行和≈1。
 6. 从保存的 OOF 文件重算指标与报告汇总一致；混淆矩阵计数合计 27。
 7. 模型重载后预测一致；错误输入给出明确错误。
 8. 最小测试、完整主实验、报告图表均已实际执行与产出。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import file_sha256  # noqa: E402
from src.metrics import class_scores, compute_metrics  # noqa: E402
from src.models import ALL_MODELS, REAL_MODELS  # noqa: E402

# 原始附件所在目录（交接说明第 2 节）。交付副本可能被单独拷到别的机器，或本机原件被清理，
# 因此这里允许缺失：缺失时只核对「项目内文件哈希 == 运行期记录哈希」，不让整份验收直接崩掉。
# 需要指向别处时用环境变量 TCM_ORIGINALS_DIR 覆盖。
ORIGINALS = Path(os.environ.get("TCM_ORIGINALS_DIR", r"C:\Users\swild\Downloads"))

_checks: list[tuple[bool, int, str, str]] = []


def check(ok: bool, item: int, title: str, detail: str = "") -> bool:
    _checks.append((bool(ok), item, title, detail))
    return bool(ok)


def main() -> int:
    cfg = load_config()
    results = ROOT / cfg["results_dir"]
    seed_dir = results / f"seed{cfg['seed']}"
    oof_dir = seed_dir / "oof"
    feats = cfg["feature_columns"]

    # ---------- 1. 数据形状与原始附件完整性 ----------
    merged = pd.read_csv(ROOT / cfg["data_processed_dir"] / "merged.csv")
    check(merged.shape[0] == 27, 1, "有效样本 27 条", f"实际 {merged.shape[0]}")
    check(
        [c for c in feats if c in merged.columns] == feats,
        1, "10 个特征列齐全且顺序正确", f"{feats}",
    )
    counts = merged["label"].value_counts().sort_index().to_dict()
    check(counts == {0: 9, 1: 9, 2: 9}, 1, "三类各 9 条", f"{counts}")
    check("FO" in merged.columns and "F0" not in merged.columns, 1, "表头保留字母 O 的 FO，未改为 F0")

    for key, fname in (("xlsx", "声诊练习.xlsx"), ("docx", "中医声诊实验.docx")):
        proj = file_sha256(ROOT / "inputs" / fname)
        rec = json.loads((results / "input_hashes.json").read_text(encoding="utf-8"))[key]
        orig_path = ORIGINALS / fname
        if orig_path.exists():
            orig = file_sha256(orig_path)
            check(
                proj == orig == rec, 1, f"原始附件未修改且哈希与记录一致：{fname}",
                f"项目内/原件/记录 = {proj[:12]}…/{orig[:12]}…/{rec[:12]}…",
            )
        else:
            # 副本被单独拷走或本机原件已清理时，不因取不到原件而崩溃。
            check(
                proj == rec, 1, f"项目内附件哈希与运行期记录一致：{fname}（未找到原件目录 {ORIGINALS}，跳过原件对比）",
                f"{proj[:12]}…",
            )

    # ---------- 2. 切分完整性 ----------
    audit = json.loads((results / "search_audit" / f"seed_{cfg['seed']}" / "splits.json").read_text(encoding="utf-8"))
    folds = audit["folds"]
    check(len(folds) == 5, 2, "外层 5 折", f"实际 {len(folds)}")
    tests = [set(f["test"]) for f in folds]
    disjoint = all(len(a & b) == 0 for i, a in enumerate(tests) for b in tests[i + 1:])
    check(disjoint, 2, "五个外层测试折互不重叠")
    check(set().union(*tests) == set(range(27)), 2, "外层测试折合并覆盖全部 27 条")
    y_all = merged["label"].to_numpy()
    check(all(set(y_all[list(t)]) == {0, 1, 2} for t in tests), 2, "每个外层测试折均含三类")
    inner_ok = True
    for f in folds:
        tr, te = set(f["train"]), set(f["test"])
        if tr & te:
            inner_ok = False
        for inn in f["inner"]:
            a, v = set(inn["train"]), set(inn["validation"])
            if (a & v) or not a <= tr or not v <= tr:
                inner_ok = False
    check(inner_ok, 2, "内外层索引不交叉，内层仅在当前外层训练折内")
    check(
        audit["sample_ids"] == merged["sample_id"].tolist(),
        2, "审计文件样本顺序与合并数据一致",
    )

    # ---------- 3. 元数据未进入 X / 预处理在 Pipeline 内 ----------
    from src.data import load_dataset, split_xy

    df = load_dataset(cfg["input_xlsx"], feats)
    X, y, meta = split_xy(df, feats)
    check(list(X.columns) == feats, 3, "X 恰好只有 10 个特征列")
    leaked = [c for c in ("sample_id", "label", "label_name", "source_sheet", "excel_row") if c in X.columns]
    check(not leaked, 3, "样本 ID / 标签 / 来源 / 行号均未进入 X", f"泄漏列 {leaked}" if leaked else "")
    check(np.array_equal(y, merged["label"].to_numpy()), 3, "y 与合并数据标签一致")

    # 必备产物：缺失必须直接判 FAIL。原先把这些检查包在 `if pipe_path.exists():` 里，
    # 结果模型文件被删除时验收仍然全绿通过（审查已复现：42/42 通过、退出码 0）。这里改为
    # 「无条件产出固定条数的检查」，缺失时就是 FAIL，验收项数量也不再随产物有无而变。
    pipe_path = ROOT / cfg["models_dir"] / "final_pipeline.joblib"
    meta_path = ROOT / cfg["models_dir"] / "final_metadata.json"
    check(pipe_path.exists(), 3, "最终模型文件存在（缺失即失败）", pipe_path.name)
    check(meta_path.exists(), 3, "模型元数据存在（缺失即失败）", meta_path.name)

    pipe, load_err = None, None
    if pipe_path.exists():
        try:
            pipe = joblib.load(pipe_path)
        except Exception as e:  # noqa: BLE001
            load_err = f"{type(e).__name__}: {e}"
    check(pipe is not None, 3, "最终模型可加载", load_err or type(pipe).__name__)

    steps = list(pipe.named_steps) if pipe is not None else []
    check("clf" in steps, 3, "最终模型为完整 Pipeline（含分类器）", f"步骤 {steps}" if steps else "模型不可用")
    pre = [s for s in steps if s != "clf"]
    check(bool(steps) and all(s in ("imputer", "scaler") for s in pre), 3,
          "拟合型预处理位于 Pipeline 内部", f"{pre}" if steps else "模型不可用")

    meta_required = ("model", "feature_order", "label_map", "class_names", "score_kind")
    final_meta, meta_err = None, None
    if meta_path.exists():
        try:
            final_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            meta_err = f"{type(e).__name__}: {e}"
    check(final_meta is not None, 3, "模型元数据可解析", meta_err or f"{len(final_meta)} 个键")
    check(bool(final_meta) and all(k in final_meta for k in meta_required), 3,
          "模型元数据含必需字段", "、".join(meta_required))
    check(bool(final_meta) and final_meta.get("feature_order") == feats, 3,
          "元数据特征顺序与配置一致",
          str(final_meta.get("feature_order")) if final_meta else "元数据不可用")

    # ---------- 4 & 5. 模型齐全 / OOF 完整 ----------
    expected = [(m, v) for m in REAL_MODELS for v in ("untuned", "tuned")] + [("dummy", "untrained")]
    missing = [f"{m}_{v}" for m, v in expected if not (oof_dir / f"{m}_{v}_oof.csv").exists()]
    check(not missing, 4, "六模型 × 调参/未调参 + Dummy 基线 OOF 全部存在", f"缺失 {missing}" if missing else f"{len(expected)} 个文件")
    check(len(ALL_MODELS) == 7, 4, "模型集合为 6 真实模型 + Dummy")

    all27, order_ok, finite_ok, rowsum_ok = True, True, True, True
    for m, v in expected:
        o = pd.read_csv(oof_dir / f"{m}_{v}_oof.csv")
        if len(o) != 27:
            all27 = False
        cols = [c for c in o.columns if c.startswith("score_")]
        if [c.split("_")[1] for c in cols] != ["0", "1", "2"]:
            order_ok = False
        sc = o[cols].to_numpy(dtype=float)
        if not np.isfinite(sc).all():
            finite_ok = False
        if m != "svm" and not np.allclose(sc.sum(axis=1), 1.0, atol=1e-6):
            rowsum_ok = False
    check(all27, 5, "每个模型版本的 OOF 预测均为 27 条")
    check(order_ok, 5, "类别分数列序统一为 0/1/2（低危/中危/高危）")
    check(finite_ok, 5, "所有类别分数均为有限值")
    check(rowsum_ok, 5, "非 SVM 模型概率行和≈1（SVM 为决策分数，不适用）")

    # ---------- 6. OOF 重算与报告一致 ----------
    reported = pd.read_csv(seed_dir / "oof_metrics.csv")
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "mAP", "pr_auc_macro"]
    recompute_ok, cm_ok = True, True
    worst = 0.0
    for _, row in reported.iterrows():
        o = pd.read_csv(oof_dir / f"{row['model']}_{row['version']}_oof.csv")
        sc = o[[c for c in o.columns if c.startswith("score_")]].to_numpy(dtype=float)
        m = compute_metrics(o["y_true"].to_numpy(int), sc, o["y_pred"].to_numpy(int), cfg["zero_division"])
        for k in keys:
            if pd.isna(row[k]):
                continue
            worst = max(worst, abs(float(m[k]) - float(row[k])))
            if not np.isclose(m[k], float(row[k]), atol=1e-9):
                recompute_ok = False
        if int(np.sum(m["confusion_matrix"])) != 27:
            cm_ok = False
    check(recompute_ok, 6, "从 OOF 文件重算的指标与报告汇总一致", f"最大偏差 {worst:.2e}")
    check(cm_ok, 6, "各模型混淆矩阵计数合计为 27")

    # 明确检查「必需结果不是 NA」：原先遇到 NaN 直接 continue，等于把缺失值静默放过。
    # 本数据集每折三类齐全，7 个主要指标都应可算且有限；出现 NA 即为异常。
    na_items = [f"{r['model']}/{r['version']}:{k}"
                for _, r in reported.iterrows() for k in keys if pd.isna(r[k])]
    check(not na_items, 6, "OOF 汇总的 7 个主要指标均为有限值（无 NA）",
          f"NA 项 {na_items[:5]}" if na_items else f"{len(reported)} 行 × {len(keys)} 指标")

    # ---------- 7. 序列化一致性与错误输入 ----------
    # 同样不放进 `if exists`：模型不可用时这里必须是 FAIL，而不是「少检查几条」。
    if pipe is not None:
        Xn = X.to_numpy(dtype=float)
        p1 = pipe.predict(Xn)
        s1 = class_scores(pipe, Xn)
        pipe2 = joblib.load(pipe_path)
        p2 = pipe2.predict(Xn)
        s2 = class_scores(pipe2, Xn)
        check(np.array_equal(p1, p2) and np.allclose(s1, s2), 7, "模型重载后预测与分数不变")
    else:
        check(False, 7, "模型重载后预测与分数不变", "模型不可用，无法验证")

    from predict import validate_and_reorder

    bad_cases = {
        "缺列": pd.DataFrame([{k: 1.0 for k in feats[:-1]}]),
        "多列": pd.DataFrame([{**{k: 1.0 for k in feats}, "extra": 1.0}]),
        "非数值": pd.DataFrame([{**{k: 1.0 for k in feats}, feats[0]: "abc"}]),
    }
    bad_ok = []
    for name, bad in bad_cases.items():
        try:
            validate_and_reorder(bad, feats)
            bad_ok.append(name)
        except ValueError:
            pass
    check(not bad_ok, 7, "缺列/多列/非数值输入均给出明确 ValueError", f"未报错 {bad_ok}" if bad_ok else "")
    reordered = validate_and_reorder(pd.DataFrame([{k: 1.0 for k in reversed(feats)}]), feats)
    check(reordered.shape == (1, 10), 7, "列顺序不同时按列名安全重排")

    # ---------- 8. 完整实验 / 报告 / 图表 / 测试产物 ----------
    figs = sorted((results / "figures").glob("*.png"))
    need_figs = ["class_counts", "model_metrics", "macro_roc", "macro_pr", "shap"]
    have = {f.stem for f in figs}
    check(all(n in have for n in need_figs), 8, "必需图表齐备", f"{len(figs)} 张：{sorted(have)}")
    check(len(list((results / "figures").glob("per_class_*.png"))) == 6, 8, "六模型逐类 ROC/PR 图齐备")
    check(len(list((results / "figures").glob("cm_*_count.png"))) >= 1, 8, "最佳模型混淆矩阵（计数）已产出")

    rpt_md = ROOT / cfg["report_dir"] / "实验报告.md"
    rpt_html = ROOT / cfg["report_dir"] / "实验报告.html"
    check(rpt_md.exists() and rpt_html.exists(), 8, "中文报告 Markdown 与 HTML 均已生成")
    txt = rpt_md.read_text(encoding="utf-8") if rpt_md.exists() else ""
    refs = [ln.split("](", 1)[1].rstrip(")") for ln in txt.splitlines() if ln.startswith("![")]
    broken = [r for r in refs if not (rpt_md.parent / r).resolve().exists()]
    check(refs and not broken, 8, "报告中图像相对路径均可解析", f"断裂 {broken}" if broken else f"{len(refs)} 个引用")

    for s in cfg["stability_seeds"]:
        check((results / f"seed{s}" / "oof_metrics.csv").exists(), 8, f"稳定性种子 {s} 的完整 OOF 已产出")
    check((results / "stability.csv").exists(), 8, "稳定性汇总表 stability.csv 已产出")
    env = json.loads((ROOT / "logs" / "environment_check.json").read_text(encoding="utf-8"))
    check(bool(env.get("packages")) and env["python"].startswith("3.14"), 8,
          "环境核验日志已记录（Python 3.14.7）", f"Python {env.get('python')}，{len(env.get('packages', {}))} 个包")

    # 解释器路径：**不要求**等于当前副本位置——副本被拷到别处、或未重跑 main.py 时，
    # 历史日志仍然有效，把这种情况判 FAIL 会与「副本可移植」的目标自相矛盾。
    # 真正有意义的不变量是「日志与报告记录的路径来自同一次运行」。
    exe = str(env.get("executable", ""))
    check(bool(exe), 8, "环境日志记录了运行期解释器路径", exe or "未记录")
    check(bool(exe) and exe in txt, 8, "报告与日志记录的解释器路径一致（同一次运行）",
          exe if exe else "未记录")

    # 补充分析产物（不确定性量化 / 配对检验 / 含模型选择的估计）
    for rel, title in (("uncertainty.csv", "bootstrap 置信区间表"),
                       ("mcnemar.json", "胜者 vs Dummy 配对检验"),
                       ("selection_estimate.json", "含模型选择的无事后挑选估计")):
        check((results / rel).exists(), 8, f"补充分析产物已产出：{title}", rel)
    check((results / "figures" / "uncertainty.png").exists(), 8, "bootstrap 区间图已产出")
    check("4.11" in txt and "含模型选择" in txt, 8, "报告中已包含补充分析小节（4.11）")

    if (results / "uncertainty.csv").exists():
        uc = pd.read_csv(results / "uncertainty.csv")
        want_models = set(REAL_MODELS) | {"dummy"}
        check(want_models <= set(uc["model"]), 8, "不确定性表覆盖 6 模型 + Dummy 基线",
              f"{len(uc['model'].unique())} 个模型 × {len(uc['metric'].unique())} 个指标")
        # 注意：百分位区间**不保证**包住原始点估计，所以不能拿「下界 ≤ 点估计」当检查。
        # 改查真正应当成立的性质：边界有限、次序正确、落在指标取值范围内、有效抽样数合法。
        ci = uc[uc["ci_lo"].notna() & uc["ci_hi"].notna()]
        bad_bounds = ci[(ci["ci_lo"] > ci["ci_hi"] + 1e-12)
                        | (ci["ci_lo"] < -1e-12) | (ci["ci_hi"] > 1 + 1e-12)]
        check(len(bad_bounds) == 0, 8, "置信区间边界有限且满足 lo≤hi、落在 [0,1]",
              f"异常 {len(bad_bounds)} 行")
        nv = uc["n_valid"]
        check(bool(((nv > 0) & (nv <= uc["n_boot"])).all()), 8, "有效抽样数在 (0, n_boot] 区间内",
              f"n_valid ∈ [{int(nv.min())}, {int(nv.max())}]，n_boot={int(uc['n_boot'].max())}")
        if "n_invalid" in uc.columns:
            n_inv = int(uc["n_invalid"].max())
            check(n_inv == 0, 8, "本轮 bootstrap 无缺类作废抽样（与报告「缺类作废数」一致）",
                  f"缺类作废数 {n_inv}（>0 时区间仅基于有效抽样，需在报告中说明）")

    if (results / "selection_estimate.json").exists():
        sel = json.loads((results / "selection_estimate.json").read_text(encoding="utf-8"))
        check(sel.get("n_samples") == 27 and len(sel.get("picked_per_fold", [])) == 5, 8,
              "含模型选择的估计覆盖 27 条样本、5 个外层折",
              f"n={sel.get('n_samples')}，折数={len(sel.get('picked_per_fold', []))}")

    # ---------- 输出 ----------
    print("=" * 78)
    print("PROJECT_BRIEF §8 运行验收核查（基于磁盘真实产物）")
    print("=" * 78)
    cur = None
    npass = 0
    for ok, item, title, detail in _checks:
        if item != cur:
            print(f"\n[验收项 {item}]")
            cur = item
        print(f"  {'PASS' if ok else 'FAIL'}  {title}" + (f"  —— {detail}" if detail else ""))
        npass += ok
    total = len(_checks)
    print("\n" + "=" * 78)
    print(f"结果：{npass}/{total} 通过")
    failed = [t for ok, _, t, _ in _checks if not ok]
    if failed:
        print("未通过项：")
        for t in failed:
            print(f"  - {t}")
    print("=" * 78)
    return 0 if npass == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
