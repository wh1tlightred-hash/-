# 中医声诊样本风险三分类项目

对疑似冠心病患者的中医声诊样本做风险三分类（低危=0 / 中危=1 / 高危=2），
实现并比较六种分类器与 Dummy 基线，进行嵌套交叉验证、超参数调优、指标评估、
SHAP 解释与完整模型交付；并在此基础上做了扩展分析：补充对比曲线、多种交叉验证方案对比与
置换检验、学习/验证曲线、概率校准、SHAP 蜂群/依赖/瀑布图、折外 SHAP 与六模型 SHAP 对比，
报告同时输出 Markdown / 单文件 HTML / Word。全部结果来自真实运行。

## 数据

- `inputs/声诊练习.xlsx`：三个工作表，共 **27 条**有效样本（三类各 9 条），10 个数值特征
  `FO, I, F1, F2, F3, F4, B1, B2, B3, B4`（表头 `FO` 为字母 O，原样保留）。
- `inputs/中医声诊实验.docx`：实验背景与参考范围（六分类器、五折 CV、Accuracy/P/R/F1/AUC/mAP、SHAP）。
- 原始附件保持不变；解析后的合并数据与来源映射见 `data/processed/`。

## 环境（已实测通过）

- 操作系统：Windows 11（AMD64）。解释器：**Python 3.14.7**（官方当前最新稳定版；3.15 仍为预发布）。
- 包管理器：本机使用 `uv`；虚拟环境可按下方命令在项目内创建为 `.venv/`。虚拟环境与下载缓存属于本机文件，不纳入 Git 交付。
- 关键版本：numpy 2.5.3 / pandas 3.0.6 / scipy 1.18.1 / scikit-learn 1.9.1 / matplotlib 3.11.2 /
  seaborn 0.13.2 / xgboost 3.4.1 / shap 0.52.0 / joblib 1.6.0 / pytest 9.1.1 / openpyxl 3.1.5 /
  python-docx 1.2.0（Word 报告导出）。
- `python -m pip check`：No broken requirements found。
- 完整环境核验见 `logs/environment_check.json` 与 `logs/pip_check.txt`；锁定依赖 `requirements-lock.txt`。

### 重建环境（Windows）

**推荐用锁定文件重建**（版本与本次交付一致）；按包名安装会随上游发布漂移，得到不同版本。

#### 方式 A：仅用 pip + 锁定文件（无需 uv）

```powershell
# PowerShell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

```bash
# Git Bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe -m pip check
```

#### 方式 B：用 uv 建环境（本次交付实际使用的方式）

```bash
# Git Bash（uv 需已在 PATH 中；本机位于 ~/AppData/Local/hermes/bin/uv.exe）
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
# uv 建的虚拟环境默认不带 pip；main.py 的环境核验要运行 `python -m pip check`，因此补装 pip
uv pip install --python .venv/Scripts/python.exe pip
./.venv/Scripts/python.exe -m pip check
```

若直连 PyPI 下载超时（例如本机代理未开启），可临时改用国内镜像，版本号仍按锁定文件安装：

```bash
uv pip install --python .venv/Scripts/python.exe \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-lock.txt
```

```powershell
# PowerShell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv .venv --python 3.14
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

说明：

- 本次实测环境为 **Python 3.14.7**（执行当日官方最新稳定版）；`requirements-lock.txt` 是 `pip freeze` 的完整结果，含间接依赖。
- 上面刻意不再写死本机的绝对解释器/工具路径；uv 不在 PATH 时自行指定其完整路径即可。
- `.uv-cache/`（约 483 MB）未随交付：它在**离线**重装依赖时才需要，联网重装不需要。
- Git 仓库不包含 `.venv/`；每台机器应按上面的锁定文件命令重建环境。

## 一键运行

```bash
.venv/Scripts/python.exe main.py                # 完整实验，本机实测约 9–12 分钟
.venv/Scripts/python.exe main.py --report-only  # 不重新训练，用上次结果重生成报告（约 10 秒）
```

`main.py` 依次完成（共 10 步）：数据加载与契约校验 → 环境核验 → 种子 42 主嵌套交叉验证（外层 5 折 / 内层 3 折）→
种子 43/44 切分稳定性 → 胜者选择、不确定性量化与基础图表 → **补充对比曲线** → **交叉验证深入分析** →
全量最终模型重拟合与保存 → SHAP（基础 + **扩展**）→ 中文报告（`report/实验报告.md` / `.html` / `.docx`）。

`--report-only` 读取 `results/report_context.joblib`（完整运行时保存），只重新渲染报告；
修改了 `src/report.py` / `src/report_export.py` 的版式后用它即可，不必重跑整个实验。

## 扩展分析（报告第 5–7 章）

扩展分析**不改动**主实验（种子、切分、内层评分、胜者判定全部不变），也不参与胜者选择。

| 章节 | 内容 | 主要产物 |
| --- | --- | --- |
| §5 补充对比曲线 | 调参前后 ROC、逐折 ROC ±1 标准差带、按风险类别的六模型 ROC/PR、逐折指标箱线图、多指标雷达图、训练-折外差距哑铃图、三种子稳定性 | `results/figures/roc_*.png`、`class_model_comparison.png`、`fold_metric_box.png`、`radar_metrics.png`、`train_vs_oof_gap.png`、`seed_stability.png` |
| §6 交叉验证深入分析 | 分层 3/5/9 折、留一法、重复分层 5 折×10 的方案对比；标签置换检验；分层学习曲线；单参数验证曲线；概率校准（Brier / log loss / ECE / 可靠性曲线） | `results/cv_analysis/*.csv`、`cv_schemes.png`、`cv_repeated_box.png`、`permutation_test.png`、`learning_curves.png`、`validation_curves.png`、`calibration.png` |
| §7 SHAP 深入分析 | 蜂群图、依赖图、代表样本瀑布图；**折外 SHAP**（每条样本由没见过它的折模型解释）与逐折排名稳定性；**六模型 SHAP 对比**（占比归一化） | `results/shap/*`、`shap_beeswarm*.png`、`shap_dependence.png`、`shap_waterfall.png`、`shap_cv_stability.png`、`shap_model_comparison.png` |

口径要点：

- §6.1–6.5 使用**默认超参数**的完整 Pipeline，不在全数据上先调参，因此无调参泄漏；6.6 使用主实验调参版折外概率（SVM 为决策分数，不参与）。
- 留一法下 Dummy 基线 Accuracy=0：三类严格均衡时，留出样本所属类在训练集中必为少数类。这是留一法的已知悲观偏差，报告中已说明。
- XGBoost 的 TreeExplainer 输出是**原始 margin**（非概率），代码与报告都如实标注；SVM 为决策分数。六模型对比因此先在各模型内归一化为占比。
- 本轮实测：6 个模型在置换检验中均未达到 p<0.05；6 个调参模型的高危类 OvR AUC 均低于 0.5（详见报告摘要）。

预算全部在 `config.json` 的 `analysis` 段配置（置换次数、重复次数、学习曲线每类样本数等）。

## 不确定性量化与选择乐观性（补充分析）

原报告只能**定性**地说「27 条样本统计功效低」。现在用已经算好的折外预测补上了三个具体数字，
**不改动**预先固定的主实验规则（种子、切分、内层评分、胜者判定全部不变）：

| 做了什么 | 结论 | 产物 |
| --- | --- | --- |
| 对 27 条折外预测做样本级 bootstrap（2000 次），给出各模型指标的 95% 百分位区间 | 胜者 Accuracy 区间约 `[0.37, 0.74]`、macro-F1 约 `[0.36, 0.70]`，各模型区间大量重叠 | `results/uncertainty.csv`、`results/figures/uncertainty.png` |
| 胜者与 Dummy 先验基线做**配对**精确 McNemar 检验 | 不一致对约 16 条，**p ≈ 0.077 > 0.05**：以本样本量尚不能认为胜者优于先验基线 | `results/mcnemar.json` |
| 每折只按**该折内层分数**选模型、再拼接折外预测（不含事后挑选） | 得 Accuracy 0.481 / macro-F1 0.462，低于事后挑出的 0.556 / 0.539。**这是两套不同策略，差值同时混合策略差异、样本波动与选择效应，不能当作选择偏差的大小**；能说的是结论对「用哪种方式选模型」敏感 | `results/selection_estimate.json` |

三点都写进报告 §4.11 与 §9.1。要点：

- bootstrap 是**样本级**重采样，不是人群抽样，也不能把种子 42/43/44 的重复预测当成更多患者；
  它**固定了已拟合的折外预测**，没有重跑训练/调参/切分，因此不是「完整建模流程泛化性能」的已证 95% 置信区间。
- 有效抽样规则：**一次抽样三类都出现才计入**，缺类则该次对全部指标作废（见 `uncertainty.csv` 的 `n_invalid` 列）。
- McNemar 应按**探索性比较**理解：其「精确」只指二项尾概率算法，且只比较错误率，不涉及 macro-F1 或 AUC。

## 新样本预测

```bash
# CSV（含同名 10 个特征列，顺序可乱、按名称重排）
.venv/Scripts/python.exe predict.py --csv 新样本.csv --out 预测结果.csv

# 内联 JSON
.venv/Scripts/python.exe predict.py --json '[{"FO":358.6,"I":46.2,"F1":897.7,"F2":1737.7,"F3":3025.7,"F4":4095.2,"B1":458.9,"B2":256.0,"B3":484.5,"B4":603.0}]'
```

输入只含 10 个特征；缺列、多列、非数值、无穷值都会明确报错；列顺序不同会安全重排。
输出类别名/编码与三类分数（概率模型为 predict_proba；SVM 为 decision_function 决策分数，均非疾病发生概率）。

## 测试

```bash
.venv/Scripts/python.exe -m pytest -q
```

覆盖：数据契约、切分泄漏与索引完整性、OOF 指标重算一致性、模型序列化预测一致性、错误输入校验、
以及不确定性量化（bootstrap 区间、配对检验、含模型选择的估计与绘图字段契约）。本次运行结果为 **74 passed**（含扩展分析测试 17 项）。
依赖 `main.py` 产物的用例在产物缺失时自动 skip；先运行 `main.py` 再跑 pytest 即全量执行。
`conftest.py` 为每次运行在项目内的 `.pytest_runs/` 创建独立临时目录，并在测试结束后清理。

## 验收核查

```bash
.venv/Scripts/python.exe verify_acceptance.py
```

按 `PROJECT_BRIEF.md` 第 8 节的验收要求，对**磁盘上已生成的真实产物**逐条核查（不重新训练、不推测）：
数据形状与原始附件哈希、外层/内层切分完整性、元数据未进入 X、六模型与 Dummy 均实际运行、
OOF 数量与分数列序、指标重算一致性、混淆矩阵合计 27、模型重载与错误输入、报告与图表产出，
以及补充分析产物与「报告中的解释器路径指向当前副本位置」。本次运行结果为 **67 项检查，67/67 通过**（含第 9 项扩展分析核查）。
全部通过时退出码为 0，任一不通过为 1 并打印未通过项。

原件目录默认取构建本项目时的 `C:\Users\swild\Downloads`。若仓库被克隆到别的机器或原件已清理，
脚本**不会崩溃**：会降级为核对「项目内文件哈希 == 运行期记录哈希」。需要时用环境变量
`TCM_ORIGINALS_DIR` 指定原件目录。

## 目录结构

```text
inputs/                  原始附件（保持不变）
src/                     数据、模型、评估、指标、绘图、解释、不确定性、报告模块
  cv_analysis.py         CV 方案对比 / 置换检验 / 学习曲线 / 验证曲线 / 校准
  curves.py              补充对比曲线
  shap_plots.py          SHAP 蜂群 / 依赖 / 瀑布 / 多模型热力图 / 折外对照
  extended.py            第 5–7 章编排（计算 → 落盘 → 生成报告小节）
  report_export.py       Markdown → 单文件 HTML（含目录）/ Word
tests/                   数据契约、切分、OOF 重算、序列化、错误输入、不确定性、扩展分析测试
data/processed/          合并数据、来源映射、标签映射、校验报告
results/                 OOF/逐折/参数/预算/图像/日志/最终模型/稳定性/不确定性
                         uncertainty.csv（bootstrap 区间）、mcnemar.json（配对检验）、
                         selection_estimate.json（含模型选择的估计）
                         cv_analysis/（§6 交叉验证深入分析）、shap/（§7 扩展 SHAP）
                         report_context.joblib（--report-only 使用）
models/                  最终管线 joblib 与元数据
report/                  实验报告.md（图像相对路径）/ 实验报告.html（图片内嵌、单文件）/ 实验报告.docx
config.json              种子、切分、搜索空间、资源预算
main.py                  一键实验入口
predict.py               新样本预测入口
requirements-lock.txt    锁定依赖
README.md / 查看指南.md
```

## 主要结论摘要

- 27 条样本下，六模型调参版 OOF 指标总体不高且方差大；详见 `report/实验报告.md`。
- 胜者按预先确定的 macro-F1 判定：**随机森林（OOF macro-F1=0.5386、Accuracy=0.5556）**。
  但它**只在部分指标上胜出**：macro ROC-AUC 与 mAP 最好的调参模型都是 **XGBoost**
  （AUC=0.6584、mAP=0.5425），不能说随机森林在所有指标上都最优。
- **最要紧的逐类弱点：高危类召回率仅 22.2%（2/9）**，7 条真实高危样本未被识别为高危；
  整体 Accuracy 主要由低危（77.8%）与中危（66.7%）支撑。报告 §4.8 已单独说明。
- 胜者相对 Dummy 先验基线的配对 McNemar 检验 **p = 0.0768 > 0.05**（不显著，且属探索性比较），
  bootstrap 95% 区间很宽且各模型互相重叠，Dummy 点估计落在几乎每个模型的区间内。
- 所有 OOF 指标为内部交叉验证结果，**不是**独立外部测试集，不构成临床部署或外部泛化结论。

## 已知限制

- 仅 27 条样本（每类 9），外层每折 5–6 条，统计功效低；bootstrap 95% 区间宽、各模型大量重叠，
  无法按点估计排出可信名次。
- 胜者 vs Dummy 配对检验不显著（p ≈ 0.077），尚不能证明有优于先验基线的判别能力；
  高危类几乎识别不出，不能把整体 Accuracy 当作可用性证据。
- 无患者 ID，无法排除同一人重复录音跨行的患者级泄漏；“每行独立”为暂定假设。
- SHAP 为训练后描述性分析，非因果、非外部验证；当前解释空间为原始特征空间
  （imputer 对无缺失数据是恒等变换，已在运行时验证并记入 `shap_meta.json`）。
- 交付的最终模型在全部 27 条上重新调参（选中 `max_depth=None`），训练集 Accuracy=1.000，
  是**演示用**的过拟合模型，不得用它的训练分数代替 OOF 性能。
- **与附件记录的旧环境不同**：附件《中医声诊实验.docx》记录的是 Python 3.7 / scikit-learn 0.21.2，
  本实现按交接说明「最新 Python 环境」的要求运行在 Python 3.14.7 / scikit-learn 1.9.1。
  若验收方按旧版本严格核对，需按上方「重建环境」另外准备，当前代码不能直接放进 Python 3.7。

## 配置项接线状态

`config.json` 中的参数并非全部参与运行时校验，为避免误解在此写明：

| 配置项 | 状态 |
| --- | --- |
| `seed` / `stability_seeds` / `outer_cv` / `inner_cv` / `inner_scoring` | 生效，主实验与稳定性流程直接使用 |
| `n_iter_random` / `n_jobs` / `zero_division` | 生效 |
| `max_candidates_per_model` | **生效（运行时硬上限）**：任一模型每折候选数超过即报错，不做静默截断（`src/eval.py::validate_candidate_budget`） |
| `shap.nsamples_kernel` | 生效（KernelExplainer 采样数） |
| `shap.background` / `shap.explain_all_train` | **描述性**：当前固定为「全部训练样本」背景 + 解释全部训练样本，改这两项不会改变行为 |
| `label_map` / `class_names` / `feature_columns` | 生效且被强校验（不匹配即报错） |
| `analysis.cv_schemes` / `permutation` / `learning_curve` / `validation_curve` / `calibration_bins` | 生效（§6 各分析的方案与预算） |
| `analysis.shap_cv` / `shap_all_models` / `shap_dependence_top_k` | 生效（§7 折外 SHAP、六模型对比开关，依赖图特征数） |
| `report.formats` / `report.html_embed_images` | 生效（是否输出 Word；HTML 是否内嵌图片） |
