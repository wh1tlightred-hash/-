"""分类器注册、管线构建与超参数搜索空间定义。

六种真实分类器 + DummyClassifier(strategy='prior') 基线。
搜索空间见 PROJECT_BRIEF.md 第 5 节；LR/SVM/GNB/KNN 用 GridSearchCV，
RF/XGBoost 空间较大，用固定随机种子的 RandomizedSearchCV，每折最多约 20 个候选。
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 显示名（中文 / 英文），用于表格与图表图例
MODEL_LABELS = {
    "logistic": "Logistic Regression (逻辑回归)",
    "svm": "SVM (支持向量机)",
    "gnb": "Gaussian Naive Bayes (高斯朴素贝叶斯)",
    "knn": "KNN (K 近邻)",
    "random_forest": "Random Forest (随机森林)",
    "xgboost": "XGBoost",
    "dummy": "Dummy (先验基线)",
}

# 简短显示名：用于表格与图表轴标签。
# 不能对 MODEL_LABELS 直接取空格前第一段——那会把 Random Forest 截成 "Random"、
# Gaussian Naive Bayes 截成 "Gaussian"，表格里看起来像另一个模型。
MODEL_SHORT = {
    "logistic": "逻辑回归",
    "svm": "SVM",
    "gnb": "高斯朴素贝叶斯",
    "knn": "KNN",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
    "dummy": "Dummy 基线",
}

# 版本显示名（中文）
VERSION_SHORT = {"untuned": "未调参", "tuned": "调参", "untrained": "基线"}

# 需要缩放的模型：线性/距离型模型至少使用 StandardScaler；树与朴素贝叶斯无需缩放。
REQUIRES_SCALER = {"logistic", "svm", "knn"}

# 并列时的简洁性排序（预先固定，用于胜者并列 tie-break，不事后更改）
SIMPLICITY_ORDER = ["logistic", "gnb", "knn", "svm", "random_forest", "xgboost"]

REAL_MODELS = ["logistic", "svm", "gnb", "knn", "random_forest", "xgboost"]
ALL_MODELS = REAL_MODELS + ["dummy"]


def make_estimator(name: str, seed: int, n_jobs: int = 1, params: dict | None = None):
    """按模型名构造分类器实例。params 为可覆盖的构造参数（来自搜索空间）。"""
    params = dict(params or {})
    if name == "logistic":
        base = dict(solver="lbfgs", max_iter=2000, random_state=seed)
        return LogisticRegression(**{**base, **params})
    if name == "svm":
        # 不用 probability=True：ROC/PR 使用 decision_function 决策分数，不进行概率校准。
        base = dict(random_state=seed)
        return SVC(**{**base, **params})
    if name == "gnb":
        return GaussianNB(**params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "random_forest":
        base = dict(random_state=seed, n_jobs=n_jobs)
        return RandomForestClassifier(**{**base, **params})
    if name == "xgboost":
        base = dict(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=seed,
            tree_method="hist",
            n_jobs=n_jobs,
        )
        return XGBClassifier(**{**base, **params})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    raise KeyError(f"未知模型名：{name}")


def build_pipeline(name: str, seed: int, n_jobs: int = 1, params: dict | None = None) -> Pipeline:
    """构建（未拟合的）完整 Pipeline：缺失填补 [+ 缩放] + 分类器。

    所有拟合型预处理都位于 Pipeline 内部，保证只会在当前训练折内拟合。
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in REQUIRES_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", make_estimator(name, seed=seed, n_jobs=n_jobs, params=params)))
    return Pipeline(steps)


def get_search_space(name: str, min_inner_train_size: int) -> tuple[dict | list, str]:
    """返回 (参数网格, 搜索类型)。参数名以 ``clf__`` 为前缀（作用于 Pipeline）。

    min_inner_train_size 用于 KNN 的邻居数上限约束（邻居数不得超过最小内层训练集大小）。
    """
    if name == "logistic":
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        return grid, "grid"
    if name == "svm":
        # 条件搜索：linear 无 gamma；rbf 才有 gamma，避免无意义组合。
        grid = [
            {"clf__kernel": ["linear"], "clf__C": [0.1, 1.0, 10.0, 100.0]},
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", 0.001, 0.01, 0.1],
            },
        ]
        return grid, "grid"
    if name == "gnb":
        grid = {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]}
        return grid, "grid"
    if name == "knn":
        neighbors = [n for n in [1, 3, 5, 7] if n <= min_inner_train_size]
        if not neighbors:
            neighbors = [1]
        grid = {
            "clf__n_neighbors": neighbors,
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
        return grid, "grid"
    if name == "random_forest":
        grid = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [2, 3, 5, None],
            "clf__min_samples_leaf": [1, 2, 3],
            "clf__max_features": ["sqrt", 1.0],
        }
        return grid, "random"
    if name == "xgboost":
        grid = {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [1, 2, 3],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1.0, 10.0],
        }
        return grid, "random"
    raise KeyError(f"未知模型名：{name}")


def grid_n_candidates(grid) -> int:
    """统计参数网格的理论候选数（用于预算记录）。list-of-dict 累加各分支。"""
    grids = grid if isinstance(grid, list) else [grid]
    total = 0
    for g in grids:
        n = 1
        for v in g.values():
            n *= len(v)
        total += n
    return total
