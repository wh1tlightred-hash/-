"""Independent regression checks for issues found during review."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from src.metrics import pr_auc_trapezoid
from predict import validate_and_reorder
from main import _winner


def test_pr_auc_preserves_vertical_segments():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    p, r, _ = precision_recall_curve(y, scores)
    expected = auc(r, p)
    assert pr_auc_trapezoid(y, scores) == pytest.approx(expected)
    assert expected != pytest.approx(average_precision_score(y, scores))


@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_prediction_rejects_nonfinite(value):
    with pytest.raises(ValueError, match='无穷'):
        validate_and_reorder(pd.DataFrame({'FO': [value]}), ['FO'])


def test_prediction_rejects_empty():
    with pytest.raises(ValueError, match='为空'):
        validate_and_reorder(pd.DataFrame(columns=['FO']), ['FO'])


def test_prediction_rejects_duplicate_headers():
    with pytest.raises(ValueError, match='重复'):
        validate_and_reorder(pd.DataFrame([[1, 2]], columns=['FO', 'FO']), ['FO'])


def test_winner_tie_break_prioritizes_accuracy_before_simplicity():
    table = pd.DataFrame([
        {'model': 'logistic', 'version': 'tuned', 'f1_macro': .5, 'accuracy': .4},
        {'model': 'random_forest', 'version': 'tuned', 'f1_macro': .5, 'accuracy': .6},
    ])
    assert _winner(table, {})['f1_best'] == 'random_forest'
