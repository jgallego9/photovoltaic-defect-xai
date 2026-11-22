"""Ranking remains informative after sigmoid saturation; decisions stay frozen."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score

from photovoltaic_xai.metrics import (
    WeightedMetrics,
    compute_metrics,
    optimize_thresholds,
    sample_group_weights,
)


def without_average_precision(report):
    report = deepcopy(report)
    del report["overall"]["map"]
    for row in report["per_class"]:
        del row["average_precision"]
    for row in report["strata"].values():
        del row["map"]
    return report


def test_logits_recover_ranking_without_changing_threshold_decisions():
    # Both cardinality strata contain all six classes; one row is healthy.
    labels = np.vstack([np.zeros((1, 6)), np.eye(6), np.ones((1, 6)), 1 - np.eye(6)])
    logits = np.where(labels, 30, 20).astype(np.float32)
    probabilities = 1 / (1 + np.exp(-logits))
    assert (probabilities == 1).all()
    thresholds = optimize_thresholds(labels, probabilities)
    saved_thresholds = thresholds.copy()

    legacy = compute_metrics(labels, probabilities, thresholds)
    ranked = compute_metrics(labels, probabilities, thresholds, logits=logits)

    assert legacy["overall"]["map"] < 1
    assert ranked["overall"]["map"] == 1
    assert ranked["strata"]["single_type"]["map"] == 1
    assert ranked["strata"]["multilabel"]["map"] == 1
    assert all(row["average_precision"] == 1 for row in ranked["per_class"])
    assert without_average_precision(legacy) == without_average_precision(ranked)
    np.testing.assert_array_equal(thresholds, saved_thresholds)


def test_logits_preserve_absent_classes_and_undefined_strata():
    labels = np.zeros((3, 6))
    labels[0, :2] = 1
    labels[1, 0] = 1
    probabilities = np.full_like(labels, 0.5)
    logits = np.where(labels, 3, -3)
    report = compute_metrics(labels, probabilities, logits=logits)

    assert report["overall"]["map"] == pytest.approx(2 / 6)
    assert report["strata"]["multilabel"]["map"] is None
    assert report["strata"]["single_type"]["map"] is None
    for row in report["per_class"][2:]:
        assert row["average_precision"] == 0
        assert row["ap_missing_positives"] is True
    assert compute_metrics(labels, probabilities) == compute_metrics(
        labels, probabilities, logits=None
    )


@pytest.mark.parametrize(
    "logits",
    [
        np.zeros((3, 5)),
        np.full((3, 6), np.nan),
    ],
)
def test_invalid_logits_are_rejected(logits):
    with pytest.raises(ValueError, match="Logits must be finite"):
        compute_metrics(np.zeros((3, 6)), np.full((3, 6), 0.5), logits=logits)


def test_logits_do_not_bypass_probability_validation():
    with pytest.raises(ValueError, match="probabilities in"):
        compute_metrics(np.zeros((3, 6)), np.full((3, 6), 2), logits=np.zeros((3, 6)))


def test_bootstrap_weighted_metrics_match_literal_image_repetition():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    logits = np.array([[1, 0], [1, 0], [1, 1], [-1, -1], [0, 1], [0, 1]], dtype=np.float32)
    scorer = WeightedMetrics(labels, logits, [0.5, 0.5])
    for weights in (np.ones(6, dtype=int), np.array([0, 3, 2, 4, 1, 0])):
        selected = np.repeat(np.arange(6), weights)
        y = labels[selected]
        predicted = torch.from_numpy(logits[selected]).sigmoid().numpy() >= 0.5
        ap = average_precision_score(y, logits[selected], average=None)
        correct = (y == predicted).all(axis=1)
        expected = [
            ap.mean(),
            f1_score(y, predicted, average="macro"),
            f1_score(y, predicted, average="micro"),
            correct.mean(),
            correct[y.sum(axis=1) > 1].mean(),
            ap[0],
            predicted[y.sum(axis=1) == 0].any(axis=1).mean(),
        ]
        np.testing.assert_allclose(scorer(weights), expected, atol=2e-14, rtol=2e-14)


def test_bootstrap_keeps_whole_groups_with_identical_paired_multiplicities():
    inverse = np.array([0, 0, 1, 2, 2, 2])
    rng, expected_rng = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(20):
        expected = np.bincount(expected_rng.integers(3, size=3), minlength=3)[inverse]
        np.testing.assert_array_equal(sample_group_weights(inverse, 3, rng), expected)


def test_thresholds_are_selected_only_in_validation():
    labels = np.vstack([np.eye(6), np.eye(6), np.zeros((1, 6))])
    probabilities = np.where(labels, 0.8, 0.1)
    with pytest.raises(ValueError, match="validation"):
        optimize_thresholds(labels, probabilities, split="test")
    thresholds = optimize_thresholds(labels, probabilities)
    report = compute_metrics(labels, probabilities, thresholds)
    assert report["overall"]["f1_macro"] == 1
    assert report["healthy"]["false_positive_rate_any_label"] == 0
