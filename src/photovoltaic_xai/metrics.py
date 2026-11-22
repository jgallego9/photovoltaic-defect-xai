"""Six-label prediction reports and validation thresholds."""

import warnings

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from . import CLASS_NAMES


def _prediction_arrays(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if (
        labels.ndim != 2
        or labels.shape[1] != 6
        or not len(labels)
        or probabilities.shape != labels.shape
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Expected binary labels and probabilities in [0,1], both [N,6]")
    return labels, probabilities


def _average_precision(labels, scores):
    # AP=0 for absent positives; reports retain an explicit missing-positive flag.
    return np.asarray(
        [
            average_precision_score(labels[:, c], scores[:, c]) if labels[:, c].any() else 0.0
            for c in range(6)
        ]
    )


def optimize_thresholds(labels, probabilities, *, split="val"):
    """Choose validation F1 thresholds, breaking ties toward 0.5."""
    if split != "val":
        raise ValueError("Threshold selection is restricted to validation (val)")
    labels, probabilities = _prediction_arrays(labels, probabilities)
    thresholds = np.full(6, 0.5)
    candidates = np.linspace(0.05, 0.95, 19)
    for c, name in enumerate(CLASS_NAMES):
        positives = labels[:, c].sum()
        if positives < 2 or positives == len(labels):
            warnings.warn(
                f"Keeping threshold 0.5 for {name}: insufficient validation support.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        scores = np.asarray(
            [
                f1_score(labels[:, c], probabilities[:, c] >= value, zero_division=0)
                for value in candidates
            ]
        )
        best = candidates[np.isclose(scores, scores.max(), rtol=0, atol=1e-12)]
        thresholds[c] = best[np.argmin(abs(best - 0.5))]
    return thresholds


def compute_metrics(labels, probabilities, thresholds=0.5, *, logits=None):
    """Use logits for AP when available and probabilities for threshold decisions."""
    labels, probabilities = _prediction_arrays(labels, probabilities)
    scores = probabilities
    if logits is not None:
        # Sigmoid saturation can tie distinct logits and change their ranking.
        scores = np.asarray(logits)
        try:
            valid_scores = scores.shape == labels.shape and np.isfinite(scores).all()
        except TypeError:
            valid_scores = False
        if not valid_scores:
            raise ValueError("Logits must be finite and have the same [N,6] shape as labels")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape == ():
        thresholds = np.full(6, thresholds)
    if (
        thresholds.shape != (6,)
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError("Thresholds must be a probability or six probabilities")
    predicted = probabilities >= thresholds
    ap = _average_precision(labels, scores)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        average=None,
        zero_division=0,
    )
    micro = precision_recall_fscore_support(labels, predicted, average="micro", zero_division=0)
    cardinality = labels.sum(1)
    healthy = cardinality == 0
    fpr = float(predicted[healthy].any(1).mean()) if healthy.any() else None
    strata = {}
    for name, selected in (
        ("single_type", cardinality == 1),
        ("multilabel", cardinality > 1),
    ):
        count = int(selected.sum())
        positives = labels[selected].sum(0)
        report = {
            "num_images": count,
            "positive_support": dict(zip(CLASS_NAMES, positives.tolist(), strict=True)),
            "subset_accuracy": None,
            "hamming_loss": None,
            "f1_micro": None,
            "f1_macro": None,
            "map": None,
        }
        if count:
            report.update(
                subset_accuracy=float(accuracy_score(labels[selected], predicted[selected])),
                hamming_loss=float(hamming_loss(labels[selected], predicted[selected])),
            )
            local_ap = _average_precision(labels[selected], scores[selected])
            report.update(
                f1_micro=float(
                    f1_score(
                        labels[selected], predicted[selected], average="micro", zero_division=0
                    )
                ),
                f1_macro=float(
                    f1_score(
                        labels[selected], predicted[selected], average="macro", zero_division=0
                    )
                ),
                map=float(local_ap.mean()) if positives.all() else None,
            )
        strata[name] = report
    return {
        "num_samples": len(labels),
        "thresholds": thresholds.tolist(),
        "overall": {
            "num_classes": 6,
            "map": float(ap.mean()),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
            "precision_micro": float(micro[0]),
            "recall_micro": float(micro[1]),
            "f1_micro": float(micro[2]),
            "hamming_loss": float(hamming_loss(labels, predicted)),
            "subset_accuracy": float(accuracy_score(labels, predicted)),
        },
        "per_class": [
            {
                "class": name,
                "positive_support": int(support[c]),
                "negative_support": int(len(labels) - support[c]),
                "average_precision": float(ap[c]),
                "ap_missing_positives": bool(support[c] == 0),
                "threshold": float(thresholds[c]),
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
            }
            for c, name in enumerate(CLASS_NAMES)
        ],
        "healthy": {
            "num_samples": int(healthy.sum()),
            "false_positive_rate_any_label": fpr,
            "specificity": 1 - fpr if fpr is not None else None,
        },
        "strata": strata,
    }


BOOTSTRAP_METRICS = (
    "map",
    "f1_macro",
    "f1_micro",
    "subset_accuracy",
    "multilabel_subset_accuracy",
    "crack_ap",
    "healthy_false_positive_rate",
)


class WeightedMetrics:
    """Evaluate multiplicities without allocating duplicated image rows."""

    def __init__(self, labels, logits, thresholds):
        if labels.shape != logits.shape:
            raise ValueError("Labels and logits must have the same shape")
        if logits.ndim != 2 or logits.shape[1] == 0:
            raise ValueError("Expected image-by-class logits")
        self.labels = np.asarray(labels, dtype=bool)
        self.logits = np.asarray(logits)
        self.predicted = torch.from_numpy(self.logits).sigmoid().numpy() >= thresholds
        self.tp = self.predicted & self.labels
        self.fp = self.predicted & ~self.labels
        self.fn = ~self.predicted & self.labels
        self.correct = (self.predicted == self.labels).all(axis=1)
        self.multilabel = self.labels.sum(axis=1) > 1
        self.healthy = self.labels.sum(axis=1) == 0
        self.healthy_alarm = self.healthy & self.predicted.any(axis=1)

    def __call__(self, weights):
        weights = np.asarray(weights)
        if (
            weights.shape != (len(self.labels),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Image multiplicities must be finite, nonnegative and nonempty")
        ap = [
            average_precision_score(labels, logits, sample_weight=weights)
            if weights[labels].sum()
            else np.nan
            for labels, logits in zip(self.labels.T, self.logits.T, strict=True)
        ]
        w = weights[:, None]
        tp, fp, fn = ((w * a).sum(axis=0) for a in (self.tp, self.fp, self.fn))
        denominator = 2 * tp + fp + fn
        f1 = np.divide(
            2 * tp, denominator, out=np.zeros(self.labels.shape[1]), where=denominator > 0
        )
        multilabel_n = weights[self.multilabel].sum()
        healthy_n = weights[self.healthy].sum()
        return np.array(
            [
                np.mean(ap),
                f1.mean(),
                2 * tp.sum() / denominator.sum() if denominator.sum() else 0.0,
                np.dot(weights, self.correct) / weights.sum(),
                np.dot(weights, self.correct & self.multilabel) / multilabel_n
                if multilabel_n
                else np.nan,
                ap[0],
                np.dot(weights, self.healthy_alarm) / healthy_n if healthy_n else np.nan,
            ]
        )


def sample_group_weights(inverse, num_groups, rng):
    """Draw G groups uniformly with replacement and retain every member."""
    multiplicities = np.bincount(rng.integers(num_groups, size=num_groups), minlength=num_groups)
    return multiplicities[np.asarray(inverse)]


def paired_group_bootstrap(labels, group_ids, variant, reference, *, replicates, seed):
    """Compare two (logits, thresholds) pairs on the same resampled test groups."""
    groups, inverse = np.unique(group_ids, return_inverse=True)
    score_variant = WeightedMetrics(labels, *variant)
    score_reference = WeightedMetrics(labels, *reference)
    ones = np.ones(len(labels), dtype=int)
    observed = score_variant(ones) - score_reference(ones)

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        weights = sample_group_weights(inverse, len(groups), rng)
        differences.append(score_variant(weights) - score_reference(weights))

    rows = []
    for metric, difference, samples in zip(
        BOOTSTRAP_METRICS, observed, np.asarray(differences).T, strict=True
    ):
        valid = samples[np.isfinite(samples)]
        lower, upper = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
        rows.append(
            {
                "metric": metric,
                "difference_pp": 100 * difference,
                "ci95_lower_pp": 100 * lower,
                "ci95_upper_pp": 100 * upper,
                "valid_replicates": len(valid),
            }
        )
    return rows
