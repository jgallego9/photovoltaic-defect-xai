"""Saved predictions must keep the split, image order and validation decisions."""

import numpy as np
import pytest

from photovoltaic_xai import CLASS_NAMES, PROTOCOL_ID
from photovoltaic_xai.metrics import compute_metrics
from photovoltaic_xai.models import checkpoint_hash, load_run, save_json


@pytest.fixture
def saved_reference(tmp_path):
    labels = np.array([[1, 0, 0, 0, 0, 0], [0] * 6], dtype=np.float32)
    logits = np.array([[1, -1, -1, -1, -1, -1], [-1] * 6], dtype=np.float32)
    prepared = {
        "data_fingerprint": "fixed-data",
        "records": [
            {"id": name, "split": "val", "labels": y.tolist()}
            for name, y in zip(("a", "b"), labels, strict=True)
        ],
    }
    run = {
        "protocol_id": PROTOCOL_ID,
        "class_names": list(CLASS_NAMES),
        "status": "complete",
        "completed_epochs": 2,
        "best_epoch": 1,
        "evaluation_precision": "float32",
        "config": {"model_name": "convnext_tiny"},
        "metadata": {"data_fingerprint": "fixed-data"},
        "metrics": {"thresholds": [0.9] * 6},
    }
    save_json(tmp_path / "run.json", run)
    (tmp_path / "best.pt").write_bytes(b"selected checkpoint")
    np.savez_compressed(
        tmp_path / "validation.npz",
        ids=np.array(["a", "b"]),
        labels=labels,
        logits=logits,
        checkpoint_sha256=checkpoint_hash(tmp_path),
    )
    return tmp_path, prepared


def test_validation_reports_preserve_thresholds_and_files(saved_reference):
    directory, prepared = saved_reference
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    run, prediction = load_run(directory, prepared)
    report = compute_metrics(
        prediction["labels"],
        prediction["probabilities"],
        run["metrics"]["thresholds"],
        logits=prediction["logits"],
    )
    assert report["thresholds"] == [0.9] * 6
    assert report["per_class"][0]["recall"] == 0
    assert report["healthy"]["false_positive_rate_any_label"] == 0
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("changed", ["ids", "labels", "data", "checkpoint"])
def test_results_reject_another_order_labels_or_preparation(saved_reference, changed):
    directory, prepared = saved_reference
    if changed == "data":
        prepared["data_fingerprint"] = "other-data"
    elif changed == "checkpoint":
        (directory / "best.pt").write_bytes(b"different checkpoint")
    else:
        with np.load(directory / "validation.npz", allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        if changed == "ids":
            values["ids"] = values["ids"][::-1]
        else:
            values["labels"] = 1 - values["labels"]
        np.savez_compressed(directory / "validation.npz", **values)
    before = (directory / "run.json").read_bytes()
    with pytest.raises(ValueError):
        load_run(directory, prepared)
    assert (directory / "run.json").read_bytes() == before
