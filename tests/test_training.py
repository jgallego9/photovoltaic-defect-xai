"""Exercise visible reference training on a small, deterministic corpus."""

import json
import math
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler

from photovoltaic_xai import CLASS_NAMES, PROTOCOL_ID
from photovoltaic_xai.metrics import compute_metrics, optimize_thresholds
from photovoltaic_xai.models import (
    checkpoint_hash,
    load_run,
    parameter_groups,
    predict,
    recalibrate_batchnorm,
    save_run,
    seed_everything,
    warmup_cosine,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyReference(nn.Module):
    def __init__(self, batch_norm):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4) if batch_norm else nn.Identity(), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(4, 6))

    def forward(self, images):
        return self.classifier(self.features(images).mean((-2, -1)))


class EpochSampler(SequentialSampler):
    def set_epoch(self, epoch):
        pass


@pytest.mark.parametrize("batch_norm", [False, True])
def test_visible_training_selects_checkpoint_and_uses_clean_batchnorm(tmp_path, batch_norm):
    images = torch.arange(7 * 3 * 4 * 4).reshape(7, 3, 4, 4).float() / 336
    labels = torch.cat([torch.eye(6), torch.zeros(1, 6)])
    clean = [
        {"image": x, "labels": y, "id": f"val-{i}"}
        for i, (x, y) in enumerate(zip(images, labels, strict=True))
    ]
    augmented = [{**row, "image": row["image"] + 0.4} for row in clean]
    config = {
        "seed": 42,
        "batch_size": 3,
        "effective_batch_size": 6,
        "max_epochs": 1,
        "warmup_epochs": 0,
        "patience": 5,
        "amp": False,
        "image_size": 4,
        "lr_backbone": 1e-3,
        "lr_head": 2e-3,
        "weight_decay": 0.01,
    }
    train_loader = DataLoader(augmented, batch_size=3, sampler=EpochSampler(augmented))
    calibration_loader = DataLoader(clean, batch_size=3)
    val_loader = DataLoader(clean, batch_size=3)
    prepared = {
        "data_fingerprint": "tiny-training",
        "records": [
            {"id": row["id"], "labels": row["labels"].tolist(), "split": "val"} for row in clean
        ],
    }

    def create_model(name, *, weights):
        model = TinyReference(batch_norm)
        model.classifier[-1] = nn.Linear(4, 6)
        return model

    namespace = dict(
        math=math,
        time=time,
        np=np,
        pd=pd,
        torch=torch,
        nn=nn,
        F=F,
        AdamW=AdamW,
        CLASS_NAMES=CLASS_NAMES,
        PROTOCOL_ID=PROTOCOL_ID,
        compute_metrics=compute_metrics,
        optimize_thresholds=optimize_thresholds,
        load_run=load_run,
        parameter_groups=parameter_groups,
        predict=predict,
        recalibrate_batchnorm=recalibrate_batchnorm,
        warmup_cosine=warmup_cosine,
        save_run=save_run,
        seed_everything=seed_everything,
        CONFIG=config,
        RUN_DIR=tmp_path,
        device=torch.device("cpu"),
        prepared=prepared,
        train_records=clean,
        train_loader=train_loader,
        calibration_loader=calibration_loader,
        val_loader=val_loader,
        create_model=create_model,
        checkpoint_hash=checkpoint_hash,
    )
    notebook = nbformat.read(ROOT / "notebooks/02_reference_models.ipynb", as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.id == "references-training")
    exec(compile(source, "reference-training", "exec"), namespace)
    train_reference = namespace["train_reference"]
    with pytest.warns(RuntimeWarning, match="insufficient validation support"):
        run, history = train_reference("efficientnet_b0", None)
    directory = tmp_path / "efficientnet_b0"
    assert run == json.loads((directory / "run.json").read_text())
    assert run["status"] == "complete" and run["best_epoch"] == 1
    assert run["completed_updates"] == 2
    assert run["selection_ranking"] == "sigmoid-float32"
    trained = TinyReference(batch_norm)
    trained.load_state_dict(torch.load(directory / "best.pt", weights_only=True))
    if batch_norm:
        expected_mean = trained.features[0](images).mean((0, 2, 3))
        torch.testing.assert_close(trained.features[1].running_mean, expected_mean)
        assert history.bn_calibration_images.iloc[0] == 7
    else:
        seed_everything(42)
        expected = TinyReference(False)
        expected.classifier[-1] = nn.Linear(4, 6)
        optimizer = AdamW(parameter_groups(expected, 1e-3, 2e-3), weight_decay=0.01)
        for index, selected in enumerate((slice(0, 6), slice(6, 7))):
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(
                expected(images[selected] + 0.4), labels[selected]
            ).backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] *= 0.5 if index == 0 else 0
        for actual, wanted in zip(trained.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-5)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    namespace["create_model"] = lambda *_, **__: pytest.fail("A completed run retrained")
    cached_run, cached_history = train_reference("efficientnet_b0", None)
    assert cached_run == run
    pd.testing.assert_frame_equal(cached_history, history)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
