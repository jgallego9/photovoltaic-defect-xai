"""Scientific checks for regional data, frozen branches and attention fusion."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from photovoltaic_xai import CLASS_NAMES
from photovoltaic_xai.dataset import PairedELTransform
from photovoltaic_xai.variant import (
    GlobalRegionalClassifier,
    RegionDataset,
    RegionFusion,
    VariantDataset,
    balanced_attention_loss,
    build_regions,
    file_hash,
    load_feature_cache,
    pair_balance_weights,
    regional_targets,
    tile_geometry,
)


def image_record(tmp_path, split="val"):
    path = tmp_path / "cell.png"
    pixels = np.random.default_rng(7).integers(0, 256, (1024, 1024), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return {
        "id": "cell",
        "split": split,
        "image_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": 1024,
        "height": 1024,
        "localization_valid": True,
        "labels": [1, 0, 0, 0, 0, 0],
        "boxes": [{"class_name": "crack", "xmin": 0, "ymin": 0, "xmax": 20, "ymax": 20}],
    }


def test_anchors_preserve_context_and_exclude_ambiguous_overlap(tmp_path):
    record = image_record(tmp_path, split="train")
    region = build_regions([record])[0]
    assert region["crop_bounds"] == (0, 0, 128, 128)
    assert region["anchor_bounds"] == (0, 0, 20, 20)
    record["boxes"].append({"class_name": "finger", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30})
    assert build_regions([record]) == []
    with pytest.raises(PermissionError):
        build_regions([{**record, "split": "test"}])


def test_native_and_reduced_views_share_anchor_and_exclude_test(tmp_path):
    record = image_record(tmp_path)
    regions = build_regions([record])
    transform = PairedELTransform(224, mean=(0, 0, 0), std=(1, 1, 1))
    native = RegionDataset(regions, {"cell": record}, tmp_path, transform, view="native")[0]
    reduced = RegionDataset(regions, {"cell": record}, tmp_path, transform, view="reduced")[0]
    assert native["id"] == reduced["id"] == "cell:box0"
    assert native["target"] == reduced["target"] == 0
    assert native["image"].shape == reduced["image"].shape == (3, 224, 224)
    assert native["image"].std() > reduced["image"].std()
    with pytest.raises(PermissionError):
        RegionDataset(
            regions, {"cell": {**record, "split": "test"}}, tmp_path, transform, view="native"
        )
    with pytest.raises(ValueError, match="training-only"):
        RegionDataset(regions, {"cell": record}, tmp_path, transform, view="mixed")


def test_spatial_supervision_uses_training_boxes_and_known_negatives(tmp_path):
    bounds, positions = tile_geometry()
    assert bounds.shape == positions.shape == (25, 4)
    box = {"class_name": "crack", "xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}
    targets, known = regional_targets([box, box], [1, 0, 0, 0, 0, 0], True, bounds)
    assert targets[0, 0] == 1 and known[0, 0]
    assert not known[-1, 0] and known[:, 1:].all()
    _, invalid = regional_targets([box], [1, 0, 0, 0, 0, 0], False, bounds)
    assert not invalid.any()
    record = image_record(tmp_path)
    sample = VariantDataset([record], tmp_path, {"mean": [0] * 3, "std": [1] * 3})[0]
    assert not sample["region_known"].any()


def test_null_attention_reproduces_global_logits_and_has_finite_gradients():
    torch.manual_seed(42)
    fusion = RegionFusion(dimension=4, hidden=4)
    global_features, global_logits = torch.randn(2, 4), torch.randn(2, 6)
    valid = torch.tensor([[True, False], [False, False]])
    local = torch.randn(2, 2, 4).masked_fill(~valid[..., None], torch.nan)
    output = fusion(global_features, global_logits, local, torch.zeros(2, 4), valid)
    torch.testing.assert_close(output["logits"], global_logits, rtol=0, atol=0)
    assert output["attention"][1, :, -1].eq(1).all()
    assert output["attention"][0, :, 1].eq(0).all()
    targets = torch.zeros(2, 2, 3)
    known = valid[..., None].expand_as(targets)
    loss = balanced_attention_loss(
        output["log_attention"], targets, known, valid, pair_balance_weights([1] * 3, [1] * 3, 2)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in fusion.parameters() if p.grad is not None)


class TinyEncoder(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, classes))

    def forward(self, images):
        return self.classifier(self.avgpool(self.features(images)))


def test_full_classifier_exposes_unchanged_global_control_with_frozen_encoders():
    model = GlobalRegionalClassifier(TinyEncoder(), TinyEncoder(), RegionFusion(4, 4)).train()
    before = {k: v.clone() for k, v in model.global_encoder.state_dict().items()}
    output = model(
        torch.rand(2, 3, 8, 8), torch.rand(2, 25, 3, 8, 8), torch.ones(2, 25, dtype=torch.bool)
    )
    torch.testing.assert_close(output["logits"], output["global_logits"], rtol=0, atol=0)
    assert not model.global_encoder.training and not model.regional_encoder.training
    assert not any(p.requires_grad for p in model.global_encoder.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.global_encoder.state_dict().items())


def test_regional_cache_checks_mmap_weights_ids_and_train_supervision(tmp_path):
    records = [{"id": "a", "labels": [1, 0, 0, 0, 0, 0], "split": "train"}]
    values = {
        "ids": np.asarray(["a"]),
        "labels": np.asarray([records[0]["labels"]], dtype=np.float32),
        "local_features": np.zeros((1, 25, 4), dtype=np.float32),
        "tile_valid": np.ones((1, 25), dtype=bool),
        "region_targets": np.zeros((1, 25, 3), dtype=np.float32),
        "region_known": np.ones((1, 25, 3), dtype=bool),
    }
    for name, value in values.items():
        np.save(tmp_path / f"{name}.npy", value)
    metadata = {
        "status": "complete",
        "identity": {
            "features": {
                "sources": {
                    "regional_weights": "frozen-regional",
                    "producer": "informational-source-hash",
                }
            }
        },
        "sha256": {p.name: file_hash(p) for p in tmp_path.glob("*.npy")},
    }
    (tmp_path / "cache.json").write_text(json.dumps(metadata))
    cached = load_feature_cache(tmp_path, records, 4, {"regional_weights": "frozen-regional"})
    assert isinstance(cached["local_features"], np.memmap)
    with pytest.raises(ValueError, match="different encoder weights"):
        load_feature_cache(tmp_path, records, 4, {"regional_weights": "different"})
    with pytest.raises(ValueError, match="identifiers disagree"):
        load_feature_cache(
            tmp_path,
            [{**records[0], "id": "different-image"}],
            4,
            {"regional_weights": "frozen-regional"},
        )
    with pytest.raises(ValueError, match="Evaluation data"):
        load_feature_cache(
            tmp_path, [{**records[0], "split": "val"}], 4, {"regional_weights": "frozen-regional"}
        )


def execute_scientific_cell(identifier, **variables):
    import math
    import time

    from sklearn.metrics import f1_score

    from photovoltaic_xai import metrics, models, variant

    notebook = Path(__file__).resolve().parents[1] / "notebooks/04_global_regional_variant.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells if cell["id"] == identifier)
    namespace = {
        **globals(),
        "math": math,
        "time": time,
        "f1_score": f1_score,
        "AdamW": torch.optim.AdamW,
        "warmup_cosine": models.warmup_cosine,
        "attention_targets": variant.attention_targets,
        "F": torch.nn.functional,
        "parameter_groups": models.parameter_groups,
        "save_run": models.save_run,
        "seed_everything": models.seed_everything,
        "compute_metrics": metrics.compute_metrics,
        "optimize_thresholds": metrics.optimize_thresholds,
        "CachedViews": variant.CachedViews,
        "masked_region_loss": variant.masked_region_loss,
        "LOCAL_CLASS_NAMES": variant.LOCAL_CLASS_NAMES,
        "LOCAL_CLASS_INDICES": variant.LOCAL_CLASS_INDICES,
        "device": torch.device("cpu"),
        "display": lambda *_: None,
        **variables,
    }
    exec(compile(source, f"notebook04:{identifier}", "exec"), namespace)
    return namespace


def test_diagnostic_cohort_is_read_without_old_search_models(tmp_path):
    from photovoltaic_xai.models import save_json

    positive = [1, 0, 0, 0, 0, 0]
    prepared = {
        "data_fingerprint": "synthetic-cohort",
        "normalization": {},
        "records": [
            {"id": "train", "split": "train", "labels": positive},
            {"id": "val-crack", "split": "val", "labels": positive},
            {"id": "val-healthy", "split": "val", "labels": [0] * 6},
        ],
    }
    cohort = {"data": {"data_fingerprint": "synthetic-cohort"}, "cohort": {"ids": ["val-crack"]}}
    cohort_path = tmp_path / "docs/research/evidence/hard-crack-cohort.json"
    global_path = tmp_path / "outputs/experiments/convnext_tiny/run.json"
    save_json(cohort_path, cohort)
    save_json(
        global_path,
        {
            "status": "complete",
            "metadata": {"data_fingerprint": "synthetic-cohort"},
            "class_names": list(CLASS_NAMES),
            "metrics": {"thresholds": [0.5] * 6},
        },
    )
    inputs = {
        "PROJECT_ROOT": tmp_path,
        "load_preparation": lambda _: prepared,
        "save_json": save_json,
    }
    result = execute_scientific_cell("variant-data", **inputs)
    assert result["hard_ids"] == ["val-crack"]
    assert result["expected_ids"].tolist() == ["val-crack", "val-healthy"]
    assert {p for p in (tmp_path / "outputs").rglob("*") if p.is_file()} == {global_path}

    cohort["data"]["data_fingerprint"] = "different-data"
    save_json(cohort_path, cohort)
    with pytest.raises(ValueError, match="another prepared dataset"):
        execute_scientific_cell("variant-data", **inputs)
    cohort["data"]["data_fingerprint"] = "synthetic-cohort"
    cohort["cohort"]["ids"] = ["val-healthy"]
    save_json(cohort_path, cohort)
    with pytest.raises(ValueError, match="unique positive validation cracks"):
        execute_scientific_cell("variant-data", **inputs)


def test_new_regional_training_selects_paired_validation_and_freezes_encoder(tmp_path):
    torch.manual_seed(9)
    encoder = TinyEncoder(classes=3)
    with torch.no_grad():
        encoder.features.weight.zero_()
        encoder.features.weight[:3, :, 0, 0].copy_(torch.eye(3))
        encoder.features.bias.zero_()
        encoder.classifier[-1].weight.zero_()
        encoder.classifier[-1].weight[:, :3].copy_(torch.eye(3))
        encoder.classifier[-1].bias.zero_()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    samples = [
        {"image": torch.eye(3)[c, :, None, None].expand(3, 4, 4), "target": c} for c in range(3)
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    reduced = DataLoader(
        [{"image": torch.zeros(3, 4, 4), "target": c} for c in range(3)], batch_size=2
    )
    result = execute_scientific_cell(
        "regional-training",
        regional_run=None,
        REGIONAL_DIR=tmp_path,
        REGIONAL_CONFIG={
            "max_epochs": 1,
            "patience": 3,
            "warmup_epochs": 1,
            "lr_backbone": 0.01,
            "lr_head": 0.02,
            "weight_decay": 0.0,
            "amp": False,
        },
        class_weights=np.ones(3),
        prepared={"data_fingerprint": "synthetic-regions"},
        regional_encoder=encoder,
        regional_train_loader=loader,
        regional_val_loaders={"native": loader, "reduced": reduced},
    )
    run = result["regional_run"]
    assert run["status"] == "complete" and run["completed_epochs"] == run["best_epoch"] == 1
    assert run["completed_updates"] == result["scheduler"].last_epoch == 2
    assert run["skipped_updates"] == 0
    assert any(not torch.equal(before[k], value) for k, value in encoder.state_dict().items())
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    history = pd.read_csv(tmp_path / "history.csv")
    assert np.isfinite(history[["train_loss", "val_loss"]]).all().all()
    assert history.native_macro_f1.iloc[0] == 1
    assert history.reduced_macro_f1.iloc[0] == pytest.approx(1 / 6)
    assert run["best_macro_f1"] == pytest.approx(
        history[["native_macro_f1", "reduced_macro_f1"]].iloc[0].mean()
    )
    selected = torch.load(tmp_path / "best.pt", weights_only=True)
    assert all(torch.equal(selected[k], value) for k, value in encoder.state_dict().items())
    plt.close("all")


def test_new_fusion_training_updates_both_attention_modes_and_restores_selected_model(tmp_path):
    rng = np.random.default_rng(19)
    labels = np.zeros((4, 6), dtype=np.float32)
    labels[np.arange(3), [0, 1, 3]] = 1
    known = np.ones((4, 25, 3), dtype=bool)
    targets = np.zeros((4, 25, 3), dtype=np.float32)
    for category in range(3):
        known[category, 1:, category] = False
        targets[category, 0, category] = 1
    train = {
        "ids": np.asarray([f"train-{i}" for i in range(4)]),
        "labels": labels,
        "global_features": rng.standard_normal((4, 4)).astype(np.float32),
        "global_logits": np.zeros((4, 6), dtype=np.float32),
        "local_features": rng.standard_normal((4, 25, 4)).astype(np.float32),
        "tile_valid": np.ones((4, 25), dtype=bool),
        "region_targets": targets,
        "region_known": known,
    }
    validation = {
        **train,
        "ids": np.asarray([f"val-{i}" for i in range(4)]),
        "region_known": np.zeros_like(known),
        "region_targets": np.zeros_like(targets),
    }
    config = {
        "seed": 42,
        "device": "cpu",
        "batch_size": 2,
        "hidden": 4,
        "max_epochs": 1,
        "patience": 5,
        "lr": 0.01,
        "weight_decay": 0.0,
        "regional_weight": 0.25,
        "alignment_weight": 0.1,
    }
    result = execute_scientific_cell(
        "fusion-training",
        train_arrays=train,
        val_arrays=validation,
        train_records=list(range(4)),
        expected_ids=validation["ids"],
        expected_labels=labels,
        dimension=4,
        positions=tile_geometry()[1],
        CONFIG=config,
        SEEDS=(42,),
        OUTPUT_DIR=tmp_path,
        inputs={
            "protocol": "reference-global-regional-v1",
            "data_fingerprint": "synthetic-fusion",
            "class_names": list(CLASS_NAMES),
            "normalization": {},
            "geometry": {},
            "encoders": {"global_weights": "frozen-global", "regional_weights": "frozen-regional"},
        },
    )
    assert set(result["runs"]) == {(42, "balanced"), (42, "unbalanced")}
    for directory, run in result["runs"].values():
        assert run["status"] == "complete" and run["completed_epochs"] == 1
        assert run["updates"] == 2 and run["presentations"] == 4
        history = pd.read_csv(directory / "history.csv")
        assert history.epoch.tolist() == [0, 1]
        assert run["best_epoch"] == int(history.loc[history.val_mAP.idxmax(), "epoch"])
    assert result["candidate"].output_weight.abs().sum() > 0
    assert result["scheduler"].last_epoch == 2
    with torch.inference_mode():
        predicted = result["model"](
            torch.from_numpy(validation["global_features"]),
            torch.from_numpy(validation["global_logits"]),
            torch.from_numpy(validation["local_features"]),
            tile_geometry()[1],
            torch.from_numpy(validation["tile_valid"]),
        )["logits"].numpy()
    with np.load(tmp_path / "fusion/validation.npz", allow_pickle=False) as selected:
        np.testing.assert_allclose(predicted, selected["logits"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(selected["ids"], validation["ids"])
