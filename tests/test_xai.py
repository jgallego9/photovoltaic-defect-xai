"""Scientific explanation oracles for reference and multiview classifiers."""

import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import photovoltaic_xai.variant_xai as variant_xai
from photovoltaic_xai.models import create_model
from photovoltaic_xai.variant import (
    GlobalRegionalClassifier,
    RegionFusion,
    VariantDataset,
    file_hash,
)
from photovoltaic_xai.variant_xai import MultiViewAdapter, variant_cam
from photovoltaic_xai.xai import (
    cam,
    cam_layers,
    localization_metrics,
    make_rise_bank,
    pair_random_curves,
    perturbation_curve,
    randomized_model,
    rise,
    select_xai_subset,
)


class ToNHWC(nn.Module):
    def forward(self, value):
        return value.permute(0, 2, 3, 1)


class RegionModel(nn.Module):
    def __init__(self, nhwc=False):
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1, bias=False)
        self.spatial = ToNHWC() if nhwc else nn.Identity()
        self.head = nn.Linear(2, 6, bias=False)
        if nhwc:
            self.model_name = "swin_t"
        with torch.no_grad():
            self.features.weight.zero_()
            self.features.weight[0, 0] = self.features.weight[1, 1] = 1
            self.head.weight.zero_()
            self.head.weight[0, 0] = self.head.weight[1, 1] = 1
        self.seen = []

    def forward(self, image):
        self.seen.append(image.detach().clone())
        features = self.spatial(self.features(image))
        if getattr(self, "model_name", None) == "swin_t":
            features = features.permute(0, 3, 1, 2)
        return self.head(features.mean((-2, -1)))


def image():
    value = torch.zeros(3, 4, 6)
    value[0, :, :3], value[1, :, 3:] = 2, 3
    return value


@pytest.mark.parametrize("method", ["gradcam", "layercam"])
def test_cam_selects_the_class_region_in_nchw_and_swin_layout(method):
    for nhwc in (False, True):
        model = RegionModel(nhwc).train()
        before = copy.deepcopy(model.state_dict())
        for c in (0, 1):
            heatmap = cam(model, image(), c, method=method, layers=[model.spatial])
            assert torch.equal(heatmap, (image()[c] > 0).float())
            assert not heatmap.requires_grad
        assert model.training and not model.spatial._forward_hooks
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        with pytest.raises(ValueError, match="participate"):
            cam(model, image(), 0, layers=[model.spatial, nn.Identity()])
        assert not model.spatial._forward_hooks


def test_cam_targets_execute_in_the_three_native_architectures():
    for name in ("efficientnet_b0", "convnext_tiny", "swin_t"):
        model = create_model(name)
        for method in ("gradcam", "layercam"):
            heatmap = cam(model, torch.rand(3, 32, 32), 0, method=method)
            assert heatmap.shape == (32, 32) and torch.isfinite(heatmap).all()
            assert all(not layer._forward_hooks for layer in cam_layers(model, method=method))


def test_rise_matches_sigmoid_weighted_masks_and_preserves_padding_across_batches():
    model, inputs = RegionModel(), image()
    inputs[:, 0] = -2
    valid = torch.ones(4, 6, dtype=torch.bool)
    valid[0] = False
    bank = make_rise_bank((4, 6), n_masks=9, grid_size=2, seed=7)
    repeated = make_rise_bank((4, 6), n_masks=9, grid_size=2, seed=7)
    assert bank["grids"].shape == (9, 1, 2, 2) and bank["grids"].dtype == torch.bool
    assert all(torch.equal(bank[key], repeated[key]) for key in ("grids", "offsets_h", "offsets_w"))
    one = rise(model, inputs, bank, batch_size=1, valid_mask=valid)
    model.seen.clear()
    batched = rise(model, inputs, bank, batch_size=4, valid_mask=valid)
    assert one.shape == (6, 4, 6) and torch.allclose(one, batched, atol=1e-7)
    assert not batched[:, 0].any()
    assert all(
        torch.equal(batch[:, :, 0], inputs[None, :, 0].expand(len(batch), -1, -1))
        for batch in model.seen
    )
    large = F.interpolate(bank["grids"].float(), (6, 9), mode="bilinear", align_corners=False)
    masks = torch.stack(
        [
            large[i, :, top : top + 4, left : left + 6]
            for i, (top, left) in enumerate(zip(bank["offsets_h"], bank["offsets_w"]))
        ]
    )
    with torch.no_grad():
        probabilities = model(inputs[None] * torch.where(valid[None, None], masks, 1)).sigmoid()
    expected = torch.einsum("nc,nhw->chw", probabilities, masks[:, 0]) / (9 * 0.5)
    assert torch.allclose(batched, expected * valid, atol=1e-7)


def test_localization_uses_own_boxes_ties_valid_area_and_absent_reasons():
    saliency = torch.tensor([[100.0, 100.0, 100.0], [1.0, 1.0, 0.0]])
    valid = torch.tensor([[False, False, False], [True, True, True]])
    boxes = torch.tensor([[False, False, False], [True, False, False]])
    report = localization_metrics(saliency, boxes, valid_mask=valid)
    assert report["pointing_game"] == report["energy"] == pytest.approx(0.5)
    assert report["area_fraction"] == pytest.approx(1 / 3)
    assert report["energy_corrected"] == pytest.approx(0.25)
    for heatmap, reference, reason in (
        (saliency, torch.zeros_like(valid), "no_class_annotation"),
        (torch.zeros_like(saliency), boxes, "null_map"),
    ):
        missing = localization_metrics(heatmap, reference, valid_mask=valid)
        assert missing["reason"] == reason and missing["energy"] is None


def test_deletion_endpoints_padding_random_baseline_and_constant_maps():
    model, inputs = RegionModel(), image()
    inputs[:, 0] = -2
    valid = torch.ones(4, 6, dtype=torch.bool)
    valid[0] = False
    saliency = inputs[0].clamp_min(0)
    curve = perturbation_curve(model, inputs, 0, saliency, valid_mask=valid)
    with torch.no_grad():
        endpoint = float(model(inputs[None].masked_fill(valid[None, None], 0))[0, 0])
    assert len(curve["fractions"]) == 21 and curve["logits"][0] > curve["logits"][-1]
    assert curve["logits"][-1] == pytest.approx(endpoint)
    assert curve == perturbation_curve(
        model, inputs, 0, saliency, valid_mask=valid, baseline=torch.zeros_like(inputs)
    )
    assert all(
        torch.equal(batch[:, :, 0], inputs[None, :, 0].expand(len(batch), -1, -1))
        for batch in model.seen
    )
    random = perturbation_curve(model, inputs, 0, saliency, valid_mask=valid, random_order=True)
    assert random == perturbation_curve(
        model, inputs, 0, saliency, valid_mask=valid, random_order=True
    )
    for value, reason in ((0.0, "null_map"), (1.0, "constant_map")):
        flat = torch.full((4, 6), value)
        flat[0] = 10  # Padding does not determine the deletion order.
        model.seen.clear()
        missing = perturbation_curve(model, inputs, 0, flat, valid_mask=valid)
        assert missing["reason"] == reason and missing["auc_probability"] is None
        assert not model.seen and missing["probabilities"] == [None] * 21
        baseline = perturbation_curve(model, inputs, 0, flat, valid_mask=valid, random_order=True)
        assert baseline["reason"] is None and baseline["auc_probability"] is not None
    nearly_flat = torch.ones(4, 6)
    nearly_flat[1, 0] += 1e-6
    assert perturbation_curve(model, inputs, 0, nearly_flat, valid_mask=valid)["reason"] is None


def test_deletion_replacement_image_preserves_alignment_padding_and_endpoints():
    model, inputs = RegionModel(), image()
    inputs[:, 0] = -2
    valid = torch.ones(4, 6, dtype=torch.bool)
    valid[0] = False
    saliency = inputs[0].clamp_min(0)
    baseline = torch.arange(inputs.numel(), dtype=torch.float32).reshape_as(inputs) / 20
    baseline[:, 0] = 999
    before = baseline.clone()
    curve = perturbation_curve(model, inputs, 0, saliency, valid_mask=valid, baseline=baseline)
    views = torch.cat(model.seen)
    expected = torch.where(valid[None], baseline, inputs)
    assert torch.equal(views[0], inputs) and torch.equal(views[-1], expected)
    assert torch.equal(views[1, :, 1, 0], baseline[:, 1, 0])
    assert torch.equal(views[:, :, 0], inputs[None, :, 0].expand(len(views), -1, -1))
    assert torch.equal(baseline, before)
    with torch.inference_mode():
        assert curve["logits"][0] == pytest.approx(float(model(inputs[None])[0, 0]))
        assert curve["logits"][-1] == pytest.approx(float(model(expected[None])[0, 0]))
    for invalid in (torch.zeros(1, 4, 6), torch.full_like(inputs, torch.nan)):
        with pytest.raises(ValueError, match="baseline"):
            perturbation_curve(model, inputs, 0, saliency, valid_mask=valid, baseline=invalid)


def test_subset_uses_only_labels_and_preserves_cardinality_and_healthy_quotas():
    records = [
        {"id": f"mono-{c}-{i}", "labels": [int(j == c) for j in range(6)]}
        for c in range(6)
        for i in range(5)
    ]
    records += [
        {"id": f"{kind}-{i}", "labels": [value] * 6}
        for kind, value in (("multi", 1), ("healthy", 0))
        for i in range(20)
    ]
    records.append({"id": "unreliable", "labels": [1] * 6, "localization_valid": False})
    selected = select_xai_subset(records, per_class=4, healthy=3)
    for record in records:
        record["probabilities"] = [1.0] * 6
    assert selected == select_xai_subset(list(reversed(records)), per_class=4, healthy=3)
    assert "unreliable" not in selected and len(selected) == len(set(selected))
    assert sum(i.startswith("healthy") for i in selected) == 3
    assert sum(i.startswith("mono") for i in selected) == 12
    assert any(i.startswith("multi") for i in selected)


def test_reinitialization_changes_only_head_or_backbone():
    model = create_model("swin_t")
    key = "features.7.1.attn.relative_position_bias_table"
    with torch.no_grad():
        model.features[7][-1].attn.relative_position_bias_table.fill_(7)
        model.head.weight.fill_(7)
    original = copy.deepcopy(model.state_dict())
    for scope in ("head", "backbone"):
        randomized = randomized_model(model, scope=scope)
        assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
        assert torch.equal(randomized.state_dict()[key], original[key]) == (scope == "head")
        assert torch.equal(randomized.head.weight, model.head.weight) == (scope == "backbone")


def test_reinitialization_seeds_are_reproducible_and_leave_caller_rng_untouched():
    model = create_model("efficientnet_b0").train()
    original = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state()
    default = randomized_model(model, scope="head")
    weights = []
    for seed in (42, 43, 44):
        control = randomized_model(model, scope="head", seed=seed)
        weights.append(control.classifier[-1].weight.detach().clone())
        assert torch.equal(control.classifier[-1].weight, default.classifier[-1].weight) == (
            seed == 42
        )
        assert torch.equal(torch.random.get_rng_state(), rng)
    assert not torch.equal(weights[1], weights[2])
    assert model.training
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(3, 2, 1, bias=False),
            *[nn.Identity() for _ in range(6)],
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2, 6, bias=False))
        with torch.no_grad():
            self.features[1].weight.zero_()
            self.features[1].weight[0, 0] = self.features[1].weight[1, 1] = 1
            self.classifier[-1].weight.zero_()
            self.classifier[-1].weight[0, 0] = self.classifier[-1].weight[1, 1] = 1


class EvidenceFusion(nn.Module):
    """A known first-crop causal path plus the unchanged global classifier."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, global_features, global_logits, local_features, positions, tile_valid):
        result = global_logits.clone()
        result[:, :2] += local_features[:, 0] * tile_valid[:, :1] * self.weight[None]
        return {"logits": result}


def classifier(fusion=None):
    return GlobalRegionalClassifier(
        TinyEncoder(), TinyEncoder(), EvidenceFusion() if fusion is None else fusion
    )


def adapter_inputs():
    global_image = torch.zeros(3, 512, 512)
    tiles = torch.zeros(25, 3, 224, 224)
    tiles[0, 0] = 2
    return global_image, tiles, torch.ones(25, dtype=torch.bool), torch.ones(512, 512).bool()


@pytest.mark.parametrize("size", [512, 1024])
def test_identity_is_bit_exact_to_native_pil_views_and_classifier(tmp_path, size):
    # Fine native texture must survive in crops; it cannot be reconstructed by
    # merely enlarging the separately quantized global view of a 1024 image.
    pixels = np.random.default_rng(7).integers(0, 256, (size, size), dtype=np.uint8)
    path = tmp_path / "native.png"
    Image.fromarray(pixels).save(path)
    record = {
        "id": "native",
        "split": "val",
        "image_path": path.name,
        "sha256": file_hash(path),
        "width": size,
        "height": size,
    }
    sample = VariantDataset([record], tmp_path, {"mean": [0.4] * 3, "std": [0.27] * 3})[0]
    model = classifier().eval()
    adapter = MultiViewAdapter(
        model,
        sample["image"],
        sample["tiles"],
        sample["tile_valid"],
        torch.ones(512, 512).bool(),
    )
    global_views, tile_views, valid = adapter.views(adapter.mask_image[None])
    torch.testing.assert_close(global_views[0], sample["image"], rtol=0, atol=0)
    torch.testing.assert_close(tile_views[0], sample["tiles"], rtol=0, atol=0)
    with torch.no_grad():
        expected = model(sample["image"][None], sample["tiles"][None], valid)["logits"]
        torch.testing.assert_close(adapter(adapter.mask_image[None]), expected, rtol=0, atol=0)


def test_retention_changes_all_routes_in_corresponding_coordinates_and_preserves_padding():
    global_image, tiles, tile_valid, valid = adapter_inputs()
    global_image.fill_(2)
    tiles.fill_(3)
    valid[:64] = False
    global_image[:, :64] = -7
    adapter = MultiViewAdapter(classifier(), global_image, tiles, tile_valid, valid)
    retention = adapter.mask_image.clone()
    retention[:, :128, :128] = 0
    global_views, tile_views, _ = adapter.views(retention[None])
    assert global_views[0, :, 64:128, :128].eq(0).all()
    assert global_views[0, :, :64].eq(-7).all()
    # Crop0 covers the top-left 128x128 in global coordinates. Its top half is
    # padding and remains unchanged; retained distant crops remain exact too.
    assert tile_views[0, 0, :, :112].eq(3).all()
    assert tile_views[0, 0, :, 114:-2, 2:-2].eq(0).all()
    assert torch.equal(tile_views[0, 24], tiles[24])
    global_zero, regional_zero, _ = adapter.views(torch.zeros_like(retention)[None])
    assert global_zero[0, :, valid].eq(0).all()
    assert global_zero[0, :, ~valid].eq(-7).all()
    inside = adapter.tile_content_mask.expand_as(tiles)
    # Away from the antialiased content boundary, complete removal is exact.
    assert regional_zero[0][~inside].eq(3).all()
    assert regional_zero[0, 24].eq(0).all()


def test_replacement_pair_controls_both_endpoints_without_changing_originals():
    image, tiles, tile_valid, valid = adapter_inputs()
    valid[:32] = False
    adapter = MultiViewAdapter(classifier(), image, tiles, tile_valid, valid)
    baseline_global = torch.full_like(image, 0.3)
    baseline_tiles = torch.full_like(tiles, 0.7)
    changed = adapter.with_baseline(baseline_global, baseline_tiles)
    original_global, original_tiles, _ = changed.views(changed.mask_image[None])
    assert torch.equal(original_global[0], image) and torch.equal(original_tiles[0], tiles)
    final_global, final_tiles, _ = changed.views(torch.zeros_like(changed.mask_image)[None])
    assert final_global[0, :, valid].eq(0.3).all()
    assert final_global[0, :, ~valid].eq(0).all()
    assert final_tiles[0, 24].eq(0.7).all()
    assert adapter.baseline_global.eq(0).all() and adapter.baseline_tiles.eq(0).all()


@pytest.mark.parametrize("method", ["gradcam", "layercam"])
def test_final_logit_cam_recovers_causal_crop_and_leaves_frozen_weights_grads_modes(method):
    model = classifier().train()
    adapter = MultiViewAdapter(model, *adapter_inputs())
    # Existing optimizer gradients must not be cleared or overwritten by CAM.
    for parameter in model.fusion.parameters():
        parameter.grad = torch.full_like(parameter, 11)
    before = copy.deepcopy(model.state_dict())
    requirements = [parameter.requires_grad for parameter in model.parameters()]
    result = variant_cam(adapter, 0, method=method)
    assert result["fused"].shape == (512, 512)
    assert result["fused"][:128, :128].max() == pytest.approx(1)
    assert result["fused"][130:, 130:].eq(0).all()
    assert result["global"].eq(0).all() and result["regional"].max() > 0
    assert torch.equal(result["global"] + result["regional"], result["fused"])
    assert all(not value.requires_grad for value in result.values())
    assert model.training and model.fusion.training
    assert not model.global_encoder.training and not model.regional_encoder.training
    assert [parameter.requires_grad for parameter in model.parameters()] == requirements
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(parameter.grad.eq(11).all() for parameter in model.fusion.parameters())
    assert all(not module._forward_hooks for module in model.modules())
    # A geographically matched intervention removes the actual evidence; a
    # distant intervention does not. This validates more than saliency shape.
    removal = adapter.mask_image.clone()
    removal[:, :128, :128] = 0
    distant = adapter.mask_image.clone()
    distant[:, 384:, 384:] = 0
    with torch.no_grad():
        unchanged = adapter(adapter.mask_image[None])[0, 0]
        assert adapter(removal[None])[0, 0] < unchanged * 0.02
        assert adapter(distant[None])[0, 0] == unchanged


def test_padding_only_crops_have_no_influence_or_cam_even_with_nonzero_pixels():
    fusion = RegionFusion(dimension=2, hidden=2)
    with torch.no_grad():
        fusion.output_weight.fill_(1)
    image, tiles, _, valid = adapter_inputs()
    tile_valid = torch.zeros(25, dtype=torch.bool)
    model = classifier(fusion)
    adapter = MultiViewAdapter(model, image, tiles, tile_valid, valid)
    baseline = adapter(adapter.mask_image[None])
    changed = MultiViewAdapter(model, image, tiles + 100, tile_valid, valid)
    torch.testing.assert_close(baseline, changed(changed.mask_image[None]), rtol=0, atol=0)
    maps = variant_cam(adapter, 0)
    assert maps["regional"].eq(0).all()
    with torch.no_grad():
        output = model.fusion(
            torch.zeros(1, 2),
            torch.zeros(1, 6),
            torch.randn(1, 25, 2),
            model.positions,
            tile_valid[None],
        )
    assert output["attention"][0, :, -1].eq(1).all()
    assert output["attention"][0, :, :-1].eq(0).all()


def test_rise_and_deletion_use_the_final_two_branch_classifier():
    adapter = MultiViewAdapter(classifier(), *adapter_inputs())
    bank = make_rise_bank(n_masks=4, grid_size=4, seed=3)
    maps = rise(adapter, adapter.mask_image, bank, valid_mask=adapter.valid_mask, batch_size=1)
    batched = rise(adapter, adapter.mask_image, bank, valid_mask=adapter.valid_mask, batch_size=2)
    torch.testing.assert_close(maps, batched, rtol=0, atol=1e-7)
    saliency = variant_cam(adapter, 0)["fused"]
    curve = perturbation_curve(
        adapter, adapter.mask_image, 0, saliency, valid_mask=adapter.valid_mask
    )
    assert curve["logits"][0] == pytest.approx(2)
    assert curve["logits"][-1] == pytest.approx(0)
    assert curve["logits"][2] < curve["logits"][0] * 0.02


def test_invalid_mask_and_baseline_contracts_are_rejected():
    adapter = MultiViewAdapter(classifier(), *adapter_inputs())
    for value in (torch.full_like(adapter.mask_image, -1), adapter.mask_image * torch.nan):
        with pytest.raises(ValueError, match="Retention"):
            adapter(value[None])
    channels = adapter.mask_image.clone()
    channels[0] = 0
    with pytest.raises(ValueError, match="channels"):
        adapter(channels[None])
    with pytest.raises(ValueError, match="Baseline"):
        adapter.with_baseline(adapter.global_image, adapter.tiles[:1])


class ControlEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 1, stride=32), *[nn.Identity() for _ in range(7)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, 6))


def test_independent_controls_preserve_trained_weights_and_rng(monkeypatch):
    monkeypatch.setattr(variant_xai, "create_model", lambda _: ControlEncoder())
    model = GlobalRegionalClassifier(ControlEncoder(), ControlEncoder(), RegionFusion(4, 4)).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state()
    for scope in ("readouts", "encoders"):
        first = variant_xai.randomized_variant(model, scope=scope, seed=43)
        again = variant_xai.randomized_variant(model, scope=scope, seed=43)
        other = variant_xai.randomized_variant(model, scope=scope, seed=44)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(torch.equal(v, again.state_dict()[k]) for k, v in first.state_dict().items())
        assert any(not torch.equal(v, other.state_dict()[k]) for k, v in first.state_dict().items())
        if scope == "readouts":
            assert torch.equal(
                first.global_encoder.features[0].weight, model.global_encoder.features[0].weight
            )
        else:
            assert torch.equal(first.fusion.output_weight, model.fusion.output_weight)
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="scope"):
        variant_xai.randomized_variant(model, scope="unknown", seed=43)


def test_xai_reuse_requires_scientific_inputs_and_complete_image_results(tmp_path, monkeypatch):
    identity = {
        "selected": {
            "data_fingerprint": "fixed-validation-data",
            "class_names": ["crack"],
            "normalization": {"mean": [0.5], "std": [0.25]},
            "thresholds": [0.7],
            "geometry": {"global_size": 512},
            "weights": {"global.pt": "original-weights"},
        },
        "ids": ["image"],
        "bank_sha256": "shared-mask-bank",
        "config": {"n_masks": 4000},
        "control_selection": {"ids": ["image"], "config": {"control_seeds": [43, 44]}},
        "secondary_baseline": "fixed-blur",
    }
    recorded = copy.deepcopy(identity)
    recorded["sources"] = {"retired-file.py": "historical-layout"}
    assert variant_xai.same_xai_inputs(recorded, identity)
    for changed in ("data", "weights", "thresholds", "ids", "bank", "config", "baseline"):
        other = copy.deepcopy(identity)
        if changed == "data":
            other["selected"]["data_fingerprint"] = "another-split"
        elif changed == "weights":
            other["selected"]["weights"]["global.pt"] = "different-checkpoint"
        elif changed == "thresholds":
            other["selected"]["thresholds"] = [0.5]
        elif changed == "ids":
            other["ids"] = ["different-image"]
        elif changed == "bank":
            other["bank_sha256"] = "different-masks"
        elif changed == "baseline":
            other["secondary_baseline"] = "different-blur"
        else:
            other["config"]["n_masks"] = 40
        assert not variant_xai.same_xai_inputs(recorded, other)

    path = tmp_path / "image.json"
    path.write_text(json.dumps({"identity": recorded}))
    with pytest.raises(ValueError, match="inputs differ"):
        variant_xai.read_xai_result(path, identity)

    report = {
        "inputs_key": variant_xai.variant_xai_key(identity),
        "spatial": [{"method": "gradcam"}],
        "curves": [{}] * 28,
        "shared_random": True,
    }
    path.write_text(json.dumps(report))
    assert variant_xai.read_xai_result(path, identity, maps=True) is None
    different_checkpoint = copy.deepcopy(identity)
    different_checkpoint["selected"]["weights"]["global.pt"] = "different-checkpoint"
    with pytest.raises(ValueError, match="inputs differ"):
        variant_xai.read_xai_result(path, different_checkpoint, maps=True)
    maps_path = path.with_suffix(".npz")
    np.savez_compressed(maps_path, gradcam=np.ones((2, 2)))
    with pytest.raises(ValueError, match="maps do not match"):
        variant_xai.read_xai_result(path, identity, maps=True)
    report["maps_sha256"] = variant_xai.file_hash(maps_path)
    path.write_text(json.dumps(report))
    assert variant_xai.read_xai_result(path, identity, maps=True) == report
    np.savez_compressed(maps_path, gradcam=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="maps do not match"):
        variant_xai.read_xai_result(path, identity, maps=True)
    report["curves"].pop()
    path.write_text(json.dumps(report))
    assert variant_xai.read_xai_result(path, identity, maps=True) is None

    # Interruption after the last new per-image commit must not revive old CSVs.
    import gc
    import time
    from pathlib import Path

    import nbformat
    import pandas as pd

    old_identity = copy.deepcopy(identity)
    identity["selected"]["weights"]["global.pt"] = "new-checkpoint"
    report = {
        "inputs_key": variant_xai.variant_xai_key(identity),
        "maps_sha256": variant_xai.file_hash(maps_path),
        "shared_random": True,
        "spatial": [
            {"method": method, "pointing_game": 1.0} for method in ("gradcam", "layercam", "rise")
        ],
        "curves": [
            {"method": method, "fraction": fraction, "random_order": method == "random"}
            for method in ("gradcam", "layercam", "rise", "random")
            for fraction in np.linspace(0, 1, 21)
        ],
    }
    path.write_text(json.dumps(report))
    summary = {"identity": old_identity, "status": "complete", "images": 1, "control_images": 1}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"pointing_game": [-1.0]}).to_csv(tmp_path / "localization_val.csv", index=False)
    pd.DataFrame({"fraction": [-1.0]}).to_csv(tmp_path / "perturbations_val.csv", index=False)
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/06_variant_xai_comparison.ipynb",
        as_version=4,
    )
    source = next(cell.source for cell in notebook.cells if cell.id == "variant-xai-main")
    scope = {
        "XAI_DIR": tmp_path,
        "identity": identity,
        "records": [{"id": "image", "labels": [1]}],
        "control_ids": ["image"],
        "read_xai_result": variant_xai.read_xai_result,
        "same_xai_inputs": variant_xai.same_xai_inputs,
        "pair_random_curves": pair_random_curves,
        "json": json,
        "np": np,
        "pd": pd,
        "time": time,
        "gc": gc,
    }
    with pytest.raises(ValueError, match="inputs differ"):
        exec(compile(source, "variant-xai-main", "exec"), scope)
    assert pd.read_csv(tmp_path / "localization_val.csv").pointing_game.iloc[0] == -1
    summary.update(identity=identity, status="incomplete")
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    exec(compile(source, "variant-xai-main", "exec"), scope)
    assert scope["new_images"] == 0
    assert scope["location"].pointing_game.eq(1).all()
    assert len(scope["curves"]) == 126

    summary["status"] = "complete"
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    def forbid_write(*args, **kwargs):
        raise AssertionError("Completed compatible aggregates must not be rewritten")

    monkeypatch.setattr(pd.DataFrame, "to_csv", forbid_write)
    exec(compile(source, "variant-xai-main", "exec"), scope)
    assert scope["committed_aggregates"]
