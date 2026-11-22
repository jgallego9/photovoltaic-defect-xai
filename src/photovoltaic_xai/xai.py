"""Generate and evaluate class-specific saliency maps."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import CLASS_NAMES
from .models import create_model


@contextmanager
def _evaluation(model: nn.Module, *, gradients: bool = False):
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.inference_mode(not gradients), torch.set_grad_enabled(gradients):
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def _image(image: Tensor) -> Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Explain one preprocessed image [3,H,W]")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("The preprocessed image must contain finite floating-point values")
    return image.unsqueeze(0)


def _valid_mask(shape, mask: Tensor, device) -> Tensor:
    mask = torch.as_tensor(mask, device=device).bool()
    if mask.shape != shape or not mask.any():
        raise ValueError("valid_mask must be a nonempty [H,W] content mask")
    return mask


def cam_layers(model: nn.Module, *, method: str = "gradcam") -> list[nn.Module]:
    """TorchVision 0.24 targets: stride 32, or strides 8/16/32 for LayerCAM."""
    name = getattr(model, "model_name", None)
    if method not in {"gradcam", "layercam"}:
        raise ValueError("method must be 'gradcam' or 'layercam'")
    if name == "efficientnet_b0":
        layers = [model.features[index] for index in (3, 5, 8)]
    elif name == "convnext_tiny":
        layers = [model.features[index] for index in (3, 5, 7)]
    elif name == "swin_t":
        layers = [model.features[index][-1].norm1 for index in (3, 5, 7)]
    else:
        raise ValueError("Provide explicit layers for a model outside the three references")
    return layers[-1:] if method == "gradcam" else layers


def cam(
    model: nn.Module,
    image: Tensor,
    class_index: int,
    *,
    method: str = "gradcam",
    layers: Sequence[nn.Module] | None = None,
) -> Tensor:
    """Compute a normalized Grad-CAM or Layer-CAM map for a class logit."""
    if method not in {"gradcam", "layercam"}:
        raise ValueError("method must be 'gradcam' or 'layercam'")
    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError("class_index must identify one of the six experimental labels")
    image = _image(image)
    targets = list(layers) if layers is not None else cam_layers(model, method=method)
    if not targets or len({id(layer) for layer in targets}) != len(targets):
        raise ValueError("Provide at least one distinct target layer")
    activations: dict[nn.Module, Tensor] = {}
    handles = []

    def capture(module, _inputs, output):
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise ValueError("CAM target layers must output a spatial four-dimensional tensor")
        activations[module] = output

    try:
        for layer in targets:
            handles.append(layer.register_forward_hook(capture))
        with _evaluation(model, gradients=True):
            inputs = image.detach().clone().requires_grad_(True)
            score = model(inputs)[:, class_index].sum()
            if any(layer not in activations for layer in targets):
                raise ValueError("A target layer did not participate in the forward pass")
            values = [activations[layer] for layer in targets]
            gradients = torch.autograd.grad(score, values)
            maps = []
            for value, gradient in zip(values, gradients, strict=True):
                if getattr(model, "model_name", None) == "swin_t":
                    value = value.permute(0, 3, 1, 2)
                    gradient = gradient.permute(0, 3, 1, 2)
                weights = (
                    gradient.mean((-2, -1), keepdim=True)
                    if method == "gradcam"
                    else (gradient.clamp_min(0))
                )
                heatmap = (weights * value).sum(1, keepdim=True).clamp_min(0)
                heatmap = F.interpolate(
                    heatmap, image.shape[-2:], mode="bilinear", align_corners=False
                )[0, 0]
                maps.append(heatmap / heatmap.max().clamp_min(torch.finfo(heatmap.dtype).tiny))
            result = torch.stack(maps).mean(0).detach()
            if not torch.isfinite(result).all():
                raise ValueError("CAM produced non-finite saliency")
            return result
    finally:
        for handle in handles:
            handle.remove()


def make_rise_bank(
    image_size: tuple[int, int] = (512, 512),
    n_masks: int = 4000,
    grid_size: int = 16,
    p: float = 0.5,
    seed: int = 42,
) -> dict:
    """Build the small CPU bank shared by every model and image in the comparison."""
    if min(*image_size, n_masks, grid_size) < 1 or not 0 < p < 1:
        raise ValueError("Positive image/mask/grid sizes and 0 < p < 1 are required")
    generator = torch.Generator().manual_seed(seed)
    height, width = image_size
    cell_h, cell_w = math.ceil(height / grid_size), math.ceil(width / grid_size)
    return {
        "grids": torch.rand(n_masks, 1, grid_size, grid_size, generator=generator) < p,
        "offsets_h": torch.randint(cell_h, (n_masks,), generator=generator),
        "offsets_w": torch.randint(cell_w, (n_masks,), generator=generator),
        "config": {
            "n_masks": n_masks,
            "grid_size": grid_size,
            "p": p,
            "seed": seed,
            "image_size": [height, width],
        },
    }


def rise(
    model: nn.Module,
    image: Tensor,
    bank: dict,
    *,
    valid_mask: Tensor,
    batch_size: int = 4,
) -> Tensor:
    """Estimate class saliency with sigmoid-weighted masks from the shared bank."""
    image = _image(image)
    height, width = image.shape[-2:]
    valid = _valid_mask((height, width), valid_mask, image.device)
    if bank["config"]["image_size"] != [height, width] or batch_size < 1:
        raise ValueError("RISE bank must match the image size and batch_size must be positive")
    grids, offsets_h, offsets_w = bank["grids"], bank["offsets_h"], bank["offsets_w"]
    n_masks, _, grid_size, _ = grids.shape
    cell_h, cell_w = math.ceil(height / grid_size), math.ceil(width / grid_size)
    accumulated = torch.zeros(
        (len(CLASS_NAMES), height, width), dtype=torch.float64, device=image.device
    )
    with _evaluation(model):
        for start in range(0, n_masks, batch_size):
            stop = min(start + batch_size, n_masks)
            large = F.interpolate(
                grids[start:stop].to(image.device, dtype=torch.float32),
                ((grid_size + 1) * cell_h, (grid_size + 1) * cell_w),
                mode="bilinear",
                align_corners=False,
            )
            masks = torch.stack(
                [
                    large[
                        offset - start,
                        :,
                        int(offsets_h[offset]) : int(offsets_h[offset]) + height,
                        int(offsets_w[offset]) : int(offsets_w[offset]) + width,
                    ]
                    for offset in range(start, stop)
                ]
            )
            retained = torch.where(valid[None, None], masks, 1)
            scores = model(image * retained).sigmoid().double()
            accumulated += torch.einsum("nc,nhw->chw", scores, masks[:, 0].double())
    maps = (accumulated / (n_masks * bank["config"]["p"])).float() * valid
    if not torch.isfinite(maps).all():
        raise ValueError("RISE produced non-finite saliency")
    return maps


def localization_metrics(
    saliency: Tensor,
    class_mask: Tensor,
    *,
    valid_mask: Tensor,
) -> dict[str, Any]:
    """Measure saliency inside the annotated boxes and correct energy for their area."""
    saliency = torch.as_tensor(saliency).detach().float()
    boxes = torch.as_tensor(class_mask, device=saliency.device).bool()
    if saliency.ndim != 2 or boxes.shape != saliency.shape:
        raise ValueError("saliency and class_mask must have the same [H,W] shape")
    if not torch.isfinite(saliency).all() or (saliency < 0).any():
        raise ValueError("Localization requires finite nonnegative saliency")
    valid = _valid_mask(saliency.shape, valid_mask, saliency.device)
    boxes = boxes & valid
    area = float(boxes.sum() / valid.sum())
    report = {
        "pointing_game": None,
        "energy": None,
        "area_fraction": area,
        "energy_corrected": None,
        "reason": None,
    }
    if not boxes.any():
        report["reason"] = "no_class_annotation"
        return report
    values = saliency * valid
    if not values.sum() > 0:
        report["reason"] = "null_map"
        return report
    maxima = valid & (values == values[valid].max())
    report["pointing_game"] = float((maxima & boxes).sum() / maxima.sum())
    report["energy"] = float(values[boxes].sum() / values.sum())
    if area < 1:
        report["energy_corrected"] = (report["energy"] - area) / (1 - area)
    else:
        report["reason"] = "full_content_annotation"
    return report


def perturbation_curve(
    model: nn.Module,
    image: Tensor,
    class_index: int,
    saliency: Tensor,
    *,
    valid_mask: Tensor,
    random_order: bool = False,
    baseline: Tensor | None = None,
) -> dict[str, Any]:
    """Measure a class score while replacing valid pixels in 5% steps."""
    image = _image(image)
    valid = _valid_mask(image.shape[-2:], valid_mask, image.device)
    saliency = torch.as_tensor(saliency, device=image.device)
    if saliency.shape != valid.shape or not torch.isfinite(saliency).all() or (saliency < 0).any():
        raise ValueError("saliency must be a finite nonnegative [H,W] map")
    if baseline is not None:
        baseline = torch.as_tensor(baseline, device=image.device, dtype=image.dtype)
        if baseline.shape != image.shape[1:] or not torch.isfinite(baseline).all():
            raise ValueError("baseline must be a finite [3,H,W] image matching the input")
    # Zero in normalized space represents the training mean.
    replacement = 0.0 if baseline is None else baseline[None]

    fractions = np.linspace(0, 1, 21)
    content = saliency[valid]
    if not random_order and content.max() == content.min():
        reason = "null_map" if content.max() == 0 else "constant_map"
        return {
            "fractions": fractions.tolist(),
            "logits": [None] * len(fractions),
            "probabilities": [None] * len(fractions),
            "random_order": False,
            "auc_probability": None,
            "reason": reason,
        }
    pixels = torch.where(valid.flatten())[0]
    if random_order:
        order = torch.randperm(len(pixels), generator=torch.Generator().manual_seed(42))
        order = order.to(image.device)
    else:
        order = torch.argsort(saliency.flatten()[pixels], descending=True, stable=True)
    pixels = pixels[order]
    scores = []
    with _evaluation(model):
        for start in range(0, len(fractions), 4):
            views = []
            for fraction in fractions[start : start + 4]:
                changed = torch.zeros_like(valid).flatten()
                changed[pixels[: round(float(fraction) * len(pixels))]] = True
                changed = changed.reshape(valid.shape)
                views.append(torch.where(changed[None, None], replacement, image)[0])
            scores.append(model(torch.stack(views))[:, class_index].float().cpu())
    logits = torch.cat(scores).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Perturbation produced non-finite logits")
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return {
        "fractions": fractions.tolist(),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "random_order": random_order,
        "auc_probability": float(np.trapezoid(probabilities, fractions)),
        "reason": None,
    }


def select_xai_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    per_class: int = 20,
    healthy: int = 20,
    seed: int = 42,
) -> list[str]:
    """Sample annotated defects and healthy images with seeded quotas."""
    if per_class < 1 or healthy < 0:
        raise ValueError("per_class must be positive and healthy nonnegative")
    records = sorted(
        (
            record
            for record in records
            if not any(record["labels"]) or record.get("localization_valid", True)
        ),
        key=lambda record: str(record["id"]),
    )
    ids = [str(record["id"]) for record in records]
    labels = np.asarray([record["labels"] for record in records])
    if (
        len(ids) != len(set(ids))
        or labels.shape != (len(ids), len(CLASS_NAMES))
        or (not np.isin(labels, (0, 1)).all())
    ):
        raise ValueError(
            "Records require unique IDs and binary labels in canonical six-class order"
        )
    generator = np.random.default_rng(seed)
    cardinality = labels.sum(1)
    chosen: set[int] = set()
    for index in range(len(CLASS_NAMES)):
        mono = generator.permutation(np.where((labels[:, index] == 1) & (cardinality == 1))[0])
        multi = generator.permutation(np.where((labels[:, index] == 1) & (cardinality > 1))[0])
        take_mono = min(len(mono), (per_class + 1) // 2)
        take_multi = min(len(multi), per_class // 2)
        selected = list(mono[:take_mono]) + list(multi[:take_multi])
        remaining = list(mono[take_mono:]) + list(multi[take_multi:])
        selected += remaining[: max(0, per_class - len(selected))]
        chosen.update(selected)
    chosen.update(generator.permutation(np.where(cardinality == 0)[0])[:healthy])
    return [ids[index] for index in sorted(chosen)]


def randomized_model(
    model: nn.Module,
    *,
    scope: str = "backbone",
    seed: int = 42,
) -> nn.Module:
    """Reinitialize one part of a model copy with a fixed seed."""
    if scope not in {"head", "backbone"}:
        raise ValueError("scope must be 'head' or 'backbone'")
    randomized = copy.deepcopy(model)
    # Fresh reference models are built on CPU; GPU random streams are untouched.
    with torch.random.fork_rng(devices=[]):
        # Equal seeds reproduce the same initialization, not independent controls.
        torch.random.default_generator.manual_seed(seed)
        fresh = create_model(model.model_name)
    prefix = "head." if model.model_name == "swin_t" else "classifier."
    state = randomized.state_dict()
    for key, value in fresh.state_dict().items():
        if key.startswith(prefix) == (scope == "head"):
            state[key] = value
    randomized.load_state_dict(state)
    return randomized


def curve_rows(identity, method, curve, *, random=False, baseline="mean"):
    return [
        {
            **identity,
            "method": method,
            "baseline": baseline,
            "random_order": random,
            "fraction": f,
            "logit": z,
            "masked_probability": p,
            "auc_probability": curve["auc_probability"],
            "curve_reason": curve["reason"],
        }
        for f, z, p in zip(curve["fractions"], curve["logits"], curve["probabilities"], strict=True)
    ]


def xai_key(inputs):
    """Identify the scientific inputs shared by an execution's image results."""
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def evaluate_xai_image(
    run, model, image, sample, valid, thresholds, bank, cam_map, *, batch_size, save_maps=False
):
    """Measure explanations and deletion for the same image and label pairs."""
    with torch.inference_mode():
        probabilities = model(image[None]).sigmoid()[0].cpu().numpy()
    healthy = not bool(sample["labels"].any())
    targets = torch.nonzero(sample["labels"]).flatten().tolist() or range(len(CLASS_NAMES))
    rise_maps = rise(model, image, bank, valid_mask=valid, batch_size=batch_size)

    spatial, curves, saved_maps = [], [], {}
    for index in targets:
        item = {
            "run": run,
            "id": sample["id"],
            "class": CLASS_NAMES[index],
            "healthy": healthy,
            "probability": float(probabilities[index]),
            "predicted_positive": bool(probabilities[index] >= thresholds[index]),
        }
        maps = {}
        for method in ("gradcam", "layercam"):
            mapped = cam_map(index, method)
            if isinstance(mapped, Mapping):
                maps[method] = mapped["fused"]
                if save_maps:
                    for branch in ("global", "regional"):
                        saved_maps[f"{method}_{index}_{branch}"] = mapped[branch].cpu().numpy()
            else:
                maps[method] = mapped
        maps["rise"] = rise_maps[index]

        random = perturbation_curve(
            model, image, index, maps["rise"], valid_mask=valid, random_order=True
        )
        curves.extend(curve_rows(item, "random", random, random=True))
        for method, saliency in maps.items():
            if save_maps:
                saved_maps[f"{method}_{index}"] = saliency.cpu().numpy()
            spatial.append(
                {
                    **item,
                    "method": method,
                    **localization_metrics(saliency, sample["masks"][index], valid_mask=valid),
                }
            )
            curve = perturbation_curve(model, image, index, saliency, valid_mask=valid)
            curves.extend(curve_rows(item, method, curve))
    return spatial, curves, saved_maps


def pair_random_curves(rows):
    """Pair each saliency method with the shared random deletion curve."""
    paired = []
    for row in rows:
        if row.get("method") == "random" and row.get("random_order"):
            paired.extend(dict(row, method=method) for method in ("gradcam", "layercam", "rise"))
        else:
            paired.append(row)
    return paired
