"""Explain the global-regional classifier with CAM and shared retention masks."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import ImageFilter
from torch import Tensor, nn
from torch.nn import functional as F

from . import CLASS_NAMES
from .dataset import PairedELTransform, PVELADDataset, read_grayscale
from .models import create_model
from .variant import canvas_content_box, encode_features, file_hash, tile_geometry, variant_views
from .xai import _evaluation, xai_key


class MultiViewAdapter(nn.Module):
    """Apply each retention mask to the global image and its native-detail crops."""

    def __init__(
        self,
        model: nn.Module,
        global_image: Tensor,
        tiles: Tensor,
        tile_valid: Tensor,
        global_valid_mask: Tensor,
        *,
        baseline_global: Tensor | None = None,
        baseline_tiles: Tensor | None = None,
        canvas_valid_mask: Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        if global_image.shape != (3, 512, 512) or tiles.shape != (25, 3, 224, 224):
            raise ValueError("Views require global [3,512,512] and tiles [25,3,224,224]")
        if not global_image.is_floating_point() or not tiles.is_floating_point():
            raise ValueError("Views must contain floating-point normalized image values")
        if global_image.device != tiles.device or global_image.dtype != tiles.dtype:
            raise ValueError("Global and regional views must share device and dtype")
        if not torch.isfinite(global_image).all() or not torch.isfinite(tiles).all():
            raise ValueError("Views must be finite")
        device = global_image.device
        valid = torch.as_tensor(global_valid_mask, device=device, dtype=torch.bool)
        tile_valid = torch.as_tensor(tile_valid, device=device, dtype=torch.bool)
        if valid.shape != (512, 512) or not valid.any() or tile_valid.shape != (25,):
            raise ValueError("Require nonempty valid_mask [512,512] and tile_valid [25]")
        if canvas_valid_mask is None:
            canvas_valid = F.interpolate(valid[None, None].float(), (1024, 1024), mode="nearest")
            canvas_valid = canvas_valid[0, 0].bool()
        else:
            canvas_valid = torch.as_tensor(canvas_valid_mask, device=device, dtype=torch.bool)
            if canvas_valid.shape != (1024, 1024) or not canvas_valid.any():
                raise ValueError("canvas_valid_mask must be nonempty [1024,1024]")
        if (baseline_global is None) != (baseline_tiles is None):
            raise ValueError("Both baseline views are required together")
        baseline_global = (
            torch.zeros_like(global_image) if baseline_global is None else baseline_global
        )
        baseline_tiles = torch.zeros_like(tiles) if baseline_tiles is None else baseline_tiles
        for replacement, original in ((baseline_global, global_image), (baseline_tiles, tiles)):
            if (
                replacement.shape != original.shape
                or replacement.device != device
                or replacement.dtype != original.dtype
                or not torch.isfinite(replacement).all()
            ):
                raise ValueError("Baseline views must match original shape, device and dtype")
        for name, value in (
            ("global_image", global_image),
            ("tiles", tiles),
            ("tile_valid", tile_valid),
            ("valid_mask", valid),
            ("canvas_valid_mask", canvas_valid),
            ("baseline_global", baseline_global),
            ("baseline_tiles", baseline_tiles),
        ):
            self.register_buffer(name, value.detach().clone(), persistent=False)
        self.register_buffer("mask_image", torch.ones_like(global_image), persistent=False)
        bounds, _ = tile_geometry()
        self._bounds = bounds.tolist()
        tile_masks = torch.stack([canvas_valid[y1:y2, x1:x2] for x1, y1, x2, y2 in self._bounds])
        tile_masks = F.interpolate(tile_masks[:, None].float(), (224, 224), mode="nearest")
        self.register_buffer("tile_content_mask", tile_masks.bool(), persistent=False)
        # Correct antialiasing so retaining every pixel reproduces the input exactly.
        unity = F.interpolate(
            global_image.new_ones(1, 1, 256, 256),
            (224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        self.register_buffer("crop_mask_unity", unity[None], persistent=False)

    def with_baseline(self, global_image: Tensor, tiles: Tensor) -> MultiViewAdapter:
        """Return an adapter with the same views and a new replacement pair."""
        return type(self)(
            self.model,
            self.global_image,
            self.tiles,
            self.tile_valid,
            self.valid_mask,
            baseline_global=global_image,
            baseline_tiles=tiles,
            canvas_valid_mask=self.canvas_valid_mask,
        )

    def views(self, masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return perturbed global views, native-detail crops and crop validity."""
        if (
            masks.ndim != 4
            or masks.shape[1:] != (3, 512, 512)
            or not masks.is_floating_point()
            or masks.device != self.global_image.device
            or not torch.isfinite(masks).all()
            or (masks < 0).any()
            or (masks > 1).any()
        ):
            raise ValueError("Retention masks require finite [B,3,512,512] values in [0,1]")
        if not torch.equal(masks[:, :1].expand_as(masks), masks):
            raise ValueError("Retention mask channels must be identical")
        masks = masks[:, :1].to(dtype=self.global_image.dtype)
        global_mask = torch.where(self.valid_mask[None, None], masks, 1)
        canvas_mask = F.interpolate(masks, (1024, 1024), mode="bilinear", align_corners=False)
        canvas_mask = torch.where(self.canvas_valid_mask[None, None], canvas_mask, 1)
        crop_masks = torch.stack(
            [canvas_mask[:, :, y1:y2, x1:x2] for x1, y1, x2, y2 in self._bounds], dim=1
        )
        crop_masks = F.interpolate(
            crop_masks.flatten(0, 1),
            (224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(len(masks), 25, 1, 224, 224)
        crop_masks = (crop_masks / self.crop_mask_unity).clamp(0, 1)
        crop_masks = torch.where(self.tile_content_mask[None], crop_masks, 1)
        # This interpolation form reproduces the original bit for bit at mask=1.
        global_views = self.global_image[None] * global_mask
        global_views = global_views + self.baseline_global[None] * (1 - global_mask)
        tile_views = self.tiles[None] * crop_masks
        tile_views = tile_views + self.baseline_tiles[None] * (1 - crop_masks)
        return global_views, tile_views, self.tile_valid[None].expand(len(masks), -1)

    def forward(self, masks: Tensor) -> Tensor:
        return self.model(*self.views(masks))["logits"]


def _targets(encoder: nn.Module, method: str, layers: Sequence[nn.Module] | None):
    targets = list(layers) if layers is not None else [encoder.features[i] for i in (3, 5, 7)]
    if layers is None and method == "gradcam":
        targets = targets[-1:]
    if not targets or len({id(layer) for layer in targets}) != len(targets):
        raise ValueError("Provide at least one distinct spatial layer per branch")
    return targets


def variant_cam(
    adapter: MultiViewAdapter,
    class_index: int,
    *,
    method: str = "gradcam",
    global_layers: Sequence[nn.Module] | None = None,
    regional_layers: Sequence[nn.Module] | None = None,
) -> dict[str, Tensor]:
    """Return global, regional and fused CAMs in global image coordinates."""
    if method not in {"gradcam", "layercam"}:
        raise ValueError("method must be 'gradcam' or 'layercam'")
    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError("class_index must identify one of the six experimental labels")
    model = adapter.model
    global_targets = _targets(model.global_encoder, method, global_layers)
    regional_targets = _targets(model.regional_encoder, method, regional_layers)
    if len(global_targets) != len(regional_targets):
        raise ValueError("Global and regional scales must be paired")
    targets = global_targets + regional_targets
    if len({id(layer) for layer in targets}) != len(targets):
        raise ValueError("The two branches must use distinct target layer instances")
    activations = {}
    handles = []

    def capture(module, _inputs, output):
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise ValueError("CAM layers must output a spatial NCHW tensor")
        activations[module] = output

    try:
        for layer in targets:
            handles.append(layer.register_forward_hook(capture))
        with _evaluation(model, gradients=True):
            image = adapter.global_image[None].detach().clone().requires_grad_(True)
            tiles = adapter.tiles.detach().clone().requires_grad_(True)
            global_features = encode_features(model.global_encoder, image)
            global_logits = model.global_encoder.classifier[-1](global_features)
            # One 25-crop execution keeps each activation's crop index unambiguous.
            local_features = encode_features(model.regional_encoder, tiles)[None]
            output = model.fusion(
                global_features,
                global_logits,
                local_features,
                model.positions,
                adapter.tile_valid[None],
            )
            if any(target not in activations for target in targets):
                raise ValueError("A target layer did not participate in the forward pass")
            values = [activations[target] for target in targets]
            # Gradients through the fused logit already include attention.
            gradients = torch.autograd.grad(output["logits"][0, class_index], values)
            raw_maps = []
            for value, gradient in zip(values, gradients, strict=True):
                weights = (
                    gradient.mean((-2, -1), keepdim=True)
                    if method == "gradcam"
                    else gradient.clamp_min(0)
                )
                raw_maps.append((value * weights).sum(1, keepdim=True).clamp_min(0))
            global_maps, regional_maps = [], []
            for index in range(len(global_targets)):
                global_map = (
                    F.interpolate(
                        raw_maps[index], (512, 512), mode="bilinear", align_corners=False
                    )[0, 0]
                    * adapter.valid_mask
                )
                crops = raw_maps[index + len(global_targets)]
                if crops.shape[0] != 25:
                    raise ValueError("Regional CAM must retain exactly 25 crop activations")
                crops = F.interpolate(crops, (224, 224), mode="bilinear", align_corners=False)
                crops = F.interpolate(crops, (256, 256), mode="bilinear", align_corners=False)
                canvas = global_map.new_zeros(1024, 1024)
                for crop_index, (x1, y1, x2, y2) in enumerate(adapter._bounds):
                    if adapter.tile_valid[crop_index]:
                        canvas[y1:y2, x1:x2] += crops[crop_index, 0]
                canvas = canvas * adapter.canvas_valid_mask
                regional_map = (
                    F.interpolate(
                        canvas[None, None], (512, 512), mode="bilinear", align_corners=False
                    )[0, 0]
                    * adapter.valid_mask
                )
                # Normalize both branches together to preserve their relative magnitude.
                maximum = (
                    (global_map + regional_map).max().clamp_min(torch.finfo(global_map.dtype).tiny)
                )
                global_maps.append(global_map / maximum)
                regional_maps.append(regional_map / maximum)
            global_map = torch.stack(global_maps).mean(0).detach()
            regional_map = torch.stack(regional_maps).mean(0).detach()
            result = {"global": global_map, "regional": regional_map}
            result["fused"] = global_map + regional_map
            if any(not torch.isfinite(value).all() for value in result.values()):
                raise ValueError("CAM produced non-finite saliency")
            return result
    finally:
        for handle in handles:
            handle.remove()


def image_adapter(root, prepared, model, record, device):
    """Read original multiview pixels and independent evaluation annotations."""
    image = read_grayscale(record, root)
    normalization = prepared["normalization"]
    sample = variant_views(image, normalization)
    transform = PairedELTransform(512, mean=normalization["mean"], std=normalization["std"])
    dataset = PVELADDataset([record], root, transform=transform, return_masks=True)
    reference = dataset.sample(0, image)
    left, top, rw, rh = canvas_content_box(image)
    canvas_valid = torch.zeros(1024, 1024, dtype=torch.bool, device=device)
    canvas_valid[top : top + rh, left : left + rw] = True
    adapter = MultiViewAdapter(
        model,
        sample["image"].to(device),
        sample["tiles"].to(device),
        sample["tile_valid"].to(device),
        reference["valid_mask"].to(device),
        canvas_valid_mask=canvas_valid,
    )
    return adapter, reference


def blur_views(root, prepared, record, device):
    """Blur the native image at a relative scale before preparing both views."""
    image = read_grayscale(record, root)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=12 * max(image.size) / 512))
    views = variant_views(blurred, prepared["normalization"])
    return views["image"].to(device), views["tiles"].to(device)


def randomized_variant(model, *, scope, seed):
    """Copy the variant and reinitialize its readouts or feature extractors."""
    if scope not in {"readouts", "encoders"}:
        raise ValueError("scope must be readouts or encoders")
    control = copy.deepcopy(model).cpu()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if scope == "readouts":
            head = control.global_encoder.classifier[-1]
            head.load_state_dict(nn.Linear(head.in_features, head.out_features).state_dict())
            readout = nn.Linear(control.fusion.output_weight.shape[1], 6)
            with torch.no_grad():
                control.fusion.output_weight.copy_(readout.weight)
                control.fusion.output_bias.copy_(readout.bias)
        else:
            for encoder in (control.global_encoder, control.regional_encoder):
                fresh = create_model("convnext_tiny")
                encoder.features.load_state_dict(fresh.features.state_dict())
    return control.eval()


def _scientific_inputs(identity):
    selected = identity["selected"]
    return (
        selected["data_fingerprint"],
        selected["class_names"],
        selected["normalization"],
        selected["thresholds"],
        selected["geometry"],
        {name: digest for name, digest in selected["weights"].items() if name.endswith(".pt")},
        identity["ids"],
        identity["bank_sha256"],
        identity["config"],
        identity["control_selection"]["ids"],
        identity["control_selection"]["config"],
        identity["secondary_baseline"],
    )


def variant_xai_key(identity):
    return xai_key(_scientific_inputs(identity))


def same_xai_inputs(recorded, expected):
    """Compare the inputs that determine the XAI results."""
    try:
        return _scientific_inputs(recorded) == _scientific_inputs(expected)
    except (KeyError, TypeError):
        return False


def read_xai_result(path, identity, *, maps=False, verify_map=True):
    """Reuse one complete image's results only for the same scientific inputs."""
    path = Path(path)
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get("inputs_key") != variant_xai_key(identity):
        raise ValueError("Saved XAI inputs differ; existing results are preserved")
    if maps:
        maps_path = path.with_suffix(".npz")
        spatial = result.get("spatial")
        if not maps_path.is_file() or not spatial:
            return None
        # Each target has three guided curves and one shared random curve.
        if not result.get("shared_random") or len(result.get("curves", [])) != 28 * len(spatial):
            return None
        expected_hash = result.get("maps_sha256")
        if not expected_hash or (verify_map and file_hash(maps_path) != expected_hash):
            raise ValueError(f"Saved XAI maps do not match their report: {maps_path}")
    return result
