"""Selected ConvNeXt-Tiny global-regional variant and its spatial objectives."""

import json
import math
import time
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from . import CLASS_NAMES, MODEL_NAMES
from .dataset import PairedELTransform, PVELADDataset, letterbox, read_grayscale
from .models import file_hash, load_run_metadata, save_json

LOCAL_CLASS_NAMES = ("crack", "finger", "thick_line")
LOCAL_CLASS_INDICES = (0, 1, 3)


def tile_geometry():
    """Fixed 5x5 grid: 256-pixel crops on a 1024-pixel canvas, stride 192."""
    starts = (0, 192, 384, 576, 768)
    bounds = torch.tensor([[x, y, x + 256, y + 256] for y in starts for x in starts])
    positions = torch.stack(
        (
            (bounds[:, 0] + bounds[:, 2]) / 2048,
            (bounds[:, 1] + bounds[:, 3]) / 2048,
            torch.full((25,), 0.25),
            torch.full((25,), 0.25),
        ),
        dim=1,
    )
    return bounds, positions


def encode_features(backbone, images):
    """The 768-dimensional embedding immediately before its linear classifier."""
    features = backbone.avgpool(backbone.features(images))
    for layer in list(backbone.classifier.children())[:-1]:
        features = layer(features)
    return features


class RegionFusion(nn.Module):
    """Combine global features with class-specific attention over regional crops."""

    def __init__(self, dimension=768, hidden=128):
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, hidden)
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU()
        )
        self.position_projection = nn.Linear(4, hidden, bias=False)
        self.keys = nn.Linear(hidden, hidden, bias=False)
        self.queries = nn.Parameter(torch.randn(6, hidden) / math.sqrt(hidden))
        self.null_key = nn.Parameter(torch.zeros(1, 1, hidden))
        self.local_head = nn.Linear(hidden, 3)
        # Six zero-initialized readouts reproduce every global logit at epoch zero.
        self.output_weight = nn.Parameter(torch.zeros(6, 2 * hidden))
        self.output_bias = nn.Parameter(torch.zeros(6))

    def forward(self, global_features, global_logits, local_features, positions, tile_valid):
        batch, count, _ = local_features.shape
        if (
            global_features.shape != (batch, self.global_projection[0].normalized_shape[0])
            or global_logits.shape != (batch, 6)
            or positions.shape != (count, 4)
            or tile_valid.shape != (batch, count)
            or tile_valid.dtype != torch.bool
            or not count
        ):
            raise ValueError("Incompatible global features, local views, coordinates or validity")
        global_context = self.global_projection(global_features)
        # Mask before projection as well as attention so invalid pixels have no influence.
        local_features = local_features.masked_fill(~tile_valid[..., None], 0)
        local = self.local_projection(local_features)
        keys = self.keys(local) + self.position_projection(positions)[None]
        keys = torch.cat((keys, self.null_key.expand(batch, -1, -1)), dim=1)
        query = global_context[:, None] + self.queries[None]
        scores = torch.einsum("bch,bkh->bck", query, keys) / math.sqrt(local.shape[-1])
        # The null slot stays valid even when every crop is masked.
        valid = torch.cat((tile_valid, tile_valid.new_ones(batch, 1)), dim=1)
        attention = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        values = torch.cat((local, local.new_zeros(batch, 1, local.shape[-1])), dim=1)
        pooled = torch.einsum("bck,bkh->bch", attention, values)
        context = torch.cat((pooled, global_context[:, None].expand(-1, 6, -1)), dim=-1)
        correction = (context * self.output_weight[None]).sum(-1) + self.output_bias
        output = {
            "logits": global_logits + correction,
            "local_logits": self.local_head(local),
            "attention": attention,
        }
        # Keep the original softmax path above, and expose stable log probabilities
        # for the supervised attention objective without changing saved predictions.
        output["log_attention"] = scores.masked_fill(~valid[:, None], -torch.inf).log_softmax(-1)
        return output


class GlobalRegionalClassifier(nn.Module):
    """Run the two frozen encoders and the selected fusion, without reference boxes."""

    def __init__(self, global_encoder, regional_encoder, fusion, tile_chunk_size=32):
        super().__init__()
        self.global_encoder = global_encoder.requires_grad_(False).eval()
        self.regional_encoder = regional_encoder.requires_grad_(False).eval()
        self.fusion = fusion
        self.tile_chunk_size = tile_chunk_size
        self.register_buffer("positions", tile_geometry()[1], persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.global_encoder.eval()
        self.regional_encoder.eval()
        return self

    def forward(self, images, tiles, tile_valid):
        if tiles.shape[:3] != (len(images), 25, 3):
            raise ValueError("Regional inputs require [B,25,3,224,224]")
        global_features = encode_features(self.global_encoder, images)
        global_logits = self.global_encoder.classifier[-1](global_features)
        local = torch.cat(
            [
                encode_features(self.regional_encoder, chunk)
                for chunk in tiles.flatten(0, 1).split(self.tile_chunk_size)
            ]
        ).reshape(len(images), 25, -1)
        result = self.fusion(global_features, global_logits, local, self.positions, tile_valid)
        result["global_logits"] = global_logits
        return result


def canvas_content_box(image):
    width, height = image.size
    rw = max(1, round(width * 1024 / max(width, height)))
    rh = max(1, round(height * 1024 / max(width, height)))
    return (1024 - rw) // 2, (1024 - rh) // 2, rw, rh


def variant_views(image, normalization, *, bounds=None):
    """Prepare the cell-wide image and fixed native-detail tiles for inference."""
    image = image.convert("L") if image.mode != "L" else image
    bounds = tile_geometry()[0] if bounds is None else bounds
    global_pixels = TF.pil_to_tensor(letterbox(image, 512)).float() / 255
    canvas = letterbox(image, 1024)
    tiles = torch.stack(
        [
            TF.pil_to_tensor(TF.resize(canvas.crop(box), [224, 224])).float() / 255
            for box in bounds.tolist()
        ]
    )
    left, top, rw, rh = canvas_content_box(image)
    lower = torch.maximum(bounds[:, :2], torch.tensor([left, top]))
    upper = torch.minimum(bounds[:, 2:], torch.tensor([left + rw, top + rh]))
    mean = torch.tensor(normalization["mean"])[:, None, None]
    std = torch.tensor(normalization["std"])[:, None, None]
    return {
        "image": (global_pixels.repeat(3, 1, 1) - mean) / std,
        "tiles": (tiles.repeat(1, 3, 1, 1) - mean) / std,
        "tile_valid": (upper > lower).all(1),
    }


class VariantDataset(Dataset):
    """Build both views from original pixels, with spatial targets only for training."""

    def __init__(self, records, project_root, normalization, allow_test=False):
        if not records or len({record["split"] for record in records}) != 1:
            raise ValueError("Views require one nonempty prepared split")
        if records[0]["split"] == "test" and not allow_test:
            raise ValueError("This variant notebook reserves test for the final evaluation")
        self.records = records
        self.project_root = Path(project_root)
        self.normalization = normalization
        self.bounds, _ = tile_geometry()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_grayscale(record, self.project_root)
        views = variant_views(image, self.normalization, bounds=self.bounds)
        valid = views["tile_valid"]
        width, height = image.size
        left, top, rw, rh = canvas_content_box(image)
        targets = torch.zeros(25, 3)
        known = torch.zeros(25, 3, dtype=torch.bool)
        if record["split"] == "train":
            boxes = [
                {
                    **box,
                    "xmin": left + box["xmin"] * rw / width,
                    "xmax": left + box["xmax"] * rw / width,
                    "ymin": top + box["ymin"] * rh / height,
                    "ymax": top + box["ymax"] * rh / height,
                }
                for box in record["boxes"]
            ]
            targets, known = regional_targets(
                boxes, record["labels"], record["localization_valid"], self.bounds
            )
        return {
            "ids": record["id"],
            **views,
            "region_targets": targets,
            "region_known": known & valid[:, None],
        }


def _feature_shapes(count, dimension, regional):
    if regional:
        return {
            "local_features": (count, 25, dimension),
            "tile_valid": (count, 25),
            "region_targets": (count, 25, 3),
            "region_known": (count, 25, 3),
        }
    return {"global_features": (count, dimension), "global_logits": (count, 6)}


def _feature_dtype(name):
    return np.dtype("bool" if name in {"tile_valid", "region_known"} else "float32")


def _read_feature_arrays(directory, metadata, shapes, records):
    shapes = {"ids": (len(records),), "labels": (len(records), 6)} | shapes
    arrays = {}
    for name, shape in shapes.items():
        path = directory / f"{name}.npy"
        if file_hash(path) != metadata["sha256"].get(path.name):
            raise ValueError(f"A feature cache file changed: {path.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or (name != "ids" and array.dtype != _feature_dtype(name)):
            raise ValueError(f"Invalid cached shape or dtype: {name}")
        arrays[name] = array
    if arrays["ids"].tolist() != [record["id"] for record in records]:
        raise ValueError("Cached identifiers disagree with the prepared split")
    if not np.array_equal(arrays["labels"], [record["labels"] for record in records]):
        raise ValueError("Cached labels disagree with the prepared split")
    return arrays


def load_feature_cache(directory, records, dimension, encoder_hashes, *, metadata=None):
    """Open regional features and validate their training-only supervision."""
    directory = Path(directory)
    metadata = (
        metadata if metadata is not None else json.loads((directory / "cache.json").read_text())
    )
    features = metadata["identity"]["features"]
    if metadata["status"] != "complete":
        raise ValueError("The feature cache is incomplete")
    if any(features["sources"].get(name) != digest for name, digest in encoder_hashes.items()):
        raise ValueError("Cached features belong to different encoder weights")
    arrays = _read_feature_arrays(
        directory, metadata, _feature_shapes(len(records), dimension, True), records
    )
    valid, known, targets = arrays["tile_valid"], arrays["region_known"], arrays["region_targets"]
    if np.any(known & ~valid[..., None]) or not np.isin(targets, (0, 1)).all():
        raise ValueError("Invalid spatial supervision in the regional features")
    if records[0]["split"] != "train" and known.any():
        raise ValueError("Evaluation data cannot supply spatial supervision")
    return arrays


def load_or_extract_features(directory, records, encoder, inputs, root, *, regional=False):
    """Read frozen ConvNeXt features, or build the cache in the original image order."""
    directory = Path(directory)
    metadata_path = directory / "cache.json"
    split = records[0]["split"]
    dimension = encoder.classifier[-1].in_features
    n = len(records)
    common_keys = ("data_fingerprint", "class_names", "normalization", "geometry")
    normalization = inputs["normalization"]
    ids = [record["id"] for record in records]
    labels = np.asarray([record["labels"] for record in records], dtype=np.float32)
    shapes = _feature_shapes(n, dimension, regional)
    if regional:
        weights = {"regional_weights": inputs["encoders"]["regional_weights"]}
        identity = {
            "split": split,
            "features": {key: inputs[key] for key in common_keys} | {"sources": weights},
        }
    else:
        identity = {"inputs": inputs, "split": split}

    if not metadata_path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="w+",
                dtype=_feature_dtype(name),
                shape=shape,
            )
            for name, shape in shapes.items()
        }
        if regional:
            dataset = VariantDataset(records, root, normalization)
        else:
            transform = PairedELTransform(
                512, mean=normalization["mean"], std=normalization["std"], noise_std=0
            )
            dataset = PVELADDataset(records, root, transform)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        device = next(encoder.parameters()).device
        id_key = "ids" if regional else "id"
        started, offset = time.perf_counter(), 0
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for batch in loader:
                end = offset + len(batch[id_key])
                if batch[id_key] != ids[offset:end]:
                    raise ValueError("Feature extraction order changed")
                if regional:
                    tiles = batch["tiles"].flatten(0, 1).to(device, dtype=torch.float32)
                    features = torch.cat(
                        [encode_features(encoder, chunk) for chunk in tiles.split(32)]
                    )
                    arrays["local_features"][offset:end] = (
                        features.reshape(end - offset, 25, dimension).cpu().numpy()
                    )
                    for name in ("tile_valid", "region_targets", "region_known"):
                        arrays[name][offset:end] = batch[name].numpy()
                else:
                    features = encode_features(
                        encoder, batch["image"].to(device, dtype=torch.float32)
                    )
                    logits = encoder.classifier[-1](features)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite global logits")
                    arrays["global_features"][offset:end] = features.cpu().numpy()
                    arrays["global_logits"][offset:end] = logits.cpu().numpy()
                if not torch.isfinite(features).all():
                    raise RuntimeError("Non-finite features")
                offset = end
        for array in arrays.values():
            array.flush()
        np.save(directory / "ids.npy", np.asarray(ids))
        np.save(directory / "labels.npy", labels)
        save_json(
            metadata_path,
            {
                "status": "complete",
                "identity": identity,
                "extraction_seconds": time.perf_counter() - started,
                "sha256": {path.name: file_hash(path) for path in directory.glob("*.npy")},
            },
        )

    metadata = json.loads(metadata_path.read_text())
    saved_inputs = metadata["identity"]["features" if regional else "inputs"]
    if metadata["status"] != "complete" or any(
        saved_inputs.get(key) != inputs[key] for key in common_keys
    ):
        raise ValueError("Feature cache belongs to different prepared data or is incomplete")
    if regional:
        return load_feature_cache(directory, records, dimension, weights, metadata=metadata)
    if saved_inputs["encoders"]["global_weights"] != inputs["encoders"]["global_weights"]:
        raise ValueError("Global feature cache uses different weights")
    arrays = _read_feature_arrays(directory, metadata, shapes, records)
    return {name: arrays[name] for name in shapes}


class CachedViews(Dataset):
    """Read one feature record from disk; no decoded images or full cache in RAM."""

    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays["ids"])

    def __getitem__(self, index):
        return {
            key: str(array[index])
            if key == "ids"
            else torch.from_numpy(np.array(array[index], copy=True))
            for key, array in self.arrays.items()
        }


def regional_targets(boxes, labels, localization_valid, tile_bounds):
    """Label covered boxes as positives and absent image classes as known negatives."""
    bounds = torch.as_tensor(tile_bounds, dtype=torch.float32)
    labels = torch.as_tensor(labels, device=bounds.device)
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not torch.isfinite(bounds).all():
        raise ValueError("Tile bounds require finite shape [K,4]")
    if not ((bounds[:, 2:] - bounds[:, :2]) > 0).all():
        raise ValueError("Tile rectangles must be nonempty")
    if labels.shape != (6,) or not torch.isin(labels, labels.new_tensor([0, 1])).all():
        raise ValueError("Regional supervision requires six binary image labels")
    targets = bounds.new_zeros((len(bounds), 3))
    known = torch.zeros_like(targets, dtype=torch.bool)
    # Untrusted annotations leave all tiles unsupervised.
    if not localization_valid:
        return targets, known
    for local_index, (name, image_index) in enumerate(zip(LOCAL_CLASS_NAMES, LOCAL_CLASS_INDICES)):
        own = [box for box in boxes if box["class_name"] == name]
        if not labels[image_index]:
            if own:
                raise ValueError("A regional box contradicts its absent image label")
            known[:, local_index] = True
            continue
        if not own:
            continue
        references = bounds.new_tensor(
            [[box[key] for key in ("xmin", "ymin", "xmax", "ymax")] for box in own]
        )
        sides = references[:, 2:] - references[:, :2]
        if not torch.isfinite(references).all() or not sides.gt(0).all():
            raise ValueError("Reference rectangles must be finite and nonempty")
        left_top = torch.maximum(bounds[:, None, :2], references[None, :, :2])
        right_bottom = torch.minimum(bounds[:, None, 2:], references[None, :, 2:])
        overlap = (right_bottom - left_top).clamp_min(0).prod(2)
        covered = (overlap / sides.prod(1)).ge(0.5).any(1)
        targets[covered, local_index] = 1
        known[covered, local_index] = True
    return targets, known


def masked_region_loss(logits, targets, known):
    """Average known tile BCE within each image/class, then across eligible pairs."""
    if (
        logits.ndim != 3
        or logits.shape[-1] != 3
        or targets.shape != logits.shape
        or known.shape != logits.shape
    ):
        raise ValueError("Regional logits, targets and masks require shape [B,K,3]")
    if known.dtype != torch.bool:
        raise ValueError("Regional supervision masks must be boolean")
    known = known.to(device=logits.device)
    targets = targets.to(device=logits.device, dtype=torch.float32)
    if (
        not torch.isfinite(targets).all()
        or not torch.isin(targets, targets.new_tensor([0, 1])).all()
    ):
        raise ValueError("Regional targets must be finite and binary")
    # Mask inputs before BCE so unknown entries cannot introduce spurious gradients.
    loss = F.binary_cross_entropy_with_logits(
        logits.float().masked_fill(~known, 0),
        targets.masked_fill(~known, 0),
        reduction="none",
    )
    counts = known.sum(1)
    per_pair = (loss * known).sum(1) / counts.clamp_min(1)
    eligible = counts > 0
    return per_pair.sum() / eligible.sum().clamp_min(1)


def pair_balance_weights(positive_counts, null_counts, num_images):
    """Give the six positive and null training strata equal weight in the risk."""
    positive = torch.as_tensor(positive_counts, dtype=torch.float64)
    null = torch.as_tensor(null_counts, dtype=torch.float64)
    if (
        type(num_images) is not int
        or num_images < 1
        or positive.shape != (3,)
        or null.shape != (3,)
    ):
        raise ValueError("Balance requires three training supports and a positive image count")
    counts = torch.stack((positive, null), dim=-1)
    if (
        not torch.isfinite(counts).all()
        or not counts.eq(counts.round()).all()
        or counts.le(0).any()
        or counts.sum(-1).gt(num_images).any()
    ):
        raise ValueError("All six training strata must be nonempty and fit the training corpus")
    return (num_images / (6 * counts)).float()


def attention_targets(targets, known, valid):
    known = known & valid[..., None]
    positive_tiles = targets.bool() & known
    positive = positive_tiles.any(1)
    null = (known.sum(1) == valid.sum(1)[:, None]) & ~positive & valid.any(1)[:, None]
    accepted = torch.cat((positive_tiles.permute(0, 2, 1), null[..., None]), dim=-1)
    return positive, null, accepted


def balanced_attention_loss(log_attention, targets, known, tile_valid, pair_weights=None):
    """Use the same positive bags and known-null targets as the unbalanced loss."""
    positive_pairs, null_pairs, accepted = attention_targets(targets, known, tile_valid)
    eligible = positive_pairs | null_pairs
    scores = log_attention[:, LOCAL_CLASS_INDICES]
    if not eligible.any():
        return scores[..., -1].sum() * 0
    log_mass = torch.logsumexp(
        scores[eligible].masked_fill(~accepted[eligible], -torch.inf), dim=-1
    )
    if pair_weights is None:
        return -log_mass.mean()
    weights = torch.as_tensor(pair_weights, device=log_attention.device, dtype=torch.float32)
    selected_weights = torch.where(positive_pairs, weights[:, 0], weights[:, 1])[eligible]
    # Divide by all images so the minibatch loss estimates the full-corpus stratified risk.
    return -(log_mass * selected_weights).sum() / len(targets)


def _variant_paths(manifest_path, manifest, root):
    paths = manifest["paths"]
    names = [paths["global_weights"], paths["regional_weights"], paths["fusion_dir"] + "/best.pt"]
    if root is None:
        root = next(
            (parent for parent in manifest_path.parents if (parent / names[0]).is_file()),
            manifest_path.parent,
        )
    return Path(root), names


def load_variant_manifest(manifest_path, prepared=None, *, root=None):
    """Check the model and the data that were used to select it."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["class_names"] != list(CLASS_NAMES):
        raise ValueError("The selected variant requires the current six-label order")
    if prepared is not None and any(
        manifest[key] != prepared[key] for key in ("data_fingerprint", "normalization")
    ):
        raise ValueError("The selected variant uses different prepared data")
    root, names = _variant_paths(manifest_path, manifest, root)
    paths = manifest["paths"]
    if "thresholds" not in manifest:
        runs = {
            "thresholds": paths["fusion_dir"] + "/run.json",
            "global_thresholds": paths["global_run"],
        }
        names.extend(runs.values())
    else:
        runs = {}
    for name in names:
        if file_hash(root / name) != manifest["sha256"][name]:
            raise ValueError(f"Selected artifact changed: {name}")
    for key, path in runs.items():
        run = json.loads((root / path).read_text())
        manifest[key] = run["metrics"]["thresholds"]
    for key in ("thresholds", "global_thresholds"):
        if key not in manifest:
            continue
        thresholds = np.asarray(manifest[key])
        if thresholds.shape != (len(CLASS_NAMES),) or not np.all(
            (0 <= thresholds) & (thresholds <= 1)
        ):
            raise ValueError(f"Invalid validation {key}")
    return manifest


def load_variant(manifest_path, device="cpu", *, root=None, manifest=None):
    """Rebuild the selected model from its three preserved parameter files."""
    from .models import create_model

    manifest_path = Path(manifest_path).resolve()
    if manifest is None:
        manifest = load_variant_manifest(manifest_path, root=root)
    root, names = _variant_paths(manifest_path, manifest, root)
    global_encoder = create_model("convnext_tiny")
    regional_encoder = create_model("convnext_tiny")
    regional_encoder.classifier[-1] = nn.Linear(regional_encoder.classifier[-1].in_features, 3)
    global_encoder.load_state_dict(
        torch.load(root / names[0], map_location="cpu", weights_only=True)
    )
    regional_encoder.load_state_dict(
        torch.load(root / names[1], map_location="cpu", weights_only=True)
    )
    state = torch.load(root / names[2], map_location="cpu", weights_only=True)
    fusion = RegionFusion(len(state["global_projection.0.weight"]), state["queries"].shape[1])
    fusion.load_state_dict(state)
    return GlobalRegionalClassifier(global_encoder, regional_encoder, fusion).to(device).eval()


def export_models(project_root, prepared):
    """Copy the selected model weights and their inference settings as one bundle."""
    root = Path(project_root)
    outputs = root / "outputs"
    variant = load_variant_manifest(
        outputs / "variant_reference_global/manifest.json", prepared, root=root
    )
    fingerprint = prepared["data_fingerprint"]

    paths = variant["paths"]
    weights = {
        f"{name}/best.pt": outputs / "experiments" / name / "best.pt" for name in MODEL_NAMES
    }
    weights["regional/best.pt"] = root / paths["regional_weights"]
    weights["fusion/best.pt"] = root / paths["fusion_dir"] / "best.pt"
    hashes = {name: file_hash(path) for name, path in weights.items()}
    for exported, original in (
        ("convnext_tiny/best.pt", paths["global_weights"]),
        ("regional/best.pt", paths["regional_weights"]),
        ("fusion/best.pt", paths["fusion_dir"] + "/best.pt"),
    ):
        if hashes[exported] != variant["sha256"][original]:
            raise ValueError(f"Selected weights changed: {original}")

    references = {}
    for name in MODEL_NAMES:
        run_dir = outputs / "experiments" / name
        run = load_run_metadata(run_dir, fingerprint)
        if run["config"]["model_name"] != name:
            raise ValueError(f"Unexpected architecture in {run_dir}")
        references[name] = {
            "weights": f"{name}/best.pt",
            "thresholds": run["metrics"]["thresholds"],
        }
    manifest = {
        "data_fingerprint": fingerprint,
        "class_names": list(CLASS_NAMES),
        "normalization": variant["normalization"],
        "geometry": variant["geometry"],
        "references": references,
        "thresholds": variant["thresholds"],
        "paths": {
            "global_weights": "convnext_tiny/best.pt",
            "regional_weights": "regional/best.pt",
            "fusion_dir": "fusion",
        },
        "sha256": hashes,
    }
    destination = outputs / "models"
    if destination.exists():
        if json.loads((destination / "manifest.json").read_text()) != manifest or any(
            file_hash(destination / name) != digest for name, digest in hashes.items()
        ):
            raise FileExistsError(f"Different model files already exist: {destination}")
        return destination

    with TemporaryDirectory(dir=outputs) as temporary:
        bundle = Path(temporary) / "models"
        for name, path in weights.items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(path, target)
        save_json(bundle / "manifest.json", manifest)
        bundle.rename(destination)
    return destination


def build_regions(records):
    """Select annotated anchors; overlapping classes are excluded from encoder fitting."""
    regions = []
    for record in records:
        if record["split"] not in {"train", "val"}:
            raise PermissionError("Regional encoder fitting excludes the test split")
        if not record["localization_valid"]:
            continue
        for index, box in enumerate(record["boxes"]):
            if box["class_name"] not in LOCAL_CLASS_NAMES:
                continue
            xmin, ymin, xmax, ymax = (box[key] for key in ("xmin", "ymin", "xmax", "ymax"))
            side = max(
                math.ceil(1.5 * max(xmax - xmin, ymax - ymin)),
                math.ceil(0.125 * max(record["width"], record["height"])),
            )
            width, height = min(side, record["width"]), min(side, record["height"])
            left = min(max(math.floor((xmin + xmax - width) / 2), 0), record["width"] - width)
            top = min(max(math.floor((ymin + ymax - height) / 2), 0), record["height"] - height)
            overlaps = any(
                other["class_name"] != box["class_name"]
                and max(xmin, other["xmin"]) < min(xmax, other["xmax"])
                and max(ymin, other["ymin"]) < min(ymax, other["ymax"])
                for other in record["boxes"]
            )
            if not overlaps:
                regions.append(
                    {
                        "id": f"{record['id']}:box{index}",
                        "image_id": record["id"],
                        "split": record["split"],
                        "class_index": LOCAL_CLASS_NAMES.index(box["class_name"]),
                        "class_name": box["class_name"],
                        "anchor_bounds": (xmin, ymin, xmax, ymax),
                        "crop_bounds": (left, top, left + width, top + height),
                    }
                )
    return regions


class RegionDataset(torch.utils.data.Dataset):
    """Use native/reduced paired validation and randomly mixed training crops."""

    def __init__(self, regions, records_by_id, project_root, transform, view):
        if not regions or view not in {"mixed", "native", "reduced"}:
            raise ValueError("Regions require a nonempty mixed, native or reduced view")
        if any(records_by_id[region["image_id"]]["split"] == "test" for region in regions):
            raise PermissionError("Regional encoder training does not use test")
        splits = {records_by_id[region["image_id"]]["split"] for region in regions}
        if len(splits) != 1 or not splits <= {"train", "val"}:
            raise ValueError("Regions require a single train or validation split")
        if (view == "mixed" and (splits != {"train"} or not transform.training)) or (
            splits == {"val"} and transform.training
        ):
            raise ValueError("Mixed views are training-only; validation must be clean")
        self.regions, self.records_by_id = regions, records_by_id
        self.root, self.transform, self.view = Path(project_root), transform, view

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, index):
        region = self.regions[index]
        record = self.records_by_id[region["image_id"]]
        image = read_grayscale(record, self.root)
        reduced = self.view == "reduced" or (self.view == "mixed" and bool(torch.rand(()) < 0.5))
        if reduced and max(image.size) > 512:
            scale = 512 / max(image.size)
            image = TF.resize(
                image,
                [max(1, round(image.height * scale)), max(1, round(image.width * scale))],
                antialias=True,
            )
            image = TF.resize(image, [record["height"], record["width"]], antialias=True)
        pixels, _ = self.transform(image.crop(tuple(region["crop_bounds"])))
        return {"image": pixels, "target": region["class_index"], "id": region["id"]}
