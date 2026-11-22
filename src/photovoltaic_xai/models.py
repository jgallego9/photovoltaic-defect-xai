"""Build reference models and manage their prediction results."""

import gc
import json
import math
import random
import time
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import convnext_tiny, efficientnet_b0, swin_t

from . import CLASS_NAMES, MODEL_NAMES, PROTOCOL_ID


def create_model(name, *, weights=None):
    """Build a six-output reference, optionally initialized from ImageNet."""
    builders = {
        "efficientnet_b0": efficientnet_b0,
        "convnext_tiny": convnext_tiny,
        "swin_t": swin_t,
    }
    if name not in builders:
        raise ValueError(f"Unknown reference architecture: {name}")
    model = builders[name](weights=weights)

    if name == "swin_t":
        model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    model.model_name = name
    return model


def parameter_groups(model, lr_backbone, lr_head):
    """Two disjoint AdamW groups, with the prescribed separate learning rates."""
    head = model.classifier if hasattr(model, "classifier") else model.head
    head_ids = {id(p) for p in head.parameters()}

    return [
        {
            "params": [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids],
            "lr": lr_backbone,
        },
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": lr_head},
    ]


def warmup_cosine(optimizer, warmup, total_updates):
    def rate(update):
        if update < warmup:
            return (update + 1) / warmup
        progress = min(1, (update - warmup) / max(1, total_updates - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)


def recalibrate_batchnorm(model, loader, device):
    layers = [
        layer for layer in model.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)
    ]
    if not layers:
        return 0
    modes = {layer: layer.training for layer in model.modules()}
    momenta = [layer.momentum for layer in layers]
    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.train()
    seen = 0
    try:
        with (
            torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            for batch in loader:
                images = batch["image"].to(device, dtype=torch.float32)
                for layer in layers:
                    layer.momentum = len(images) / (seen + len(images))
                if not torch.isfinite(model(images)).all():
                    raise RuntimeError("Non-finite BatchNorm calibration logits")
                seen += len(images)
    finally:
        for layer, momentum in zip(layers, momenta, strict=True):
            layer.momentum = momentum
        for layer, mode in modes.items():
            layer.training = mode
    return seen


def time_cuda(infer, device):
    """Measure one FP32 operation with the thesis CUDA timing protocol."""
    durations = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for _ in range(10):
            infer()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(50):
            started = time.perf_counter()
            infer()
            torch.cuda.synchronize(device)
            durations.append(1000 * (time.perf_counter() - started))
    return {
        "latency_median_ms": float(np.median(durations)),
        "latency_p90_ms": float(np.percentile(durations, 90)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def benchmark_model(model):
    """Benchmark a reference model with the CUDA FP32 protocol."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA access and the project's CUDA extra are required")
    device = torch.device("cuda", torch.cuda.current_device())
    original_device = next(model.parameters()).device
    modes = {module: module.training for module in model.modules()}
    try:
        model.to(device).eval()
        gc.collect()
        torch.cuda.empty_cache()
        image = torch.zeros(1, 3, 512, 512, device=device)
        timing = time_cuda(lambda: model(image), device)
        if hasattr(model, "classifier"):
            head = model.classifier[-1]
        else:
            head = model.head
        return {
            "device": torch.cuda.get_device_name(device),
            "image_size": 512,
            "batch_size": 1,
            "precision": "float32",
            "warmup": 10,
            "repetitions": 50,
            "parameters": sum(p.numel() for p in model.parameters()),
            "adapted_linear_head_parameters": sum(p.numel() for p in head.parameters()),
            **timing,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    finally:
        model.to(original_device)
        for module, mode in modes.items():
            module.training = mode


def seed_everything(seed: int) -> None:
    """Set the seed before constructing the model and data loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, loader, *, device="auto") -> dict:
    """Return FP32 predictions in loader order."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model.to(device).eval()
    logits_parts, label_parts, identifiers = [], [], []

    # Disable autocast to avoid overflowing FP16 logits during evaluation.
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            labels = batch["labels"].float()
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != 6:
                raise ValueError("Model must return a [batch, 6] logits tensor")
            if labels.shape != logits.shape or len(batch["id"]) != logits.shape[0]:
                raise ValueError("Batch labels or identifiers do not match model output")
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite prediction logits")
            identifiers.extend(batch["id"])
            logits_parts.append(logits.float().cpu())
            label_parts.append(labels.cpu())

    if not logits_parts:
        raise ValueError("Cannot predict from an empty loader")

    logits = torch.cat(logits_parts)
    return {
        "ids": identifiers,
        "labels": torch.cat(label_parts).numpy(),
        "logits": logits.numpy(),
        "probabilities": logits.sigmoid().numpy(),
    }


def save_json(path, value):
    """Write a JSON result atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_run(run_dir: str | Path, run: dict) -> None:
    save_json(Path(run_dir) / "run.json", run)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_hash(run_dir):
    return file_hash(Path(run_dir) / "best.pt")


def load_run_metadata(run_dir, data_fingerprint=None) -> dict:
    """Read completed results for the current protocol and prepared data."""
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        raise ValueError("No completed run metadata: the run is incomplete")
    run = json.loads(path.read_text())

    if run.get("protocol_id") != PROTOCOL_ID or run.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Run protocol/class order is incompatible with the current experiment")

    config = run.get("config", {})
    if config.get("model_name") not in MODEL_NAMES:
        raise ValueError("The saved model must be one of the three references")

    if run.get("evaluation_precision") != "float32":
        raise ValueError("Run requires the current FP32 evaluation protocol")

    epochs, best = run.get("completed_epochs"), run.get("best_epoch")
    if (
        run.get("status") != "complete"
        or type(epochs) is not int
        or type(best) is not int
        or not 1 <= best <= epochs
    ):
        raise ValueError("The run is incomplete or its best epoch is inconsistent")

    if (
        data_fingerprint is not None
        and run.get("metadata", {}).get("data_fingerprint") != data_fingerprint
    ):
        raise ValueError("Run belongs to different prepared data")
    return run


def load_trained_model(run_dir: str | Path, prepared: dict, device="cuda"):
    """Rebuild the recorded architecture and load its best weights."""
    run = load_run_metadata(run_dir, prepared["data_fingerprint"])
    model = create_model(run["config"]["model_name"]).to(device)
    state = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval(), run


def load_run(run_dir: str | Path, prepared: dict):
    """Load and verify validation predictions."""
    run = load_run_metadata(run_dir, prepared["data_fingerprint"])
    thresholds = np.asarray(run["metrics"]["thresholds"], dtype=float)
    if (
        thresholds.shape != (6,)
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError("Validation thresholds must be six finite values in [0, 1]")

    part = [record for record in prepared["records"] if record["split"] == "val"]
    with np.load(Path(run_dir) / "validation.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in ("ids", "labels", "logits")}
        if archive["checkpoint_sha256"].item() != checkpoint_hash(run_dir):
            raise ValueError("Saved predictions in validation.npz belong to different weights")

    if predictions["ids"].tolist() != [record["id"] for record in part]:
        raise ValueError("Stale predictions: IDs or their order differ from the current split")
    for key in ("labels", "logits"):
        values = predictions[key]
        if values.shape != (len(part), 6) or not np.isfinite(values).all():
            raise ValueError(f"Saved {key} require finite shape [N, 6]")
    if not np.array_equal(predictions["labels"], np.asarray([record["labels"] for record in part])):
        raise ValueError("Stale predictions: labels differ from the prepared data")

    predictions["probabilities"] = torch.from_numpy(predictions["logits"]).sigmoid().numpy()
    return run, predictions
