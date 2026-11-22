"""PVEL-AD image loading, masks, preprocessing and normalization."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from . import CLASS_NAMES

_CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def read_grayscale(record: dict, project_root: str | Path) -> Image.Image:
    """Read the verified original image referenced by a prepared record."""
    path = Path(project_root) / record["image_path"]
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"The image changed after preparation: {path}")
    with Image.open(BytesIO(content)) as opened:
        image = opened.convert("L")
    if image.size != (record["width"], record["height"]):
        raise ValueError(f"The record and image have different dimensions: {path}")
    return image


def letterbox(
    image: Image.Image | torch.Tensor,
    image_size: int,
    *,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Image.Image | torch.Tensor:
    """Preserve the aspect ratio and pad to a square with zero intensity."""
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    width, height = image.size if isinstance(image, Image.Image) else image.shape[-2:][::-1]
    scale = image_size / max(width, height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    horizontal, vertical = image_size - resized_width, image_size - resized_height
    left, top = horizontal // 2, vertical // 2
    resized = TF.resize(
        image,
        [resized_height, resized_width],
        interpolation=interpolation,
        antialias=interpolation is not InterpolationMode.NEAREST,
    )
    return TF.pad(resized, [left, top, horizontal - left, vertical - top], fill=0)


@dataclass(slots=True)
class PairedELTransform:
    """Preprocess images and apply the same geometry to their masks."""

    image_size: int = 512
    mean: Sequence[float] = (0.5, 0.5, 0.5)
    std: Sequence[float] = (0.5, 0.5, 0.5)
    training: bool = False
    noise_std: float = 0.015

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.mean, self.std = tuple(self.mean), tuple(self.std)
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("mean/std require three values and positive standard deviations")

    def __call__(
        self,
        image: Image.Image,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = image.convert("L")
        if masks is not None and (
            masks.ndim != 3 or masks.shape[-2:] != (image.height, image.width)
        ):
            raise ValueError("Masks (C, height, width) must match the image dimensions")

        image = letterbox(image, self.image_size)
        if masks is not None:
            masks = letterbox(masks, self.image_size, interpolation=InterpolationMode.NEAREST)

        if self.training:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                masks = None if masks is None else TF.hflip(masks)
            if torch.rand(()) < 0.5:
                image = TF.vflip(image)
                masks = None if masks is None else TF.vflip(masks)
            if torch.rand(()) < 0.5:
                image = TF.adjust_brightness(image, float(torch.empty(()).uniform_(0.9, 1.1)))
                image = TF.adjust_contrast(image, float(torch.empty(()).uniform_(0.9, 1.1)))
                image = TF.adjust_gamma(image, float(torch.empty(()).uniform_(0.9, 1.1)))

        image = TF.pil_to_tensor(image).to(torch.float32).div_(255.0)
        if self.training:
            # Replicate luminance noise across channels to avoid introducing color information.
            image = (image + torch.randn_like(image) * self.noise_std).clamp_(0.0, 1.0)

        image = image.repeat(3, 1, 1)
        mean, std = (
            image.new_tensor(self.mean)[:, None, None],
            image.new_tensor(self.std)[:, None, None],
        )
        image = (image - mean) / std
        return image, masks


def _build_masks(record: dict, width: int, height: int) -> torch.Tensor:
    masks = torch.zeros((6, height, width), dtype=torch.float32)
    if not record["localization_valid"] or record["source"] == "good":
        return masks
    classes = set()
    for box in record["boxes"]:
        xmin, ymin, xmax, ymax = (box[key] for key in ("xmin", "ymin", "xmax", "ymax"))
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"Empty or out-of-bounds box in {record['id']}: {box}")
        name = box["class_name"]
        classes.add(name)
        masks[_CLASS_TO_INDEX[name], ymin:ymax, xmin:xmax] = 1

    if classes != {name for name, positive in zip(CLASS_NAMES, record["labels"]) if positive}:
        raise ValueError(f"Boxes and labels do not match: {record['id']}")
    return masks


class PVELADDataset(Dataset[dict[str, Any]]):
    """Load a prepared PVEL-AD split with verified images and optional masks."""

    def __init__(
        self,
        records: list[dict],
        project_root: str | Path,
        transform: PairedELTransform,
        return_masks: bool = False,
        allow_test: bool = False,
    ) -> None:
        if not records:
            raise ValueError("The dataset requires prepared records")
        self.records = records
        self.project_root = Path(project_root)
        self.return_masks = return_masks
        first = records[0]
        self.split = first["split"]
        if self.split not in {"train", "val", "test"} or any(
            record["split"] != self.split for record in records
        ):
            raise ValueError("The dataset requires a single explicit train/val/test split")
        if self.split == "test" and not allow_test:
            raise PermissionError("The test split requires allow_test=True for final evaluation")
        self.labels = torch.tensor([record["labels"] for record in records], dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.sample(index, read_grayscale(self.records[index], self.project_root))

    def sample(self, index: int, image: Image.Image) -> dict[str, Any]:
        record = self.records[index]
        masks = None
        if self.return_masks:
            masks = _build_masks(record, image.width, image.height)
            # The auxiliary channel shares the geometry without adding a seventh label.
            valid = torch.ones((1, image.height, image.width))
            masks = torch.cat((masks, valid))

        image, masks = self.transform(image, masks)
        sample = {"image": image, "labels": self.labels[index], "id": record["id"]}
        if self.return_masks:
            sample.update(masks=masks[:6], valid_mask=masks[6].bool())
        return sample


def compute_normalization(
    records: list[dict],
    project_root: str | Path,
    *,
    image_size: int = 512,
) -> dict[str, Any]:
    """Mean and standard deviation of original training images, including letterbox padding."""
    if any(record["split"] not in {"train", "val", "test"} for record in records):
        raise ValueError("Normalization requires an explicit train/val/test split")
    train = [record for record in records if record["split"] == "train"]
    if not train:
        raise ValueError("There are no records in the train split")
    pixel_sum, squared_sum, pixel_count = 0.0, 0.0, 0
    for record in train:
        resized = letterbox(read_grayscale(record, project_root), image_size)
        channel = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
        pixel_sum += float(channel.sum(dtype=np.float64))
        squared_sum += float(np.square(channel, dtype=np.float64).sum(dtype=np.float64))
        pixel_count += int(channel.size)
    mean = pixel_sum / pixel_count
    std = max(math.sqrt(max(squared_sum / pixel_count - mean**2, 0.0)), 1e-8)
    return {
        "mean": [mean] * 3,
        "std": [std] * 3,
        "image_size": image_size,
        "num_images": len(train),
    }
