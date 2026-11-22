"""Curate PVEL-AD records, assign splits, and persist the experiment manifest."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from . import CLASS_NAMES, PROTOCOL_ID, SPLIT_SEED
from .pvel_ad import _BOX_FIELDS

_PREPARED_PATH = Path("data/prepared") / f"{PROTOCOL_ID}.json"


class _HashIndex:
    """Exact radius-two search using three disjoint bands of the 64-bit pHash."""

    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._buckets: dict[tuple[int, int], list[str]] = defaultdict(list)

    @staticmethod
    def _band_keys(value: int) -> tuple[tuple[int, int], ...]:
        return (
            (0, value & ((1 << 22) - 1)),
            (1, (value >> 22) & ((1 << 21) - 1)),
            (2, (value >> 43) & ((1 << 21) - 1)),
        )

    def add(self, value: int, payload: str) -> None:
        self._values[payload] = value
        for key in self._band_keys(value):
            self._buckets[key].append(payload)

    def query(self, value: int) -> list[tuple[int, str]]:
        payloads = {payload for key in self._band_keys(value) for payload in self._buckets[key]}
        return [
            (distance, payload)
            for payload in payloads
            if (distance := (value ^ self._values[payload]).bit_count()) <= 2
        ]


def _digest(payload: dict | list) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def deduplicate_sources(sources: list[dict]) -> list[dict]:
    """Keep one record per file hash and flag conflicting annotations."""
    ordered = sorted(sources, key=itemgetter("sha256", "image_path"))
    rows = []
    for sha, group in groupby(ordered, key=itemgetter("sha256")):
        copies = list(group)
        first = copies[0]
        label_sets = {tuple(copy["source_labels"]) for copy in copies}
        box_sets = {
            frozenset(tuple(box[key] for key in _BOX_FIELDS) for box in copy["boxes"])
            for copy in copies
        }
        boxes = sorted(set().union(*box_sets))
        rows.append(
            {
                "id": "sha_" + sha,
                "sha256": sha,
                "image_path": first["image_path"],
                "width": int(first["width"]),
                "height": int(first["height"]),
                "phash": int(first["phash"]),
                "thumbnail": first["thumbnail"],
                "origin": sorted({copy["origin"] for copy in copies}),
                "source_labels": sorted(set().union(*label_sets)),
                "boxes": [dict(zip(_BOX_FIELDS, box, strict=True)) for box in boxes],
                "label_conflict": len(label_sets) > 1,
                "box_conflict": len(box_sets) > 1,
            }
        )
    return rows


def group_visual_candidates(inventory: list[dict]) -> list[dict]:
    """Group visually similar images for splitting while preserving their annotations."""
    representatives = {image["id"]: image for image in inventory}
    parents = {identifier: identifier for identifier in representatives}

    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    index = _HashIndex()
    for identifier, info in representatives.items():
        for _, other in index.query(info["phash"]):
            mae = sum(
                abs(a - b)
                for a, b in zip(
                    info["thumbnail"],
                    representatives[other]["thumbnail"],
                    strict=True,
                )
            ) / len(info["thumbnail"])
            if mae <= 8:
                left, right = sorted((find(identifier), find(other)))
                parents[right] = left
        index.add(info["phash"], identifier)
    components = defaultdict(list)
    for identifier in parents:
        components[find(identifier)].append(identifier)
    group_ids = {
        identifier: (members[0] if len(members) == 1 else "perceptual_" + _digest(members))
        for members in components.values()
        for identifier in members
    }
    return [dict(image, group_id=group_ids[image["id"]]) for image in inventory]


def select_records(inventory: list[dict]) -> tuple[list[dict], dict]:
    """Select the six study classes and keep box conflicts for classification only."""
    records = []
    excluded_ids = []
    classes = set(CLASS_NAMES)
    record_fields = ("id", "sha256", "image_path", "width", "height", "origin", "group_id")
    for image in inventory:
        labels = image["source_labels"]
        if image["label_conflict"] or not classes.issuperset(labels):
            excluded_ids.append(image["id"])
            continue
        valid_boxes = not image["box_conflict"]
        records.append(
            {
                **{field: image[field] for field in record_fields},
                "labels": [int(name in labels) for name in CLASS_NAMES],
                "source": "positive" if labels else "good",
                "boxes": image["boxes"] if valid_boxes else [],
                "localization_valid": valid_boxes,
            }
        )

    targets = np.asarray([record["labels"] for record in records]).reshape(-1, len(CLASS_NAMES))
    selection = {
        "unique_images": len(inventory),
        "selected_images": len(records),
        "excluded_ids": sorted(excluded_ids),
        "support": dict(zip(CLASS_NAMES, targets.sum(axis=0, dtype=int).tolist(), strict=True)),
        "box_conflicts_retained": sum(not record["localization_valid"] for record in records),
    }
    return records, selection


def build_splits(records: list[dict]) -> list[dict]:
    """Stratify whole groups at 70/15/15."""
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record["labels"] + [int(not any(record["labels"]))])
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three groups are required for train/val/test")
    labels = np.asarray([np.max(groups[key], axis=0) for key in keys], dtype=np.int8)
    placeholder = np.zeros((len(keys), 1), dtype=np.int8)
    heldout_count = max(2, min(len(keys) - 1, round(0.30 * len(keys))))
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=heldout_count, random_state=SPLIT_SEED
    )
    train, heldout = next(first.split(placeholder, labels))
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=SPLIT_SEED + 1
    )
    val, test = next(second.split(placeholder[heldout], labels[heldout]))
    assignments = {keys[index]: "train" for index in train}
    assignments.update({keys[heldout[index]]: "val" for index in val})
    assignments.update({keys[heldout[index]]: "test" for index in test})
    return sorted(
        (dict(record, split=assignments[record["group_id"]]) for record in records),
        key=lambda record: (record["split"], record["source"], record["id"]),
    )


def save_preparation(payload: dict, project_root: str | Path) -> Path:
    """Save prepared data without overwriting different experimental content."""
    fingerprint = _digest(
        {key: value for key, value in payload.items() if key != "data_fingerprint"}
    )
    if payload.get("data_fingerprint", fingerprint) != fingerprint:
        raise ValueError("The prepared data fingerprint does not match its content")
    payload["data_fingerprint"] = fingerprint
    destination = Path(project_root) / _PREPARED_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != payload:
            raise FileExistsError(f"Different prepared data already exist: {destination}")
    else:
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return destination


def load_preparation(project_root: str | Path) -> dict:
    """Read prepared data and verify its fingerprint and experimental protocol."""
    root = Path(project_root).resolve()
    payload = json.loads((root / _PREPARED_PATH).read_text())
    content = {key: value for key, value in payload.items() if key != "data_fingerprint"}
    if _digest(content) != payload["data_fingerprint"]:
        raise ValueError("Prepared data fingerprint does not match its content")
    if (
        payload["protocol_id"] != PROTOCOL_ID
        or payload["class_names"] != list(CLASS_NAMES)
        or payload["seed"] != SPLIT_SEED
    ):
        raise ValueError("Preparation requires the current protocol, classes and seed")
    return payload | {"root": root}
