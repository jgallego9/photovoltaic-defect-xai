"""Independent checks of annotations, geometry, splits and multilabel balancing."""

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from photovoltaic_xai import CLASS_NAMES, PROTOCOL_ID, SOURCE_CLASS_NAMES, SPLIT_SEED
from photovoltaic_xai.balance import BalancedPatternSampler, build_balance_plan
from photovoltaic_xai.dataset import PairedELTransform, PVELADDataset, compute_normalization
from photovoltaic_xai.preparation import (
    _HashIndex,
    build_splits,
    deduplicate_sources,
    group_visual_candidates,
    load_preparation,
    save_preparation,
)
from photovoltaic_xai.pvel_ad import (
    DEFAULT_DATASET_DIR,
    _inspect_image,
    parse_source_annotation,
    read_sources,
)


def _record(root, name="cell", split="train", healthy=False):
    boxes = (
        []
        if healthy
        else [
            {"class_name": "crack", "xmin": 2, "ymin": 2, "xmax": 6, "ymax": 4},
            {"class_name": "crack", "xmin": 11, "ymin": 6, "xmax": 15, "ymax": 8},
            {"class_name": "horizontal_dislocation", "xmin": 2, "ymin": 6, "xmax": 9, "ymax": 8},
        ]
    )
    pixels = np.full((10, 20), 32, dtype=np.uint8)
    for box in boxes:
        pixels[box["ymin"] : box["ymax"], box["xmin"] : box["xmax"]] = 220
    path = root / f"{name}.png"
    Image.fromarray(pixels).save(path)
    return {
        "id": name,
        "image_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": 20,
        "height": 10,
        "boxes": boxes,
        "localization_valid": True,
        "labels": [int(not healthy and i in (0, 5)) for i in range(6)],
        "source": "good" if healthy else "positive",
        "split": split,
    }


def test_xml_reader_checks_source_vocabulary_and_spatial_annotations(tmp_path, raw_corpus):
    raw_corpus(tmp_path)
    base = tmp_path / "data/raw/pvel_ad/EL2021/trainval"
    xml = base / "Annotations/sample_000.xml"
    original = xml.read_text().replace("<size>", "<owner><name>Subinyi</name></owner><size>")
    info = _inspect_image(base / "JPEGImages/sample_000.png")
    for name in SOURCE_CLASS_NAMES:
        xml.write_text(original.replace("<name>crack</name>", f"<name>{name}</name>"))
        assert {box[0] for box in parse_source_annotation(xml, info)} == {name}
    for content in (
        original.replace("<name>crack</name>", "<name>unknown</name>"),
        original.replace("<width>16</width>", "<width>15</width>"),
        original.replace("<xmin>1</xmin>", "<xmin>5</xmin>"),
    ):
        xml.write_text(content)
        with pytest.raises(ValueError):
            parse_source_annotation(xml, info)


def test_read_sources_accepts_an_external_dataset_directory(tmp_path, raw_corpus):
    raw_corpus(tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    sources = read_sources(project, dataset_dir=tmp_path / DEFAULT_DATASET_DIR)

    assert len(sources) == 100
    assert all((project / row["image_path"]).is_file() for row in sources)
    assert all(
        (project / row["annotation_path"]).is_file() for row in sources if row["annotation_path"]
    )


def test_exact_and_visual_groups_keep_annotations_and_never_cross_splits(tmp_path, raw_corpus):
    raw_corpus(tmp_path)
    sources = read_sources(tmp_path)
    inventory = group_visual_candidates(deduplicate_sources(sources))
    assert len(sources) == 100 and len(inventory) == 98
    conflict = next(row for row in inventory if row["label_conflict"])
    assert conflict["source_labels"] == ["crack", "finger"]
    original = next(row for row in sources if row["image_path"].endswith("sample_001.png"))
    path = tmp_path / original["image_path"]
    recoded = path.with_name("recoded.png")
    with Image.open(path) as image:
        image.save(recoded, compress_level=0)
    xml = tmp_path / original["annotation_path"]
    xml.with_name("recoded.xml").write_text(
        xml.read_text().replace("sample_001.png", "recoded.png")
    )
    inventory = group_visual_candidates(deduplicate_sources(read_sources(tmp_path)))
    group = next(row["group_id"] for row in inventory if row["sha256"] == original["sha256"])
    recoded_group = next(
        row["group_id"] for row in inventory if row["image_path"].endswith("recoded.png")
    )
    assert recoded_group == group
    records = [
        dict(
            row,
            source="synthetic",
            labels=[int(name in row["source_labels"]) for name in CLASS_NAMES],
        )
        for row in inventory
    ]
    splits = build_splits(records)
    assert splits == build_splits(records)
    assert all(
        len({row["split"] for row in splits if row["group_id"] == key}) == 1
        for key in {row["group_id"] for row in splits}
    )
    index = _HashIndex()
    values = [
        0,
        1 << 21,
        1 << 22,
        1 << 42,
        1 << 43,
        (1 << 0) | (1 << 63),
        (1 << 22) | (1 << 43),
        (1 << 21) | (1 << 42),
        (1 << 64) - 1,
    ]
    for i, value in enumerate(values):
        index.add(value, str(i))
    for query in values:
        assert sorted(index.query(query)) == sorted(
            ((query ^ v).bit_count(), str(i))
            for i, v in enumerate(values)
            if (query ^ v).bit_count() <= 2
        )


def test_six_labels_box_union_and_shared_geometry(tmp_path, monkeypatch):
    positive, healthy = _record(tmp_path), _record(tmp_path, "healthy", healthy=True)
    positive["origin"] = [
        "old_test"
    ]  # The original source does not prevent assignment to the new train split.
    transform = PairedELTransform(40, mean=(0.0,) * 3, std=(1.0,) * 3)
    dataset = PVELADDataset([positive, healthy], tmp_path, transform, return_masks=True)
    original, good = dataset[0], dataset[1]
    assert original["labels"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert original["image"].shape == (3, 40, 40) and original["masks"].shape == (6, 40, 40)
    assert original["masks"][0].sum() == 64  # Two boxes, one positive label.
    assert not good["labels"].any() and not good["masks"].any()
    draws = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(next(draws)))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    augmented = PVELADDataset(
        [positive],
        tmp_path,
        PairedELTransform(40, mean=(0.0,) * 3, std=(1.0,) * 3, training=True),
        return_masks=True,
    )[0]
    for field in ("image", "masks", "valid_mask"):
        assert torch.equal(augmented[field], original[field].flip(-1, -2))
    valid = augmented["valid_mask"]
    assert valid.sum() == 800 and not (augmented["masks"].bool() & ~valid).any()
    y, x = augmented["masks"][5].nonzero(as_tuple=True)
    assert x.max() - x.min() > y.max() - y.min()
    positive["localization_valid"] = False
    assert not dataset[0]["masks"].any() and dataset[0]["labels"].sum() == 2
    with pytest.raises(PermissionError):
        PVELADDataset([dict(positive, split="test", origin=["old_trainval"])], tmp_path, transform)


def test_normalization_reads_only_train_pixels_and_includes_padding(tmp_path):
    train, val = _record(tmp_path), _record(tmp_path, "val", split="val")
    (tmp_path / val["image_path"]).unlink()
    stats = compute_normalization([train, val], tmp_path, image_size=20)
    with Image.open(tmp_path / train["image_path"]) as image:
        pixels = np.pad(np.asarray(image, dtype=np.float32) / np.float32(255.0), ((5, 5), (0, 0)))
    mean = pixels.sum(dtype=np.float64) / pixels.size
    std = (np.square(pixels, dtype=np.float64).sum(dtype=np.float64) / pixels.size - mean**2) ** 0.5
    assert stats["mean"] == [mean] * 3 and stats["std"] == [std] * 3
    assert stats["num_images"] == 1
    Image.new("L", (20, 10), 255).save(tmp_path / train["image_path"])
    with pytest.raises(ValueError, match="changed"):
        compute_normalization([train], tmp_path)


def _single_labels(counts, healthy=0):
    return np.concatenate(
        [
            np.repeat(np.eye(6, dtype=int)[i : i + 1], count, axis=0)
            for i, count in enumerate(counts)
        ]
        + [np.zeros((healthy, 6), dtype=int)]
    )


def _plan(labels):
    ids = [f"original-{i:03}" for i in range(len(labels))]
    return build_balance_plan(labels, ids=ids), ids


def test_milp_optimum_and_ties_match_exhaustive_search():
    labels = np.tile(np.array([[0, 1]] + [[1, 0]] * 3 + [[1, 1]] * 2), (1, 3))
    plan, _ = _plan(labels)
    patterns, counts = np.array(plan.patterns), np.array(plan.original_counts)
    feasible = []
    for allocation in itertools.product(*(range(1, 3 * n + 1) for n in counts)):
        if np.all(patterns.T @ allocation == 4):  # Median 4; integer band [4, 4].
            objective = sum(
                abs(len(labels) * x - n * sum(allocation)) for x, n in zip(allocation, counts)
            )
            feasible.append((objective, allocation))
    objective = sum(
        abs(len(labels) * x - n * sum(plan.allocations)) for x, n in zip(plan.allocations, counts)
    )
    assert (objective, plan.allocations) == min(feasible)
    assert _plan(labels)[0].allocations == plan.allocations


def test_sampling_keeps_patterns_caps_and_rotates_every_original():
    labels = np.vstack(
        (_single_labels([10, 40, 10, 10, 10, 3], healthy=25), [[1, 1, 0, 0, 0, 0]] * 2)
    )
    plan, ids = _plan(labels)
    sampler = BalancedPatternSampler(labels, plan, ids=ids)
    seen, streams = set(), {pattern: [] for pattern in (*plan.patterns, (0,) * 6)}
    for epoch in range(8):
        sampler.set_epoch(epoch)
        selected = list(sampler)
        assert max(Counter(selected).values()) <= 3
        assert Counter(tuple(row) for row in labels[selected] if row.any()) == dict(
            zip(plan.patterns, plan.allocations)
        )
        assert np.all((labels[selected].sum(0) >= 9) & (labels[selected].sum(0) <= 11))
        assert np.count_nonzero(labels[selected].sum(1) == 0) == 10
        seen.update(selected)
        for i in selected:
            streams[tuple(labels[i])].append(i)
    assert len(seen) == len(labels)
    for pattern, stream in streams.items():
        originals = set(np.flatnonzero(np.all(labels == pattern, axis=1)))
        assert set(stream[: len(originals)]) == originals
    repeated = BalancedPatternSampler(labels, plan, ids=ids)
    repeated.set_epoch(7)
    assert list(sampler) == list(repeated)
    with pytest.raises(ValueError):
        BalancedPatternSampler(labels, plan, ids=ids[::-1])


def test_rounding_infeasible_balance_and_individual_pattern_cap():
    labels = _single_labels([12, 12, 12, 13, 13, 13], healthy=100)
    plan, _ = _plan(labels)
    assert np.all(
        (np.array(plan.patterns).T @ plan.allocations >= 12)
        & (np.array(plan.patterns).T @ plan.allocations <= 13)
    )
    sampler = BalancedPatternSampler(
        labels, plan, ids=[f"original-{i:03}" for i in range(len(labels))]
    )
    assert np.count_nonzero(labels[list(sampler)].sum(1) == 0) == 12  # round(12.5).
    for impossible in (
        _single_labels([10, 10, 10, 10, 10, 2]),
        _single_labels([10] * 6, healthy=2),
    ):
        with pytest.raises(ValueError):
            _plan(impossible)
    labels = np.vstack((_single_labels([10] * 6), np.ones((1, 6), dtype=int)))
    plan, ids = _plan(labels)
    invalid = replace(plan, allocations=tuple(4 if all(p) else 7 for p in plan.patterns))
    assert (np.asarray(invalid.patterns).T @ invalid.allocations == 11).all()
    with pytest.raises(ValueError, match="exposure cap"):
        BalancedPatternSampler(labels, invalid, ids=ids)


def test_prepared_json_integrity_and_no_overwrite(tmp_path):
    payload = {"protocol_id": PROTOCOL_ID, "class_names": list(CLASS_NAMES), "seed": SPLIT_SEED}
    path = save_preparation(payload, tmp_path)
    assert load_preparation(tmp_path) == payload | {"root": tmp_path}
    assert save_preparation(payload, tmp_path) == path
    with pytest.raises(FileExistsError):
        save_preparation(
            {k: v for k, v in payload.items() if k != "data_fingerprint"} | {"records": []},
            tmp_path,
        )
    path.write_text(json.dumps(payload | {"seed": 43}))
    with pytest.raises(ValueError, match="fingerprint"):
        load_preparation(tmp_path)
