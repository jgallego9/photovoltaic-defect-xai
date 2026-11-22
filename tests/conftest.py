"""Small synthetic sources; the notebooks execute the actual procedures."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def raw_corpus():
    """Create 100 files: two excluded SHA groups and one box-only discrepancy."""
    import numpy as np
    from PIL import Image

    from photovoltaic_xai import CLASS_NAMES

    def create(root):
        base = root / "data/raw/pvel_ad/EL2021"
        train_images, test_images = (
            base / origin / "JPEGImages" for origin in ("trainval", "test")
        )
        train_xml, test_xml = base / "trainval/Annotations", base / "test/Annotations"
        healthy = base / "othertypes/good"
        for directory in (train_images, test_images, train_xml, test_xml, healthy):
            directory.mkdir(parents=True, exist_ok=True)
        cases = [
            (
                f"sample_{i:03}",
                (test_images if i % 2 else train_images) if i < 72 else healthy,
                (test_xml if i % 2 else train_xml) if i < 72 else None,
                None,
                [CLASS_NAMES[i % 6]] if i < 72 else [],
                5,
            )
            for i in range(96)
        ] + [
            ("excluded_mixed", train_images, train_xml, None, ["crack", "star_crack"], 5),
            ("conflict_train", train_images, train_xml, None, ["crack"], 5),
            (
                "conflict_test",
                test_images,
                test_xml,
                train_images / "conflict_train.png",
                ["finger"],
                5,
            ),
            ("boxes_copy", test_images, test_xml, train_images / "sample_000.png", ["crack"], 6),
        ]
        rng = np.random.default_rng(29)
        for name, image_dir, annotation_dir, copied_from, labels, xmax in cases:
            image_path = image_dir / f"{name}.png"
            if copied_from is None:
                Image.fromarray(rng.integers(0, 256, (12, 16), dtype=np.uint8)).save(image_path)
            else:
                image_path.write_bytes(copied_from.read_bytes())
            if annotation_dir is not None:
                objects = "".join(
                    f"<object><name>{label}</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
                    f"<xmax>{xmax}</xmax><ymax>5</ymax></bndbox></object>"
                    for label in labels
                )
                (annotation_dir / f"{name}.xml").write_text(
                    f"<annotation><filename>{name}.png</filename>"
                    f"<size><width>16</width><height>12</height></size>{objects}</annotation>"
                )
        return root

    return create
