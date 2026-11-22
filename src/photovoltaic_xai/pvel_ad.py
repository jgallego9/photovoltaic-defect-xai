"""Read PVEL-AD images and their source annotations."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import imagehash
from PIL import Image

from . import SOURCE_CLASS_NAMES

DEFAULT_DATASET_DIR = Path("data/raw/pvel_ad/EL2021")
_BOX_FIELDS = ("class_name", "xmin", "ymin", "xmax", "ymax")


@dataclass(frozen=True)
class _ImageInfo:
    path: Path
    sha256: str
    width: int
    height: int
    phash: int
    thumbnail: tuple[int, ...]


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root, walk_up=True).as_posix()


def _inspect_image(path: Path) -> _ImageInfo:
    try:
        content = path.read_bytes()
        with Image.open(BytesIO(content)) as image:
            grayscale = image.convert("L")
            width, height = grayscale.size
            phash_value = int(str(imagehash.phash(grayscale)), 16)
            thumbnail_image = grayscale.resize((16, 16), Image.Resampling.LANCZOS)
            thumbnail = tuple(thumbnail_image.get_flattened_data())
    except OSError as error:
        raise ValueError(f"Unreadable image: {path}") from error
    return _ImageInfo(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        width=width,
        height=height,
        phash=phash_value,
        thumbnail=thumbnail,
    )


def _parse_int(element: ET.Element, field: str, xml_path: Path) -> int:
    try:
        return int(element.findtext(field) or "")
    except ValueError as error:
        raise ValueError(f"Missing or invalid integer {field} in {xml_path}") from error


def parse_source_annotation(
    xml_path: Path, image_info: _ImageInfo
) -> list[tuple[str, int, int, int, int]]:
    """Return annotated boxes after checking their classes and coordinates."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid source XML: {xml_path}") from error

    filename = (root.findtext("filename") or "").strip()
    if filename and Path(filename).name != image_info.path.name:
        raise ValueError(
            f"XML filename {filename!r} does not match {image_info.path.name!r}: {xml_path}"
        )

    width = _parse_int(root, "size/width", xml_path)
    height = _parse_int(root, "size/height", xml_path)
    if (width, height) != (image_info.width, image_info.height):
        raise ValueError(
            f"Declared size {(width, height)} differs from image size "
            f"{(image_info.width, image_info.height)}: {xml_path}"
        )

    boxes: list[tuple[str, int, int, int, int]] = []
    for object_element in root.findall("./object"):
        class_name = (object_element.findtext("./name") or "").strip()
        if class_name not in SOURCE_CLASS_NAMES:
            raise ValueError(f"Unknown object class {class_name!r}: {xml_path}")
        xmin, ymin, xmax, ymax = (
            _parse_int(object_element, f"bndbox/{field}", xml_path) for field in _BOX_FIELDS[1:]
        )
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"Out-of-bounds or empty box {(xmin, ymin, xmax, ymax)} in {xml_path}")
        boxes.append((class_name, xmin, ymin, xmax, ymax))
    if not boxes:
        raise ValueError(f"Positive source XML has no objects: {xml_path}")
    return boxes


def _sources(dataset_dir: Path) -> list[tuple[Path, Path | None, str]]:
    folders = [
        (f"old_{split}", dataset_dir / split / "JPEGImages", dataset_dir / split / "Annotations")
        for split in ("trainval", "test")
    ]
    folders.append(("good", dataset_dir / "othertypes/good", None))

    sources = []
    for origin, image_dir, annotation_dir in folders:
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        annotations = {}
        if annotation_dir is not None:
            annotations = {
                path.stem: path for path in annotation_dir.iterdir() if path.suffix == ".xml"
            }
            image_stems = {path.stem for path in images}
            if image_stems != annotations.keys():
                raise ValueError(
                    f"{origin} image/XML filename mismatch; "
                    f"missing XML={sorted(image_stems - annotations.keys())[:10]}, "
                    f"missing images={sorted(annotations.keys() - image_stems)[:10]}"
                )
        sources.extend((image, annotations.get(image.stem), origin) for image in images)

    return sources


def read_sources(
    project_root: str | Path, *, dataset_dir: str | Path = DEFAULT_DATASET_DIR
) -> list[dict]:
    """Read PVEL-AD from an absolute or project-relative dataset directory."""
    root = Path(project_root).resolve()

    def read_row(source: tuple[Path, Path | None, str]) -> dict:
        path, annotation, origin = source
        info = _inspect_image(path)
        boxes = parse_source_annotation(annotation, info) if annotation else []
        return {
            "image_path": _relative(path, root),
            "origin": origin,
            "annotation_path": _relative(annotation, root) if annotation else None,
            "sha256": info.sha256,
            "width": info.width,
            "height": info.height,
            "phash": info.phash,
            "thumbnail": info.thumbnail,
            "source_labels": sorted({box[0] for box in boxes}),
            "boxes": [dict(zip(_BOX_FIELDS, box, strict=True)) for box in boxes],
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(read_row, _sources(root / dataset_dir)))
