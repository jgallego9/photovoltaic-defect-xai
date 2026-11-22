"""Show the cooccurrence and defect figures for the PVEL-AD notebook."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from . import CLASS_NAMES
from .notebook_support import figure_style, show_figure

_DEFECT_NAMES = [
    "Grieta",
    "Interrupción de dedo",
    "Núcleo negro",
    "Línea gruesa",
    "Cortocircuito",
    "Dislocación horizontal",
]


def show_cooccurrence(records):
    short_names = ["Grieta", "Dedo", "Núcleo", "Línea", "Cortoc.", "Disloc."]
    positions = np.arange(len(CLASS_NAMES))
    targets = np.asarray([record["labels"] for record in records])
    cooccurrence = targets.T @ targets
    counts = cooccurrence.diagonal()

    with plt.rc_context(figure_style()):
        fig, (ax, bx) = plt.subplots(
            1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]}
        )
        ax.barh(positions, counts, color="#326a8e")
        ax.set_yticks(positions, _DEFECT_NAMES)
        ax.invert_yaxis()
        ax.set(xlabel="Imágenes originales positivas", title="Soporte por tipo de defecto")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for position, count in enumerate(counts):
            label = f"{count:,}".replace(",", " ")
            ax.text(count + 100, position, label, va="center", fontsize=9)
        ax.set_xlim(0, max(counts) * 1.2)

        heatmap = bx.imshow(cooccurrence, cmap="Blues")
        bx.set_xticks(positions, short_names, rotation=45, ha="right")
        bx.set_yticks(positions, short_names)
        bx.set_title("Imágenes con ambos tipos de defecto")
        for (row, column), count in np.ndenumerate(cooccurrence):
            color = "white" if count > 6500 else "#1d2833"
            bx.text(column, row, str(count), ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(heatmap, ax=bx, fraction=0.046, pad=0.04)
        fig.tight_layout()
        show_figure(fig, dpi=200)


def _annotation_bounds(record):
    keys = ("xmin", "ymin", "xmax", "ymax")
    return [tuple(box[key] for key in keys) for box in record["boxes"]]


def _box_size(bounds):
    left, top, right, bottom = bounds
    return right - left, bottom - top


def _box_area(bounds):
    return math.prod(_box_size(bounds))


def _thick_line_score(record):
    bounds = max(_annotation_bounds(record), key=_box_area)
    short_side, long_side = sorted(_box_size(bounds))
    image_side = max(record["width"], record["height"])
    if 0.25 <= long_side / image_side <= 0.5 and short_side / image_side >= 0.01:
        if long_side / short_side >= 4:
            return _box_area(bounds) / (record["width"] * record["height"])
    return 0


def _draw_box(ax, bounds, color, offset=(0, 0), **style):
    left, top, _, _ = bounds
    width, height = _box_size(bounds)
    ax.add_patch(
        Rectangle(
            (left - offset[0], top - offset[1]),
            width,
            height,
            fill=False,
            edgecolor=color,
            **style,
        )
    )


def show_defect_examples(train_records, project_root):
    single_defects = [
        record
        for record in train_records
        if record["localization_valid"] and sum(record["labels"]) == 1
    ]
    examples = []
    for index, (class_name, title) in enumerate(zip(CLASS_NAMES, _DEFECT_NAMES)):
        candidates = (record for record in single_defects if record["labels"][index])
        record = (
            max(candidates, key=_thick_line_score)
            if class_name == "thick_line"
            else next(candidates)
        )
        examples.append((record, class_name, title))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 14}):
        for group in (examples[:3], examples[3:]):
            fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.5))
            for col, (record, class_name, title) in enumerate(group):
                with Image.open(project_root / record["image_path"]) as opened:
                    image = opened.convert("L")
                boxes = _annotation_bounds(record)
                anchor = max(boxes, key=_box_area)
                xmin, ymin, xmax, ymax = anchor
                partial = class_name == "horizontal_dislocation"
                side = 256 if partial else max(128, math.ceil(1.5 * max(_box_size(anchor))))
                width, height = min(side, image.width), min(side, image.height)
                left = min(max((xmin + xmax - width) // 2, 0), image.width - width)
                top = min(max((ymin + ymax - height) // 2, 0), image.height - height)
                crop = (left, top, left + width, top + height)

                for row, shown in enumerate((image, image.crop(crop))):
                    ax = axes[row, col]
                    ax.imshow(shown, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                    offset = (left, top) if row else (0, 0)
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        if row and (x2 <= left or x1 >= crop[2] or y2 <= top or y1 >= crop[3]):
                            continue
                        _draw_box(ax, box, "#ffb000", offset, linewidth=1.5)
                    ax.set(xlim=(-0.5, shown.width - 0.5), ylim=(shown.height - 0.5, -0.5))
                    ax.set_axis_off()

                _draw_box(axes[0, col], crop, "#14cbd9", linewidth=1.6, linestyle=(0, (4, 3)))
                axes[0, col].set_title(f"{title}\n({class_name})", fontsize=16, pad=11)
                label = "Detalle parcial" if partial else "Detalle"
                detail = f"{label}:\n{width} × {height} px"
                if (width, height) == image.size:
                    detail = "Región extensa:\nimagen completa"
                axes[1, col].set_title(detail, fontsize=11, pad=7)

            fig.subplots_adjust(
                left=0.065, right=0.985, top=0.89, bottom=0.075, wspace=0.08, hspace=0.31
            )
            for y, label in ((0.69, "Imagen completa"), (0.27, "Detalle de la anotación")):
                fig.text(0.02, y, label, rotation=90, ha="center", va="center", fontsize=12)
            fig.text(
                0.5,
                0.015,
                "Amarillo: anotaciones originales · Cian discontinuo: zona del detalle",
                ha="center",
                va="bottom",
                fontsize=11,
            )
            show_figure(fig, dpi=240)
