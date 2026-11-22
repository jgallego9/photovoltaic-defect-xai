"""Check the experiment notebook sources."""

from pathlib import Path

import nbformat

NOTEBOOK_NAMES = (
    "01_data_preparation.ipynb",
    "02_reference_models.ipynb",
    "03_reference_errors_xai.ipynb",
    "04_global_regional_variant.ipynb",
    "05_test_and_cost.ipynb",
    "06_variant_xai_comparison.ipynb",
)


def validate_notebooks(root):
    for name in NOTEBOOK_NAMES:
        notebook = nbformat.read(root / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell-{index}", "exec")
                if any(output.output_type == "error" for output in cell.get("outputs", [])):
                    raise ValueError(f"Saved execution error: {name}, cell {index}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    validate_notebooks(root)
    print(f"Checked {len(NOTEBOOK_NAMES)} notebooks")
