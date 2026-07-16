from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from .filters import filter_small_patches
from .io import validate_matrix


def save_label_png(
    matrix: np.ndarray,
    output_path: str | Path,
    classes: list[int] | None = None,
    min_patch_cells: int = 0,
    connectivity: int = 4,
    max_grid_cells: int = 20_000,
) -> Path:
    labels = validate_matrix(matrix)
    labels = filter_small_patches(labels, min_patch_cells, connectivity)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "Saving PNG label maps requires matplotlib. Install matplotlib or omit --plot-dir."
        ) from exc

    class_values = classes or sorted(int(value) for value in np.unique(labels) if value > 0)
    cmap = ListedColormap(_label_colors(len(class_values)))
    boundaries = np.arange(-0.5, len(class_values) + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    display = _remap_to_color_indices(labels, class_values)

    height, width = labels.shape
    figure_width = min(14.0, max(4.0, width / 18.0))
    figure_height = min(14.0, max(4.0, height / 18.0))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=180)
    ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(output_path.stem, fontsize=8)

    if labels.size <= max_grid_cells:
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.35)

    legend_handles = [Patch(facecolor="#ffffff", edgecolor="#bbbbbb", label="0")]
    for index, cls in enumerate(class_values, start=1):
        legend_handles.append(
            Patch(facecolor=cmap(index), edgecolor="none", label=str(cls))
        )
    if len(legend_handles) <= 25:
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=6,
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _remap_to_color_indices(labels: np.ndarray, classes: list[int]) -> np.ndarray:
    remapped = np.zeros(labels.shape, dtype=np.int64)
    for index, cls in enumerate(classes, start=1):
        remapped[labels == cls] = index
    return remapped


def _label_colors(class_count: int) -> list[str]:
    palette = [
        "#ffffff",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    if class_count <= len(palette) - 1:
        return palette[: class_count + 1]
    extra = []
    for index in range(class_count - (len(palette) - 1)):
        hue = index / max(1, class_count - (len(palette) - 1))
        extra.append(_hsv_to_hex(hue, 0.65, 0.85))
    return palette + extra


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
