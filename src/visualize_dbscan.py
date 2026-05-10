import csv
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from config import INTEGRATED_OBJECTS_CSV, LAYOUT_IMAGE, STATIC_OBJECT_REGISTRY_CSV


def safe_float(value, default=np.nan):
    if value is None:
        return default
    value = str(value).strip()
    if value.lower() in {"none", "nan", "null"}:
        return default
    return float(value) if value else default


def first_present(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() not in {"none", "nan", "null"}:
            return value
    return ""


def load_csv(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_dbscan_clusters(
    integrated_csv: Path = INTEGRATED_OBJECTS_CSV,
    registry_csv: Path = STATIC_OBJECT_REGISTRY_CSV,
    layout_path: Path = LAYOUT_IMAGE,
    label_filter: str | None = None,
    map_width: float = 15.0,
    map_height: float = 12.0,
    max_points: int | None = None,
    point_alpha: float = 0.18,
    point_size: float = 9.0,
    anchor_size: float = 135.0,
    figsize=(13, 9),
):
    integrated_rows = load_csv(integrated_csv)
    registry_rows = load_csv(registry_csv)

    machine_rows = [
        row
        for row in integrated_rows
        if row.get("people_or_machine") == "machine"
        and (label_filter is None or row.get("label") == label_filter or row.get("canonical_label") == label_filter)
    ]
    if max_points is not None and len(machine_rows) > max_points:
        rng = np.random.default_rng(42)
        keep_indices = rng.choice(len(machine_rows), size=max_points, replace=False)
        machine_rows = [machine_rows[index] for index in sorted(keep_indices)]

    unassigned_methods = {"dbscan_noise", "anchor_unassigned", "knn_unassigned"}
    cluster_ids = sorted(
        {
            row.get("object_id", "")
            for row in machine_rows
            if row.get("clustering_method") not in unassigned_methods
        }
    )
    cmap = plt.get_cmap("tab20", max(20, len(cluster_ids)))
    color_by_cluster = {
        object_id: cmap(index % cmap.N)
        for index, object_id in enumerate(cluster_ids)
    }

    fig, ax = plt.subplots(figsize=figsize)
    if Path(layout_path).exists():
        layout_image = mpimg.imread(layout_path)
        ax.imshow(layout_image, extent=[0, map_width, 0, map_height], origin="upper", alpha=0.95)

    point_groups = defaultdict(lambda: {"xs": [], "ys": [], "method": ""})
    for row in machine_rows:
        x = safe_float(row.get("projected_x"))
        y = safe_float(row.get("projected_y"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        object_id = row.get("object_id", "")
        method = row.get("clustering_method", "")
        group_key = ("unassigned", method) if method in unassigned_methods else ("assigned", object_id)
        point_groups[group_key]["xs"].append(x)
        point_groups[group_key]["ys"].append(y)
        point_groups[group_key]["method"] = method

    for (kind, object_id), group in point_groups.items():
        method = group["method"]
        if method in unassigned_methods:
            color = (0.35, 0.35, 0.35, 0.22)
            marker = "x"
        else:
            color = color_by_cluster.get(object_id, (0.2, 0.2, 0.2, point_alpha))
            marker = "o"
        ax.scatter(group["xs"], group["ys"], s=point_size, color=color, alpha=point_alpha, marker=marker, linewidths=0)

    filtered_registry_rows = [
        row
        for row in registry_rows
        if label_filter is None or row.get("canonical_label") == label_filter
    ]
    for row in filtered_registry_rows:
        x = safe_float(first_present(row, "anchor_x", "mean_projected_x"))
        y = safe_float(first_present(row, "anchor_y", "mean_projected_y"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        object_id = row.get("object_id", "")
        color = color_by_cluster.get(object_id, "black")
        ax.scatter(
            [x],
            [y],
            s=anchor_size,
            marker="*",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            x + 0.08,
            y + 0.08,
            f"{object_id}\n{row.get('canonical_label', '')}",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.6},
            zorder=6,
        )

    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Static Object Detection Points and Fixed Anchors")
    ax.grid(alpha=0.2)
    return fig, ax, filtered_registry_rows


def plot_static_clusters(*args, **kwargs):
    return plot_dbscan_clusters(*args, **kwargs)


def summarize_registry(registry_csv: Path = STATIC_OBJECT_REGISTRY_CSV):
    rows = load_csv(registry_csv)
    return [
        {
            "object_id": row.get("object_id", ""),
            "label": row.get("canonical_label", ""),
            "anchor_x": first_present(row, "anchor_x", "mean_projected_x"),
            "anchor_y": first_present(row, "anchor_y", "mean_projected_y"),
            "observations": row.get("observations", ""),
            "first_seen": row.get("first_seen", ""),
            "last_seen": row.get("last_seen", ""),
        }
        for row in rows
    ]
