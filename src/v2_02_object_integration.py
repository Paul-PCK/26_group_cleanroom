import argparse
import csv
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_config


# DBSCAN defaults
# these values are used when the notebook does not override label-specific settings.
DEFAULT_DBSCAN_PARAMS_BY_LABEL = {
    "Machine": {"eps": 0.40, "min_samples": 8},
    "Light": {"eps": 0.25, "min_samples": 8},
    "Screen": {"eps": 0.15, "min_samples": 10},
    "Window": {"eps": 0.20, "min_samples": 5},
}


@dataclass
class ObjectIntegrationSettings:
    # store the paths and parameters used by v2_02 object integration.
    input_csv: Path
    output_csv: Path
    registry_csv: Path
    layout_preview_png: Path
    outlier_summary_png: Path
    frame_iou_threshold: float
    track_person: bool


# command line interface
# notebook usage passes these settings through ObjectIntegrationSettings.
def parse_args():
    # collect object integration settings and output paths.
    parser = argparse.ArgumentParser(description="Build v2 static anchors from projected detections.")
    parser.add_argument("--input-csv", type=Path, default=v2_config.V2_PROJECTED_DETECTIONS_CSV)
    parser.add_argument("--output-csv", type=Path, default=v2_config.V2_INTEGRATED_OBJECTS_CSV)
    parser.add_argument("--registry-csv", type=Path, default=v2_config.V2_STATIC_OBJECT_REGISTRY_CSV)
    parser.add_argument("--layout-preview-png", type=Path, default=v2_config.V2_DBSCAN_LAYOUT_PREVIEW_PNG)
    parser.add_argument("--outlier-summary-png", type=Path, default=v2_config.V2_DBSCAN_OUTLIER_SUMMARY_PNG)
    parser.add_argument("--frame-iou-threshold", type=float, default=0.35)
    parser.add_argument("--track-person", action="store_true")
    return parser.parse_args()


# shared parsing utilities
# timestamp and numeric parsing are kept strict so malformed rows fail visibly.
def parse_timestamp(value):
    # parse supported timestamp strings for stable grouping and sorting.
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")


def normalized_timestamp(value):
    # write timestamps in one consistent downstream format.
    return parse_timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(row, key, default=math.nan):
    # parse a numeric CSV field and let invalid values fail directly.
    return float(row.get(key, default))


def load_rows(path: Path):
    # load CSV rows as dictionaries while preserving source column names.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    # write CSV output with an explicit column order.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# same-frame geometry
# bbox overlap and projected distance define duplicate detections within one image.
def row_geometry(row):
    # return bbox and projected coordinates as numeric values.
    return (
        safe_float(row, "bbox_x0", 0.0),
        safe_float(row, "bbox_y0", 0.0),
        safe_float(row, "bbox_x1", 0.0),
        safe_float(row, "bbox_y1", 0.0),
        safe_float(row, "projected_x", 0.0),
        safe_float(row, "projected_y", 0.0),
    )


def bbox_area(row):
    # compute bbox area in image-coordinate units.
    x0, y0, x1, y1, _, _ = row_geometry(row)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(row_a, row_b):
    # compute intersection over union for two same-frame bboxes.
    ax0, ay0, ax1, ay1, _, _ = row_geometry(row_a)
    bx0, by0, bx1, by1, _, _ = row_geometry(row_b)
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    if inter_area <= 0:
        return 0.0
    union_area = bbox_area(row_a) + bbox_area(row_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def confidence_value(row):
    # use confidence to choose one representative from duplicate detections.
    return safe_float(row, "confidence", -1.0)


def is_person_label(label):
    # identify dynamic person rows separately from static object rows.
    return str(label).strip().lower() == v2_config.PERSON_LABEL


def same_frame_key(row):
    # group detections by image name only.
    image_name = str(row.get("image_name") or "").strip()
    if not image_name:
        raise ValueError("Missing image_name. V2 same-frame dedupe requires image_name.")
    return image_name


def should_merge_same_frame(row_a, row_b, iou_threshold):
    # merge only same-label detections with overlapping image bboxes.
    if str(row_a.get("label") or "").strip() != str(row_b.get("label") or "").strip():
        return False
    return bbox_iou(row_a, row_b) >= iou_threshold


def choose_representative(rows):
    # keep the most confident detection from a connected duplicate component.
    ranked = sorted(rows, key=lambda row: (confidence_value(row), bbox_area(row)), reverse=True)
    return deepcopy(ranked[0])


def dedupe_same_frame(rows, iou_threshold=0.35):
    # reduce duplicated image detections before anchor clustering.
    grouped = defaultdict(list)
    for row in rows:
        grouped[same_frame_key(row)].append(row)

    deduped = []
    for frame_key in sorted(grouped):
        frame_rows = grouped[frame_key]
        visited = [False] * len(frame_rows)
        for index, row in enumerate(frame_rows):
            if visited[index]:
                continue
            component = [row]
            visited[index] = True
            queue = [index]
            while queue:
                current_index = queue.pop()
                current_row = frame_rows[current_index]
                for candidate_index, candidate_row in enumerate(frame_rows):
                    if visited[candidate_index]:
                        continue
                    if should_merge_same_frame(current_row, candidate_row, iou_threshold):
                        visited[candidate_index] = True
                        component.append(candidate_row)
                        queue.append(candidate_index)
            deduped.append(choose_representative(component))
    return deduped


# DBSCAN anchor construction
# static labels are clustered independently so anchors are learned within each class.
def label_key(label):
    # convert labels into stable object_id prefixes.
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_") or "object"


def run_dbscan(points, eps, min_samples):
    # cluster projected points without assuming a fixed number of anchors.
    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install scikit-learn before running v2 DBSCAN integration.") from exc
    if not points:
        return []
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(np.asarray(points, dtype=float)).tolist()


def point_inside_region(x, y, region):
    # check whether one projected point is inside a named rectangular region.
    _name, x0, y0, x1, y1 = region
    return float(x0) <= x <= float(x1) and float(y0) <= y <= float(y1)


def bounds_exclusion_reason(x, y, bounds):
    # apply simple numeric x/y bounds and return the first failed rule.
    bounds = bounds or {}
    if "min_x" in bounds and x < float(bounds["min_x"]):
        return f"projected_x < {float(bounds['min_x']):g}"
    if "max_x" in bounds and x >= float(bounds["max_x"]):
        return f"projected_x >= {float(bounds['max_x']):g}"
    if "min_y" in bounds and y < float(bounds["min_y"]):
        return f"projected_y < {float(bounds['min_y']):g}"
    if "max_y" in bounds and y >= float(bounds["max_y"]):
        return f"projected_y >= {float(bounds['max_y']):g}"
    return ""


def anchor_learning_decision(row, dbscan_general_rule=None, dbscan_label_rules_by_label=None):
    # decide once whether a row can be used to learn DBSCAN anchors.
    x = safe_float(row, "projected_x", math.inf)
    y = safe_float(row, "projected_y", math.inf)
    general_reason = bounds_exclusion_reason(x, y, dbscan_general_rule)
    if general_reason:
        return False, f"general_rule: {general_reason}"

    label_rules = (dbscan_label_rules_by_label or {}).get(row.get("label", ""), {})
    label_reason = bounds_exclusion_reason(x, y, label_rules)
    if label_reason:
        return False, f"label_rule: {label_reason}"

    allowed_regions = label_rules.get("allowed_regions", [])
    if allowed_regions and not any(point_inside_region(x, y, region) for region in allowed_regions):
        return False, "label_rule: outside_allowed_regions"
    return True, ""


def build_registry_and_assign_rows(
    rows,
    dbscan_params_by_label=None,
    dbscan_general_rule=None,
    dbscan_label_rules_by_label=None,
    track_person=False,
):
    # learn anchors by label and assign every same-label row to a learned anchor.
    dbscan_params_by_label = dbscan_params_by_label or DEFAULT_DBSCAN_PARAMS_BY_LABEL
    person_rows = []
    static_rows_by_label = defaultdict(list)
    for row in rows:
        row = deepcopy(row)
        row["timestamp"] = normalized_timestamp(row["timestamp"])
        if is_person_label(row.get("label", "")) and not track_person:
            person_rows.append(row)
        else:
            static_rows_by_label[row.get("label", "")].append(row)

    integrated_rows = []
    registry_rows = []
    outlier_summary = []

    person_counter = 0
    for row in person_rows:
        # persons remain frame-level detections unless person tracking is enabled.
        person_counter += 1
        row["object_id"] = f"person_frame_{person_counter:05d}"
        row["anchor_x"] = ""
        row["anchor_y"] = ""
        integrated_rows.append(row)

    for label in sorted(static_rows_by_label):
        # run DBSCAN independently for each label using label-specific parameters.
        label_rows = static_rows_by_label[label]
        if label not in dbscan_params_by_label:
            raise ValueError(f"Missing DBSCAN parameters for label: {label}")
        params = dbscan_params_by_label[label]
        eps = float(params["eps"])
        min_samples = int(params["min_samples"])
        anchor_decisions = [
            anchor_learning_decision(
                row,
                dbscan_general_rule=dbscan_general_rule,
                dbscan_label_rules_by_label=dbscan_label_rules_by_label,
            )
            for row in label_rows
        ]
        anchor_training_indices = [index for index, decision in enumerate(anchor_decisions) if decision[0]]
        anchor_training_points = [
            (safe_float(label_rows[index], "projected_x", 0.0), safe_float(label_rows[index], "projected_y", 0.0))
            for index in anchor_training_indices
        ]
        cluster_labels = run_dbscan(anchor_training_points, eps=eps, min_samples=min_samples)
        cluster_label_by_row_index = {
            row_index: cluster_label
            for row_index, cluster_label in zip(anchor_training_indices, cluster_labels)
        }
        cluster_to_indices = defaultdict(list)
        for row_index, cluster_label in cluster_label_by_row_index.items():
            cluster_to_indices[cluster_label].append(row_index)

        cluster_records = []
        for cluster_label, indices in sorted(cluster_to_indices.items()):
            if cluster_label == -1:
                continue
            # use the median projected position as the stable anchor for the cluster.
            xs = [safe_float(label_rows[index], "projected_x", 0.0) for index in indices]
            ys = [safe_float(label_rows[index], "projected_y", 0.0) for index in indices]
            cluster_records.append(
                {
                    "cluster_label": cluster_label,
                    "indices": indices,
                    "anchor_x": float(np.median(xs)),
                    "anchor_y": float(np.median(ys)),
                    "mean_x": float(np.mean(xs)),
                    "mean_y": float(np.mean(ys)),
                }
            )

        cluster_records = sorted(cluster_records, key=lambda item: (item["anchor_y"], item["anchor_x"]))
        cluster_lookup = {}
        for object_index, record in enumerate(cluster_records, start=1):
            object_id = f"{label_key(label)}_{object_index}"
            cluster_lookup[record["cluster_label"]] = {
                "object_id": object_id,
                "anchor_x": record["anchor_x"],
                "anchor_y": record["anchor_y"],
            }
            cluster_rows = [label_rows[index] for index in record["indices"]]
            registry_rows.append(
                {
                    "object_id": object_id,
                    "label": label,
                    "anchor_x": f"{record['anchor_x']:.6f}",
                    "anchor_y": f"{record['anchor_y']:.6f}",
                    "observations": str(len(record["indices"])),
                    "first_seen": min(row["timestamp"] for row in cluster_rows),
                    "last_seen": max(row["timestamp"] for row in cluster_rows),
                    "dbscan_eps": f"{eps:.6f}",
                    "dbscan_min_samples": str(min_samples),
                }
            )

        noise_count = sum(1 for cluster_label in cluster_labels if cluster_label == -1)
        anchor_rule_excluded_count = len(label_rows) - len(anchor_training_indices)
        outlier_summary.append(
            {
                "label": label,
                "rows": len(label_rows),
                "anchor_training_rows": len(anchor_training_indices),
                "anchor_rule_excluded_rows": anchor_rule_excluded_count,
                "anchors": len(cluster_records),
                "dbscan_noise_rows": noise_count,
                "dbscan_noise_ratio": noise_count / len(anchor_training_indices) if anchor_training_indices else 0.0,
                "dbscan_eps": eps,
                "dbscan_min_samples": min_samples,
            }
        )

        for index, row in enumerate(label_rows):
            # all rows are assigned back to the nearest same-label anchor after anchor learning.
            row = deepcopy(row)
            anchor_valid, anchor_reason = anchor_decisions[index]
            point = (safe_float(row, "projected_x", 0.0), safe_float(row, "projected_y", 0.0))
            cluster_label = cluster_label_by_row_index.get(index, -1)
            assigned_cluster = cluster_label if cluster_label != -1 else nearest_cluster_for_point(point, cluster_lookup)
            if assigned_cluster is None:
                row["object_id"] = f"{label_key(label)}_unassigned"
                row["anchor_x"] = row.get("projected_x", "")
                row["anchor_y"] = row.get("projected_y", "")
            else:
                anchor = cluster_lookup[assigned_cluster]
                row["object_id"] = anchor["object_id"]
                row["anchor_x"] = f"{anchor['anchor_x']:.6f}"
                row["anchor_y"] = f"{anchor['anchor_y']:.6f}"
            integrated_rows.append(row)

    integrated_rows = sorted(
        integrated_rows,
        key=lambda row: (
            parse_timestamp(row["timestamp"]),
            str(row.get("image_name", "")),
            str(row.get("object_id", "")),
        ),
    )
    return integrated_rows, registry_rows, outlier_summary


def nearest_cluster_for_point(point, cluster_lookup):
    # select the nearest learned anchor for one projected point.
    if not cluster_lookup:
        return None
    px, py = point
    return min(
        cluster_lookup,
        key=lambda label: math.hypot(px - cluster_lookup[label]["anchor_x"], py - cluster_lookup[label]["anchor_y"]),
    )


# integration pipeline
# this is the main v2_02 entry point used by the notebook and CLI.
def run_v2_object_integration(args, dbscan_params_by_label=None, dbscan_general_rule=None, dbscan_label_rules_by_label=None):
    # run same-frame dedupe, DBSCAN anchor learning, and output export.
    v2_config.ensure_v2_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing v2 projected CSV: {args.input_csv}")
    rows = [row for row in load_rows(args.input_csv) if str(row.get("label") or "").strip()]
    deduped = dedupe_same_frame(
        rows,
        iou_threshold=args.frame_iou_threshold,
    )
    integrated_rows, registry_rows, outlier_summary = build_registry_and_assign_rows(
        deduped,
        dbscan_params_by_label=dbscan_params_by_label,
        dbscan_general_rule=dbscan_general_rule,
        dbscan_label_rules_by_label=dbscan_label_rules_by_label,
        track_person=args.track_person,
    )

    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = ["object_id", "anchor_x", "anchor_y"]
    output_fields = base_fields + [field for field in extra_fields if field not in base_fields]
    registry_fields = [
        "object_id",
        "label",
        "anchor_x",
        "anchor_y",
        "observations",
        "first_seen",
        "last_seen",
        "dbscan_eps",
        "dbscan_min_samples",
    ]
    write_csv(args.output_csv, integrated_rows, output_fields)
    write_csv(args.registry_csv, registry_rows, registry_fields)
    save_layout_preview(integrated_rows, registry_rows, args.layout_preview_png)
    save_outlier_summary(outlier_summary, args.outlier_summary_png)
    return rows, deduped, integrated_rows, registry_rows, outlier_summary


# visualization outputs
# saved figures support manual review of learned anchors and DBSCAN noise.
def save_layout_preview(integrated_rows, registry_rows, output_path):
    # draw projected detections and learned anchors on the room layout.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    if v2_config.LAYOUT_IMAGE.exists():
        image = plt.imread(v2_config.LAYOUT_IMAGE)
        ax.imshow(image, extent=[0, v2_config.MAP_WIDTH, 0, v2_config.MAP_HEIGHT], origin="upper", alpha=0.55)

    rows_df = pd.DataFrame(integrated_rows)
    if not rows_df.empty:
        rows_df = rows_df[rows_df["label"].astype(str).str.lower() != v2_config.PERSON_LABEL].copy()
        rows_df["projected_x"] = pd.to_numeric(rows_df["projected_x"], errors="coerce")
        rows_df["projected_y"] = pd.to_numeric(rows_df["projected_y"], errors="coerce")
        for label, part in rows_df.dropna(subset=["projected_x", "projected_y"]).groupby("label"):
            ax.scatter(part["projected_x"], part["projected_y"], s=5, alpha=0.12, label=f"{label} points")

    registry_df = pd.DataFrame(registry_rows)
    if not registry_df.empty:
        registry_df["anchor_x"] = pd.to_numeric(registry_df["anchor_x"], errors="coerce")
        registry_df["anchor_y"] = pd.to_numeric(registry_df["anchor_y"], errors="coerce")
        for _, row in registry_df.dropna(subset=["anchor_x", "anchor_y"]).iterrows():
            ax.scatter(row["anchor_x"], row["anchor_y"], marker="*", s=150, color="black", edgecolor="white", linewidth=0.7)
            ax.text(row["anchor_x"] + 0.05, row["anchor_y"] + 0.05, row["object_id"], fontsize=8)

    ax.set_xlim(0, v2_config.MAP_WIDTH)
    ax.set_ylim(0, v2_config.MAP_HEIGHT)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("v2 DBSCAN anchors from projected detections")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_outlier_summary(outlier_summary, output_path):
    # show how many DBSCAN noise rows each label produced before nearest-anchor reassignment.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(outlier_summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not summary_df.empty:
        summary_df = summary_df.sort_values("dbscan_noise_rows", ascending=False)
        ax.bar(summary_df["label"], summary_df["dbscan_noise_rows"], color=["#d62728", "#ff7f0e", "#f1c40f", "#2ca02c", "#1f77b4"][: len(summary_df)])
    ax.set_title("v2 DBSCAN outlier rows by label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Noise rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    # run object integration from the command line with default v2 paths.
    args = parse_args()
    rows, deduped, integrated_rows, registry_rows, outlier_summary = run_v2_object_integration(args)
    print(f"Input rows: {len(rows)}")
    print(f"After same-frame dedupe: {len(deduped)}")
    print(f"Integrated rows: {len(integrated_rows)}")
    print(f"Static anchors: {len(registry_rows)}")
    print(pd.DataFrame(outlier_summary))


if __name__ == "__main__":
    main()
