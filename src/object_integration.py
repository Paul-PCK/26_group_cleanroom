import argparse
import csv
import math
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    FINAL_TABLE_CSV,
    INTEGRATED_OBJECTS_CSV,
    STATIC_OBJECT_REGISTRY_CSV,
    ensure_output_dirs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Integrate duplicate detections and static objects.")
    parser.add_argument("--input-csv", type=Path, default=FINAL_TABLE_CSV)
    parser.add_argument("--output-csv", type=Path, default=INTEGRATED_OBJECTS_CSV)
    parser.add_argument("--registry-csv", type=Path, default=STATIC_OBJECT_REGISTRY_CSV)
    parser.add_argument("--frame-iou-threshold", type=float, default=0.35)
    parser.add_argument("--frame-projected-distance-threshold", type=float, default=0.50)
    parser.add_argument("--track-distance-threshold", type=float, default=0.60)
    parser.add_argument(
        "--static-clustering",
        choices=("dbscan", "dbscan_by_label", "kmeans", "kmeans_by_label", "gmm", "gmm_by_label"),
        default="dbscan",
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.50)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument(
        "--dbscan-params-by-label",
        default="Cableduct:eps=0.30,min_samples=8;Machine:eps=0.40,min_samples=8;Screen:eps=0.15,min_samples=10;Window:eps=0.20,min_samples=5",
    )
    parser.add_argument("--kmeans-clusters", type=int, default=15)
    parser.add_argument("--kmeans-clusters-by-label", default="Cableduct:7,Machine:6,Screen:2,Window:1")
    parser.add_argument(
        "--kmeans-params-by-label",
        default="Cableduct:n_clusters=7;Machine:n_clusters=6;Screen:n_clusters=2;Window:n_clusters=1",
    )
    parser.add_argument("--kmeans-random-state", type=int, default=42)
    parser.add_argument("--gmm-max-components", type=int, default=20)
    parser.add_argument("--gmm-max-components-by-label", default="Cableduct:7,Machine:6,Screen:2,Window:1")
    parser.add_argument(
        "--gmm-params-by-label",
        default="Cableduct:max_components=7;Machine:max_components=6;Screen:max_components=2;Window:max_components=1",
    )
    parser.add_argument("--gmm-random-state", type=int, default=42)
    parser.add_argument("--track-person", action="store_true")
    return parser.parse_args()


def parse_timestamp(value: str):
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")


def normalized_timestamp(value: str):
    return parse_timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(row, key, default=math.nan):
    value = (row.get(key) or "").strip()
    if value.lower() in {"none", "nan", "null"}:
        return default
    return float(value) if value else default


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_geometry(row):
    return (
        safe_float(row, "bbox_x0", 0.0),
        safe_float(row, "bbox_y0", 0.0),
        safe_float(row, "bbox_x1", 0.0),
        safe_float(row, "bbox_y1", 0.0),
        safe_float(row, "projected_x", 0.0),
        safe_float(row, "projected_y", 0.0),
    )


def bbox_area(row):
    x0, y0, x1, y1, _, _ = row_geometry(row)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(row_a, row_b):
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


def projected_distance(row_a, row_b):
    _, _, _, _, ax, ay = row_geometry(row_a)
    _, _, _, _, bx, by = row_geometry(row_b)
    return math.hypot(ax - bx, ay - by)


def people_or_machine(label: str):
    return "person" if label.lower() == "person" else "machine"


def confidence_value(row):
    return safe_float(row, "confidence", -1.0)


def should_merge_same_frame(row_a, row_b, iou_threshold, projected_threshold):
    if row_a["label"].strip() != row_b["label"].strip():
        return False
    if people_or_machine(row_a["label"]) != people_or_machine(row_b["label"]):
        return False
    return bbox_iou(row_a, row_b) >= iou_threshold or projected_distance(row_a, row_b) <= projected_threshold


def choose_representative(rows):
    ranked = sorted(rows, key=lambda row: (confidence_value(row), bbox_area(row)), reverse=True)
    representative = deepcopy(ranked[0])
    label_counter = Counter(row["label"] for row in rows)
    representative["same_frame_count"] = str(len(rows))
    representative["same_frame_duplicate"] = "true" if len(rows) > 1 else "false"
    representative["same_frame_labels"] = "|".join(sorted(label_counter))
    representative["same_frame_label_counts"] = "|".join(
        f"{label}:{count}" for label, count in sorted(label_counter.items())
    )
    representative["merge_role"] = "keep"
    return representative


def dedupe_same_frame(rows, iou_threshold, projected_threshold):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_name"]].append(row)

    deduped = []
    for image_name in sorted(grouped):
        image_rows = grouped[image_name]
        visited = [False] * len(image_rows)
        for index, row in enumerate(image_rows):
            if visited[index]:
                continue
            component = [row]
            visited[index] = True
            queue = [index]
            while queue:
                current_index = queue.pop()
                current_row = image_rows[current_index]
                for candidate_index, candidate_row in enumerate(image_rows):
                    if visited[candidate_index]:
                        continue
                    if should_merge_same_frame(current_row, candidate_row, iou_threshold, projected_threshold):
                        visited[candidate_index] = True
                        component.append(candidate_row)
                        queue.append(candidate_index)
            deduped.append(choose_representative(component))
    return deduped


def run_sklearn_dbscan(points, eps, min_samples):
    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing scikit-learn. Install it in the active environment before running DBSCAN clustering."
        ) from exc

    if not points:
        return []
    points_array = np.asarray(points, dtype=float)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_array).tolist()


def run_sklearn_kmeans(points, n_clusters, random_state):
    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing scikit-learn. Install it in the active environment before running KMeans clustering."
        ) from exc

    if not points:
        return []
    points_array = np.asarray(points, dtype=float)
    n_clusters = max(1, min(int(n_clusters), len(points_array)))
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return model.fit_predict(points_array).tolist()


def run_sklearn_gmm_bic(points, max_components, random_state):
    try:
        from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing scikit-learn. Install it in the active environment before running GMM clustering."
        ) from exc

    if not points:
        return [], 0, math.nan

    points_array = np.asarray(points, dtype=float)
    max_components = max(1, min(int(max_components), len(points_array)))
    best_model = None
    best_bic = math.inf
    for n_components in range(1, max_components + 1):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
            n_init=3,
        )
        model.fit(points_array)
        bic = float(model.bic(points_array))
        if bic < best_bic:
            best_bic = bic
            best_model = model
    return best_model.predict(points_array).tolist(), best_model.n_components, best_bic


def label_key(label: str):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "object"


def prepare_unsupervised_rows(rows, track_person):
    rows_out = []
    person_counter = 0

    normalized_rows = []
    for row in rows:
        row = deepcopy(row)
        row["timestamp"] = normalized_timestamp(row["timestamp"])
        row["people_or_machine"] = people_or_machine(row["label"])
        normalized_rows.append(row)

    dbscan_rows = []
    for row in normalized_rows:
        if row["people_or_machine"] == "person" and not track_person:
            person_counter += 1
            row["object_id"] = f"person_frame_{person_counter:05d}"
            row["canonical_label"] = row["label"]
            row["static_cluster_id"] = ""
            row["anchor_x"] = row.get("projected_x", "")
            row["anchor_y"] = row.get("projected_y", "")
            row["display_x"] = row.get("projected_x", "")
            row["display_y"] = row.get("projected_y", "")
            row["position_source"] = "frame_projection"
            row["clustering_method"] = "none_person"
            rows_out.append(row)
        else:
            dbscan_rows.append(row)
    return rows_out, dbscan_rows


def cluster_rows_from_labels(
    rows_out,
    clustering_rows,
    points,
    cluster_labels,
    method_name,
    registry_extra=None,
    assign_noise_to_nearest_anchor=False,
):
    registry_rows = []
    noise_counter = 0
    registry_extra = registry_extra or {}

    cluster_to_indices = defaultdict(list)
    for index, cluster_label in enumerate(cluster_labels):
        cluster_to_indices[cluster_label].append(index)

    cluster_anchor = {}
    cluster_object_id = {}
    cluster_canonical_label = {}
    cluster_records = []
    for cluster_label, indices in sorted(cluster_to_indices.items()):
        if cluster_label == -1:
            continue
        xs = [points[index][0] for index in indices]
        ys = [points[index][1] for index in indices]
        anchor_x = float(np.median(xs))
        anchor_y = float(np.median(ys))
        cluster_rows = [clustering_rows[index] for index in indices]
        label_counter = Counter(row["label"] for row in cluster_rows)
        canonical_label = label_counter.most_common(1)[0][0]
        label_counts = "|".join(f"{label}:{count}" for label, count in sorted(label_counter.items()))
        cluster_records.append(
            {
                "cluster_label": cluster_label,
                "indices": indices,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "mean_x": float(np.mean(xs)),
                "mean_y": float(np.mean(ys)),
                "canonical_label": canonical_label,
                "label_counts": label_counts,
                "cluster_rows": cluster_rows,
            }
        )

    label_counters = defaultdict(int)
    cluster_records = sorted(
        cluster_records,
        key=lambda record: (
            label_key(record["canonical_label"]),
            record["anchor_y"],
            record["anchor_x"],
        ),
    )
    for record in cluster_records:
        cluster_label = record["cluster_label"]
        canonical_label = record["canonical_label"]
        label_prefix = label_key(canonical_label)
        label_counters[label_prefix] += 1
        object_id = f"{label_prefix}_{label_counters[label_prefix]}"

        cluster_anchor[cluster_label] = (record["anchor_x"], record["anchor_y"])
        cluster_object_id[cluster_label] = object_id
        cluster_canonical_label[cluster_label] = canonical_label
        registry_rows.append(
            {
                "object_id": object_id,
                "people_or_machine": "machine",
                "canonical_label": canonical_label,
                "static_cluster_id": f"{method_name}_{cluster_label:04d}",
                "label_counts": record["label_counts"],
                "mean_projected_x": f"{record['mean_x']:.6f}",
                "mean_projected_y": f"{record['mean_y']:.6f}",
                "anchor_x": f"{record['anchor_x']:.6f}",
                "anchor_y": f"{record['anchor_y']:.6f}",
                "observations": str(len(record["indices"])),
                "first_seen": min(row["timestamp"] for row in record["cluster_rows"]),
                "last_seen": max(row["timestamp"] for row in record["cluster_rows"]),
                "clustering_method": f"{method_name}_global_median_anchor",
                **registry_extra,
            }
        )

    for row, cluster_label in zip(clustering_rows, cluster_labels):
        row = deepcopy(row)
        if cluster_label == -1:
            row_label = row["label"]
            nearest_cluster_label = None
            nearest_distance = math.inf
            row_x = safe_float(row, "projected_x", math.nan)
            row_y = safe_float(row, "projected_y", math.nan)
            if assign_noise_to_nearest_anchor and math.isfinite(row_x) and math.isfinite(row_y):
                for candidate_label, (anchor_x, anchor_y) in cluster_anchor.items():
                    if cluster_canonical_label[candidate_label] != row_label:
                        continue
                    distance = math.hypot(row_x - anchor_x, row_y - anchor_y)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_cluster_label = candidate_label

            if nearest_cluster_label is not None:
                anchor_x, anchor_y = cluster_anchor[nearest_cluster_label]
                row["object_id"] = cluster_object_id[nearest_cluster_label]
                row["canonical_label"] = cluster_canonical_label[nearest_cluster_label]
                row["static_cluster_id"] = f"{method_name}_{nearest_cluster_label:04d}"
                row["anchor_x"] = f"{anchor_x:.6f}"
                row["anchor_y"] = f"{anchor_y:.6f}"
                row["display_x"] = row["anchor_x"]
                row["display_y"] = row["anchor_y"]
                row["position_source"] = f"{method_name}_nearest_same_label_anchor"
                row["clustering_method"] = method_name
            else:
                noise_counter += 1
                row["object_id"] = f"{method_name}_noise_{noise_counter:05d}"
                row["canonical_label"] = row["label"]
                row["static_cluster_id"] = "noise"
                row["anchor_x"] = row.get("projected_x", "")
                row["anchor_y"] = row.get("projected_y", "")
                row["display_x"] = row.get("projected_x", "")
                row["display_y"] = row.get("projected_y", "")
                row["position_source"] = f"{method_name}_noise_frame_projection"
                row["clustering_method"] = f"{method_name}_noise"
        else:
            anchor_x, anchor_y = cluster_anchor[cluster_label]
            row["object_id"] = cluster_object_id[cluster_label]
            row["canonical_label"] = cluster_canonical_label[cluster_label]
            row["static_cluster_id"] = f"{method_name}_{cluster_label:04d}"
            row["anchor_x"] = f"{anchor_x:.6f}"
            row["anchor_y"] = f"{anchor_y:.6f}"
            row["display_x"] = row["anchor_x"]
            row["display_y"] = row["anchor_y"]
            row["position_source"] = f"{method_name}_global_median_anchor"
            row["clustering_method"] = method_name
        rows_out.append(row)

    rows_out = sorted(
        rows_out,
        key=lambda row: (
            parse_timestamp(row["timestamp"]),
            row["image_name"],
            row["label"],
            safe_float(row, "display_x", 0.0),
            safe_float(row, "display_y", 0.0),
        ),
    )
    return rows_out, registry_rows


def cluster_static_objects_dbscan(rows, dbscan_eps, dbscan_min_samples, track_person):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    points = [(safe_float(row, "projected_x", 0.0), safe_float(row, "projected_y", 0.0)) for row in clustering_rows]
    labels = run_sklearn_dbscan(points, eps=dbscan_eps, min_samples=dbscan_min_samples)
    return cluster_rows_from_labels(
        rows_out,
        clustering_rows,
        points,
        labels,
        "dbscan",
        {
            "dbscan_eps": f"{dbscan_eps:.6f}",
            "dbscan_min_samples": str(dbscan_min_samples),
        },
    )


def cluster_label_groups(rows_out, grouped_rows, label_clusterer):
    all_rows = rows_out
    all_registry = []
    for label in sorted(grouped_rows):
        label_rows = grouped_rows[label]
        points = [(safe_float(row, "projected_x", 0.0), safe_float(row, "projected_y", 0.0)) for row in label_rows]
        cluster_labels, method_name, registry_extra = label_clusterer(label, label_rows, points)
        if not cluster_labels:
            continue
        all_rows, label_registry = cluster_rows_from_labels(
            all_rows,
            label_rows,
            points,
            cluster_labels,
            method_name,
            registry_extra,
            assign_noise_to_nearest_anchor=True,
        )
        all_registry.extend(label_registry)

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            parse_timestamp(row["timestamp"]),
            row["image_name"],
            row["label"],
            safe_float(row, "display_x", 0.0),
            safe_float(row, "display_y", 0.0),
        ),
    )
    return all_rows, all_registry


def group_rows_by_label(rows):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["label"]].append(row)
    return grouped_rows


def parse_label_params_by_label(value):
    if isinstance(value, dict):
        params_by_label = {}
        for label, params in value.items():
            if isinstance(params, dict):
                params_by_label[str(label)] = {str(key): param_value for key, param_value in params.items()}
            else:
                params_by_label[str(label)] = {"value": params}
        return params_by_label

    params_by_label = {}
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label parameter setting: {item}. Use Label:key=value,key=value.")
        label, raw_params = item.split(":", 1)
        label_params = {}
        if "=" not in raw_params:
            label_params["value"] = raw_params.strip()
        else:
            for pair in raw_params.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(f"Invalid parameter pair: {pair}. Use key=value.")
                key, param_value = pair.split("=", 1)
                label_params[key.strip()] = param_value.strip()
        params_by_label[label.strip()] = label_params
    return params_by_label


def param_float(params, key, default):
    return float(params.get(key, default))


def param_int(params, key, default):
    return int(params.get(key, default))


def cluster_static_objects_dbscan_by_label(rows, dbscan_eps, dbscan_min_samples, track_person, dbscan_params_by_label=None):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    grouped_rows = group_rows_by_label(clustering_rows)
    params_by_label = parse_label_params_by_label(dbscan_params_by_label)

    def label_clusterer(label, label_rows, points):
        label_params = params_by_label.get(label, {})
        label_eps = param_float(label_params, "eps", dbscan_eps)
        label_min_samples = param_int(label_params, "min_samples", dbscan_min_samples)
        cluster_labels = run_sklearn_dbscan(points, eps=label_eps, min_samples=label_min_samples)
        return (
            cluster_labels,
            f"dbscan_{label_key(label)}",
            {
                "dbscan_eps": f"{label_eps:.6f}",
                "dbscan_min_samples": str(label_min_samples),
                "dbscan_label": label,
            },
        )

    return cluster_label_groups(rows_out, grouped_rows, label_clusterer)


def cluster_static_objects_kmeans(rows, kmeans_clusters, kmeans_random_state, track_person):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    points = [(safe_float(row, "projected_x", 0.0), safe_float(row, "projected_y", 0.0)) for row in clustering_rows]
    labels = run_sklearn_kmeans(
        points,
        n_clusters=kmeans_clusters,
        random_state=kmeans_random_state,
    )
    actual_clusters = max(1, min(int(kmeans_clusters), len(points))) if points else 0
    return cluster_rows_from_labels(
        rows_out,
        clustering_rows,
        points,
        labels,
        "kmeans",
        {
            "kmeans_clusters": str(actual_clusters),
            "kmeans_random_state": str(kmeans_random_state),
        },
    )


def parse_clusters_by_label(value):
    if isinstance(value, dict):
        return {str(label): int(count) for label, count in value.items()}
    clusters = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label cluster setting: {item}. Use Label:count.")
        label, count = item.split(":", 1)
        clusters[label.strip()] = int(count)
    return clusters


def cluster_static_objects_kmeans_by_label(
    rows,
    kmeans_clusters_by_label,
    kmeans_random_state,
    track_person,
    kmeans_params_by_label=None,
):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    clusters_by_label = parse_clusters_by_label(kmeans_clusters_by_label)
    params_by_label = parse_label_params_by_label(kmeans_params_by_label)
    grouped_rows = group_rows_by_label(clustering_rows)

    def label_clusterer(label, label_rows, points):
        label_params = params_by_label.get(label, {})
        requested_clusters = param_int(label_params, "n_clusters", clusters_by_label.get(label, 0))
        label_random_state = param_int(label_params, "random_state", kmeans_random_state)
        if requested_clusters <= 0:
            return [], f"kmeans_{label_key(label)}", {}
        cluster_labels = run_sklearn_kmeans(
            points,
            n_clusters=requested_clusters,
            random_state=label_random_state,
        )
        actual_clusters = max(1, min(int(requested_clusters), len(points))) if points else 0
        return (
            cluster_labels,
            f"kmeans_{label_key(label)}",
            {
                "kmeans_clusters": str(actual_clusters),
                "kmeans_random_state": str(label_random_state),
                "kmeans_label": label,
            },
        )

    return cluster_label_groups(rows_out, grouped_rows, label_clusterer)


def cluster_static_objects_gmm(rows, gmm_max_components, gmm_random_state, track_person):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    points = [(safe_float(row, "projected_x", 0.0), safe_float(row, "projected_y", 0.0)) for row in clustering_rows]
    labels, selected_components, selected_bic = run_sklearn_gmm_bic(
        points,
        max_components=gmm_max_components,
        random_state=gmm_random_state,
    )
    return cluster_rows_from_labels(
        rows_out,
        clustering_rows,
        points,
        labels,
        "gmm",
        {
            "gmm_selected_components": str(selected_components),
            "gmm_bic": f"{selected_bic:.6f}",
            "gmm_max_components": str(gmm_max_components),
            "gmm_random_state": str(gmm_random_state),
        },
    )


def cluster_static_objects_gmm_by_label(
    rows,
    gmm_max_components_by_label,
    gmm_random_state,
    track_person,
    gmm_params_by_label=None,
):
    rows_out, clustering_rows = prepare_unsupervised_rows(rows, track_person)
    max_components_by_label = parse_clusters_by_label(gmm_max_components_by_label)
    params_by_label = parse_label_params_by_label(gmm_params_by_label)
    grouped_rows = group_rows_by_label(clustering_rows)

    def label_clusterer(label, label_rows, points):
        label_params = params_by_label.get(label, {})
        max_components = param_int(label_params, "max_components", max_components_by_label.get(label, 0))
        label_random_state = param_int(label_params, "random_state", gmm_random_state)
        if max_components <= 0:
            return [], f"gmm_{label_key(label)}", {}
        cluster_labels, selected_components, selected_bic = run_sklearn_gmm_bic(
            points,
            max_components=max_components,
            random_state=label_random_state,
        )
        return (
            cluster_labels,
            f"gmm_{label_key(label)}",
            {
                "gmm_selected_components": str(selected_components),
                "gmm_bic": f"{selected_bic:.6f}",
                "gmm_max_components": str(max_components),
                "gmm_label": label,
                "gmm_random_state": str(label_random_state),
            },
        )

    return cluster_label_groups(rows_out, grouped_rows, label_clusterer)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def integrate_objects(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing final table CSV: {args.input_csv}")
    rows = [row for row in load_rows(args.input_csv) if (row.get("label") or "").strip()]
    deduped = dedupe_same_frame(rows, args.frame_iou_threshold, args.frame_projected_distance_threshold)
    clustering_method = getattr(args, "static_clustering", "dbscan")
    if clustering_method == "dbscan":
        integrated_rows, registry_rows = cluster_static_objects_dbscan(
            deduped,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            track_person=args.track_person,
        )
    elif clustering_method == "dbscan_by_label":
        integrated_rows, registry_rows = cluster_static_objects_dbscan_by_label(
            deduped,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            track_person=args.track_person,
            dbscan_params_by_label=getattr(args, "dbscan_params_by_label", None),
        )
    elif clustering_method == "kmeans":
        integrated_rows, registry_rows = cluster_static_objects_kmeans(
            deduped,
            kmeans_clusters=args.kmeans_clusters,
            kmeans_random_state=args.kmeans_random_state,
            track_person=args.track_person,
        )
    elif clustering_method == "kmeans_by_label":
        integrated_rows, registry_rows = cluster_static_objects_kmeans_by_label(
            deduped,
            kmeans_clusters_by_label=getattr(args, "kmeans_clusters_by_label", ""),
            kmeans_random_state=args.kmeans_random_state,
            track_person=args.track_person,
            kmeans_params_by_label=getattr(args, "kmeans_params_by_label", None),
        )
    elif clustering_method == "gmm":
        integrated_rows, registry_rows = cluster_static_objects_gmm(
            deduped,
            gmm_max_components=args.gmm_max_components,
            gmm_random_state=args.gmm_random_state,
            track_person=args.track_person,
        )
    elif clustering_method == "gmm_by_label":
        integrated_rows, registry_rows = cluster_static_objects_gmm_by_label(
            deduped,
            gmm_max_components_by_label=getattr(args, "gmm_max_components_by_label", ""),
            gmm_random_state=args.gmm_random_state,
            track_person=args.track_person,
            gmm_params_by_label=getattr(args, "gmm_params_by_label", None),
        )
    else:
        raise ValueError(
            f"Unsupported static_clustering: {clustering_method}. "
            "Use 'dbscan', 'dbscan_by_label', 'kmeans', 'kmeans_by_label', 'gmm', or 'gmm_by_label'."
        )

    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "canonical_label",
        "people_or_machine",
        "object_id",
        "same_frame_count",
        "same_frame_duplicate",
        "same_frame_labels",
        "same_frame_label_counts",
        "merge_role",
        "static_cluster_id",
        "anchor_x",
        "anchor_y",
        "display_x",
        "display_y",
        "position_source",
        "clustering_method",
    ]
    output_fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    registry_fieldnames = [
        "object_id",
        "people_or_machine",
        "canonical_label",
        "static_cluster_id",
        "label_counts",
        "mean_projected_x",
        "mean_projected_y",
        "anchor_x",
        "anchor_y",
        "observations",
        "first_seen",
        "last_seen",
        "clustering_method",
        "dbscan_eps",
        "dbscan_label",
        "dbscan_min_samples",
        "kmeans_clusters",
        "kmeans_label",
        "kmeans_random_state",
        "gmm_selected_components",
        "gmm_bic",
        "gmm_max_components",
        "gmm_label",
        "gmm_random_state",
    ]
    write_csv(args.output_csv, integrated_rows, output_fieldnames)
    write_csv(args.registry_csv, registry_rows, registry_fieldnames)
    return rows, deduped, integrated_rows, registry_rows


def main():
    args = parse_args()
    rows, deduped, integrated_rows, registry_rows = integrate_objects(args)
    print(f"Input rows: {len(rows)}")
    print(f"After same-frame dedupe: {len(deduped)}")
    print(f"Integrated rows: {len(integrated_rows)}")
    print(f"Static registry rows: {len(registry_rows)}")
    print(f"Integrated CSV: {args.output_csv}")
    print(f"Registry CSV: {args.registry_csv}")


if __name__ == "__main__":
    main()
