import argparse
import csv
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import FINAL_TABLE_CSV, PROJECTED_DETECTIONS_CSV, TIMESTAMP_LOOKUP_CSV, ensure_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the final detection, projection, and temperature table."
    )
    parser.add_argument("--input-csv", type=Path, default=PROJECTED_DETECTIONS_CSV)
    parser.add_argument("--output-csv", type=Path, default=FINAL_TABLE_CSV)
    parser.add_argument("--timestamp-lookup-csv", type=Path, default=TIMESTAMP_LOOKUP_CSV)
    return parser.parse_args()


def load_timestamp_lookup(path: Path):
    lookup = {}
    if not path.exists():
        return lookup
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_name = (row.get("original_name") or row.get("image_name") or "").strip()
            timestamp = (row.get("timestamp") or "").strip()
            if image_name and timestamp:
                lookup[image_name] = timestamp
    return lookup


def safe_float(row, key):
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing {key}")
    return float(value)


def clamp_bbox(x0, y0, x1, y1, width, height):
    left = max(0, min(width, int(math.floor(x0))))
    top = max(0, min(height, int(math.floor(y0))))
    right = max(0, min(width, int(math.ceil(x1))))
    bottom = max(0, min(height, int(math.ceil(y1))))
    return left, top, right, bottom


def bbox_temperature_stats(temperature_map, row):
    x0 = safe_float(row, "bbox_x0")
    y0 = safe_float(row, "bbox_y0")
    x1 = safe_float(row, "bbox_x1")
    y1 = safe_float(row, "bbox_y1")
    height, width = temperature_map.shape
    left, top, right, bottom = clamp_bbox(x0, y0, x1, y1, width, height)
    if right <= left or bottom <= top:
        return math.nan, math.nan
    crop = temperature_map[top:bottom, left:right]
    valid = crop[np.isfinite(crop)]
    if valid.size == 0:
        return math.nan, math.nan
    return float(valid.mean()), float(valid.max())


def load_temperature_map(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing temperature map: {path}")
    return np.load(path).astype(np.float32)


def build_final_table(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing projected detections CSV: {args.input_csv}")

    timestamp_lookup = load_timestamp_lookup(args.timestamp_lookup_csv)
    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    temperature_cache = {}
    output_rows = []
    for row in tqdm(rows, desc="Final table", unit="object"):
        image_name = (row.get("image_name") or "").strip()
        temperature_map_path = Path((row.get("temperature_map_path") or "").strip())
        cache_key = str(temperature_map_path)
        if cache_key not in temperature_cache:
            temperature_cache[cache_key] = load_temperature_map(temperature_map_path)
        temp_mean_c, temp_max_c = bbox_temperature_stats(temperature_cache[cache_key], row)
        timestamp = (row.get("timestamp") or "").strip() or timestamp_lookup.get(image_name, "")

        output_rows.append(
            {
                "image_name": image_name,
                "timestamp": timestamp,
                "detection_index": row.get("detection_index", ""),
                "label": (row.get("label") or "").strip(),
                "confidence": row.get("confidence", ""),
                "bbox_x0": row.get("bbox_x0", ""),
                "bbox_y0": row.get("bbox_y0", ""),
                "bbox_x1": row.get("bbox_x1", ""),
                "bbox_y1": row.get("bbox_y1", ""),
                "projected_x": row.get("projected_x", ""),
                "projected_y": row.get("projected_y", ""),
                "projection_model": row.get("projection_model", ""),
                "temp_mean_c": "" if math.isnan(temp_mean_c) else f"{temp_mean_c:.6f}",
                "temp_max_c": "" if math.isnan(temp_max_c) else f"{temp_max_c:.6f}",
                "source_image_path": row.get("source_image_path", ""),
                "preprocessed_image_path": row.get("preprocessed_image_path", ""),
                "temperature_map_path": row.get("temperature_map_path", ""),
                "annotated_image_path": row.get("annotated_image_path", ""),
                "scale_top": row.get("scale_top", ""),
                "scale_bottom": row.get("scale_bottom", ""),
                "scale_source": row.get("scale_source", ""),
            }
        )

    fieldnames = [
        "image_name",
        "timestamp",
        "detection_index",
        "label",
        "confidence",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "projected_x",
        "projected_y",
        "projection_model",
        "temp_mean_c",
        "temp_max_c",
        "source_image_path",
        "preprocessed_image_path",
        "temperature_map_path",
        "annotated_image_path",
        "scale_top",
        "scale_bottom",
        "scale_source",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main():
    args = parse_args()
    rows = build_final_table(args)
    print(f"Final rows: {len(rows)}")
    print(f"Final CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
