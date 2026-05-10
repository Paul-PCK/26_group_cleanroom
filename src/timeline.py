import argparse
import csv
from pathlib import Path

from config import INTEGRATED_OBJECTS_CSV, OBJECT_TIMELINE_CSV, ensure_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Build object timeline temperature CSV.")
    parser.add_argument("--input-csv", type=Path, default=INTEGRATED_OBJECTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=OBJECT_TIMELINE_CSV)
    return parser.parse_args()


def build_timeline(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing integrated objects CSV: {args.input_csv}")
    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    for row in rows:
        if (row.get("merge_role") or "keep") != "keep":
            continue
        output_rows.append(
            {
                "image_name": row.get("image_name", ""),
                "timestamp": row.get("timestamp", ""),
                "detection_index": row.get("detection_index", ""),
                "label": row.get("label", ""),
                "people_or_machine": row.get("people_or_machine", ""),
                "object_id": row.get("object_id", ""),
                "canonical_label": row.get("canonical_label", ""),
                "merge_role": row.get("merge_role", "keep"),
                "bbox_x0": row.get("bbox_x0", ""),
                "bbox_y0": row.get("bbox_y0", ""),
                "bbox_x1": row.get("bbox_x1", ""),
                "bbox_y1": row.get("bbox_y1", ""),
                "projected_x": row.get("projected_x", ""),
                "projected_y": row.get("projected_y", ""),
                "anchor_x": row.get("anchor_x", ""),
                "anchor_y": row.get("anchor_y", ""),
                "display_x": row.get("display_x", row.get("projected_x", "")),
                "display_y": row.get("display_y", row.get("projected_y", "")),
                "position_source": row.get("position_source", ""),
                "static_cluster_id": row.get("static_cluster_id", ""),
                "clustering_method": row.get("clustering_method", ""),
                "temp_mean_c": row.get("temp_mean_c", ""),
                "temp_max_c": row.get("temp_max_c", ""),
            }
        )

    fieldnames = [
        "image_name",
        "timestamp",
        "detection_index",
        "label",
        "people_or_machine",
        "object_id",
        "canonical_label",
        "merge_role",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "projected_x",
        "projected_y",
        "anchor_x",
        "anchor_y",
        "display_x",
        "display_y",
        "position_source",
        "static_cluster_id",
        "clustering_method",
        "temp_mean_c",
        "temp_max_c",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main():
    args = parse_args()
    rows = build_timeline(args)
    print(f"Timeline rows: {len(rows)}")
    print(f"Timeline CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
