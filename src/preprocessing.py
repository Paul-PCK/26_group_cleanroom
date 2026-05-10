import argparse
import csv
from pathlib import Path

import numpy as np

from config import (
    PREPROCESSED_IMAGES_DIR,
    PREPROCESSED_TEMPERATURE_MAPS_DIR,
    PREPROCESSING_SUMMARY_CSV,
    SCALE_LABELS_CSV,
    THERMAL_IMAGES_DIR,
    ensure_output_dirs,
    list_thermal_images,
)
from scale_bar import (
    DEFAULT_FALLBACK_BOTTOM,
    DEFAULT_FALLBACK_TOP,
    DEFAULT_NORMALIZE_MAX,
    DEFAULT_NORMALIZE_MIN,
    DEFAULT_SCALE_BAR,
    build_scale_digit_classifier,
    estimate_image_temperature_map,
    image_from_normalized,
    normalize_temperature_map,
    transform_to_bbox_space,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw thermal images for YOLO.")
    parser.add_argument("--thermal-dir", type=Path, default=THERMAL_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=PREPROCESSED_IMAGES_DIR)
    parser.add_argument("--temperature-map-dir", type=Path, default=PREPROCESSED_TEMPERATURE_MAPS_DIR)
    parser.add_argument("--summary-csv", type=Path, default=PREPROCESSING_SUMMARY_CSV)
    parser.add_argument("--scale-labels-csv", type=Path, default=SCALE_LABELS_CSV)
    parser.add_argument("--scale-bar", type=int, nargs=4, default=DEFAULT_SCALE_BAR)
    parser.add_argument("--fallback-top", type=float, default=DEFAULT_FALLBACK_TOP)
    parser.add_argument("--fallback-bottom", type=float, default=DEFAULT_FALLBACK_BOTTOM)
    parser.add_argument("--normalize-min", type=float, default=DEFAULT_NORMALIZE_MIN)
    parser.add_argument("--normalize-max", type=float, default=DEFAULT_NORMALIZE_MAX)
    parser.add_argument("--pad-height", type=int, default=640)
    parser.add_argument("--no-rotate-180", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def preprocess_images(args):
    ensure_output_dirs()
    if not args.thermal_dir.exists():
        raise FileNotFoundError(f"Missing thermal image directory: {args.thermal_dir}")

    image_paths = list_thermal_images(args.thermal_dir, args.max_images)
    if not image_paths:
        raise ValueError(f"No thermal images found in: {args.thermal_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temperature_map_dir.mkdir(parents=True, exist_ok=True)
    classifier = build_scale_digit_classifier(args.thermal_dir, args.scale_labels_csv)
    rotate_180 = not args.no_rotate_180

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        temperature_map, scale_info = estimate_image_temperature_map(image_path, classifier, args)
        normalized = normalize_temperature_map(
            temperature_map,
            args.normalize_min,
            args.normalize_max,
        )
        normalized_bbox_space = transform_to_bbox_space(
            normalized,
            rotate_180=rotate_180,
            pad_height=args.pad_height,
            pad_value=0.0,
        )
        temperature_bbox_space = transform_to_bbox_space(
            temperature_map,
            rotate_180=rotate_180,
            pad_height=args.pad_height,
            pad_value=np.nan,
        )

        preprocessed_path = args.output_dir / f"{image_path.stem}.png"
        temperature_map_path = args.temperature_map_dir / f"{image_path.stem}.npy"
        image_from_normalized(normalized_bbox_space).save(preprocessed_path)
        np.save(temperature_map_path, temperature_bbox_space.astype(np.float32))

        rows.append(
            {
                "index": index,
                "image_name": image_path.name,
                "source_image_path": str(image_path),
                "preprocessed_image_path": str(preprocessed_path),
                "temperature_map_path": str(temperature_map_path),
                "scale_top": scale_info["top"],
                "scale_bottom": scale_info["bottom"],
                "scale_source": scale_info["source"],
                "scale_ocr_score": scale_info["ocr_score"],
                "normalize_min": args.normalize_min,
                "normalize_max": args.normalize_max,
                "width": normalized_bbox_space.shape[1],
                "height": normalized_bbox_space.shape[0],
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "image_name",
        "source_image_path",
        "preprocessed_image_path",
        "temperature_map_path",
        "scale_top",
        "scale_bottom",
        "scale_source",
        "scale_ocr_score",
        "normalize_min",
        "normalize_max",
        "width",
        "height",
    ]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    args = parse_args()
    rows = preprocess_images(args)
    print(f"Preprocessed images: {len(rows)}")
    print(f"Images: {args.output_dir}")
    print(f"Temperature maps: {args.temperature_map_dir}")
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
