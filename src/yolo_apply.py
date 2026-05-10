import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    PREPROCESSING_SUMMARY_CSV,
    YOLO_ANNOTATED_DIR,
    YOLO_DETECTIONS_CSV,
    YOLO_MODEL_PATH,
    ensure_output_dirs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained YOLO detection on preprocessed images.")
    parser.add_argument("--preprocessing-summary-csv", type=Path, default=PREPROCESSING_SUMMARY_CSV)
    parser.add_argument("--model-path", type=Path, default=YOLO_MODEL_PATH)
    parser.add_argument("--output-csv", type=Path, default=YOLO_DETECTIONS_CSV)
    parser.add_argument("--annotated-dir", type=Path, default=YOLO_ANNOTATED_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_preprocessing_rows(path: Path, max_images: int | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessing summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if Path(row.get("preprocessed_image_path", "")).exists()]
    if max_images is not None:
        rows = rows[:max_images]
    return rows


def class_name_from_result(result, class_id):
    names = result.names
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return names[class_id]


def run_detection(args):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing 'ultralytics'. Run this script from the YOLO Python environment."
        ) from exc

    ensure_output_dirs()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing YOLO model: {args.model_path}")

    preprocessed_rows = load_preprocessing_rows(args.preprocessing_summary_csv, args.max_images)
    if not preprocessed_rows:
        raise ValueError("No preprocessed images found. Run src/preprocessing.py first.")

    model = YOLO(str(args.model_path))
    args.annotated_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    for prep_row in tqdm(preprocessed_rows, desc="YOLO apply", unit="image"):
        preprocessed_image_path = Path(prep_row["preprocessed_image_path"])
        image_name = prep_row["image_name"]
        with Image.open(preprocessed_image_path) as image:
            processed_np = np.array(image.convert("RGB"))

        result = model(processed_np, conf=args.conf, verbose=False)[0]
        annotated_path = args.annotated_dir / f"{Path(image_name).stem}_pred.jpg"
        Image.fromarray(result.plot()).save(annotated_path)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            output_rows.append(
                {
                    **prep_row,
                    "annotated_image_path": str(annotated_path),
                    "detection_index": 0,
                    "label": "",
                    "confidence": "",
                    "bbox_x0": "",
                    "bbox_y0": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                }
            )
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for detection_index, (bbox, class_id, confidence) in enumerate(
            zip(xyxy, cls_ids, confidences),
            start=1,
        ):
            output_rows.append(
                {
                    **prep_row,
                    "annotated_image_path": str(annotated_path),
                    "detection_index": detection_index,
                    "label": class_name_from_result(result, class_id),
                    "confidence": f"{float(confidence):.6f}",
                    "bbox_x0": f"{float(bbox[0]):.6f}",
                    "bbox_y0": f"{float(bbox[1]):.6f}",
                    "bbox_x1": f"{float(bbox[2]):.6f}",
                    "bbox_y1": f"{float(bbox[3]):.6f}",
                }
            )

    fieldnames = [
        "index",
        "image_name",
        "source_image_path",
        "preprocessed_image_path",
        "temperature_map_path",
        "annotated_image_path",
        "scale_top",
        "scale_bottom",
        "scale_source",
        "scale_ocr_score",
        "normalize_min",
        "normalize_max",
        "width",
        "height",
        "detection_index",
        "label",
        "confidence",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main():
    args = parse_args()
    rows = run_detection(args)
    print(f"Detection rows: {len(rows)}")
    print(f"Detection CSV: {args.output_csv}")
    print(f"Annotated images: {args.annotated_dir}")


if __name__ == "__main__":
    main()
