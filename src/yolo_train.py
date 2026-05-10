import argparse
from pathlib import Path

from config import YOLO_BASE_MODEL_PATH, YOLO_TRAINING_PROJECT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO from a prepared dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to YOLO data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=YOLO_BASE_MODEL_PATH,
        help="Detection base model. Default is models/yolov8n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=YOLO_TRAINING_PROJECT_DIR)
    parser.add_argument("--name", type=str, default="cleanroom_yolo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def train_yolo(args):
    if not args.data.exists():
        raise FileNotFoundError(f"Missing data yaml: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing YOLO base model: {args.model}")

    from ultralytics import YOLO

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "workers": args.workers,
        "exist_ok": args.exist_ok,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model = YOLO(str(args.model))
    return model.train(**train_kwargs)


def main():
    args = parse_args()
    train_yolo(args)


if __name__ == "__main__":
    main()
