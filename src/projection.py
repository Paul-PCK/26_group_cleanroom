import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    HUMAN_PROJECTION_MODEL_PATH,
    MACHINE_PROJECTION_MODEL_PATH,
    NORMALIZED_FACTOR,
    PERSON_LABEL,
    PROJECTED_DETECTIONS_CSV,
    YOLO_DETECTIONS_CSV,
    ensure_output_dirs,
)


class HumanNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.sigmoid(self.output_layer(x))
        return torch.stack([x[:, 0] * 14.0, x[:, 1] * 8.0], dim=1)


class MachineNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Project YOLO bounding boxes to the 2D room map.")
    parser.add_argument("--input-csv", type=Path, default=YOLO_DETECTIONS_CSV)
    parser.add_argument("--output-csv", type=Path, default=PROJECTED_DETECTIONS_CSV)
    parser.add_argument("--human-model-path", type=Path, default=HUMAN_PROJECTION_MODEL_PATH)
    parser.add_argument("--machine-model-path", type=Path, default=MACHINE_PROJECTION_MODEL_PATH)
    parser.add_argument("--normalized-factor", type=float, default=NORMALIZED_FACTOR)
    return parser.parse_args()


def load_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    return checkpoint.get("model_state_dict", checkpoint)


def load_model(model_class, path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model = model_class()
    model.load_state_dict(load_state_dict(checkpoint))
    model.eval()
    return model


def safe_float(row, key):
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing {key}")
    return float(value)


def projection_model_name(label: str):
    return "human" if label.strip().lower() == PERSON_LABEL else "machine"


def project_row(row, human_model, machine_model, normalized_factor):
    label = (row.get("label") or "").strip()
    model_name = projection_model_name(label)
    model = human_model if model_name == "human" else machine_model
    x0 = safe_float(row, "bbox_x0")
    y0 = safe_float(row, "bbox_y0")
    x1 = safe_float(row, "bbox_x1")
    y1 = safe_float(row, "bbox_y1")
    nn_input = torch.tensor(
        [[x0 / normalized_factor, y0 / normalized_factor, x1 / normalized_factor, y1 / normalized_factor]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        projected_x, projected_y = model(nn_input).squeeze(0).tolist()
    return model_name, float(projected_x), float(projected_y)


def run_projection(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing detection CSV: {args.input_csv}")
    if not args.human_model_path.exists():
        raise FileNotFoundError(f"Missing human projection model: {args.human_model_path}")
    if not args.machine_model_path.exists():
        raise FileNotFoundError(f"Missing machine projection model: {args.machine_model_path}")

    human_model = load_model(HumanNN, args.human_model_path)
    machine_model = load_model(MachineNN, args.machine_model_path)

    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))

    output_rows = []
    for row in tqdm(input_rows, desc="Projection", unit="object"):
        if not (row.get("label") or "").strip():
            continue
        model_name, projected_x, projected_y = project_row(
            row,
            human_model=human_model,
            machine_model=machine_model,
            normalized_factor=args.normalized_factor,
        )
        output_rows.append(
            {
                **row,
                "projected_x": f"{projected_x:.6f}",
                "projected_y": f"{projected_y:.6f}",
                "projection_model": model_name,
            }
        )

    fieldnames = list(input_rows[0].keys()) if input_rows else []
    for field in ("projected_x", "projected_y", "projection_model"):
        if field not in fieldnames:
            fieldnames.append(field)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main():
    args = parse_args()
    rows = run_projection(args)
    print(f"Projected rows: {len(rows)}")
    print(f"Projection CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
