import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from tqdm import tqdm

import v2_config


# model definitions
# the projection models transform normalized bbox coordinates into 2D room coordinates.
class HumanNN(nn.Module):
    # use a compact fully connected NN for person position projection.
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
        # constrain person outputs before scaling to the room coordinate range.
        x = self.backbone(x)
        x = torch.sigmoid(self.output_layer(x))
        return torch.stack([x[:, 0] * 14.0, x[:, 1] * 8.0], dim=1)


class MachineNN(nn.Module):
    # use batch normalization for non-person projection where bbox scales vary more.
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
        # constrain non-person outputs to the configured layout dimensions.
        x = torch.sigmoid(self.model(x))
        return torch.stack(
            [x[:, 0] * v2_config.MAP_WIDTH, x[:, 1] * v2_config.MAP_HEIGHT],
            dim=1,
        )


# command line interface
# notebook cells usually pass these values through SimpleNamespace instead.
def parse_args():
    # collect projection model paths and CSV locations for standalone execution.
    parser = argparse.ArgumentParser(description="Project v2 bounding boxes to the 2D room map.")
    parser.add_argument("--input-csv", type=Path, default=v2_config.V2_STANDARDIZED_BBOX_CSV)
    parser.add_argument("--output-csv", type=Path, default=v2_config.V2_PROJECTED_DETECTIONS_CSV)
    parser.add_argument("--human-model-path", type=Path, default=v2_config.HUMAN_PROJECTION_MODEL_PATH)
    parser.add_argument("--machine-model-path", type=Path, default=v2_config.MACHINE_PROJECTION_MODEL_PATH)
    parser.add_argument("--normalized-factor", type=float, default=v2_config.NORMALIZED_FACTOR)
    return parser.parse_args()


# model loading
# v2 projection models are expected to be saved by v2_00_projection_train.py.
def load_state_dict(checkpoint):
    # extract the pytorch state dict from the v2 checkpoint format.
    return checkpoint["model_state_dict"]


def load_model(model_class, path: Path):
    # restore a trained projection model and switch it to inference mode.
    checkpoint = torch.load(path, map_location="cpu")
    model = model_class()
    model.load_state_dict(load_state_dict(checkpoint))
    model.eval()
    return model


# row-level projection helpers
# these helpers keep CSV parsing separate from NN inference.
def safe_float(row, key):
    # parse a required numeric CSV field and fail early if it is missing.
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing {key}")
    return float(value)


def project_row(row, human_model, machine_model, normalized_factor):
    # normalize one bbox row, run the selected model, and return projected coordinates.
    label = (row.get("label") or "").strip()
    model = human_model if label.lower() == v2_config.PERSON_LABEL else machine_model
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
    return float(projected_x), float(projected_y)


# projection pipeline
# this stage preserves the original detection columns and appends projected_x/y.
def run_projection(args):
    # project all standardized detections into 2D room coordinates.
    v2_config.ensure_v2_output_dirs()
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
    for row in tqdm(input_rows, desc="V2 projection", unit="object"):
        if not (row.get("label") or "").strip():
            continue
        # keep the source detection record intact and append projection fields.
        projected_x, projected_y = project_row(
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
            }
        )

    fieldnames = list(input_rows[0].keys()) if input_rows else []
    for field in ("projected_x", "projected_y"):
        if field not in fieldnames:
            fieldnames.append(field)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # save projected detections for v2_02 object integration.
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main():
    # run projection from the command line with default v2 paths.
    args = parse_args()
    rows = run_projection(args)
    print(f"Projected rows: {len(rows)}")
    print(f"Projection CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
