import argparse
import ast
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

import v2_config
from v2_01_projection import HumanNN, MachineNN

DEFAULT_RANDOM_STATE = 123


# dataset wrapper
# pytorch dataloaders use this class to pair normalized bbox inputs with map targets.
class ProjectionDataset(Dataset):
    # store normalized bbox inputs and 2D room-coordinate targets.
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


# command line interface
# notebook usage passes the same fields through SimpleNamespace.
def parse_args():
    # collect hotspot paths, model output paths, and training settings.
    parser = argparse.ArgumentParser(description="Train v2 projection neural networks from hotspot annotations.")
    parser.add_argument("--machine-hotspot-dir", type=Path, default=v2_config.HOTSPOT_MACHINES_DIR)
    parser.add_argument("--human-hotspot-dir", type=Path, default=v2_config.HOTSPOT_PEOPLE_DIR)
    parser.add_argument("--machine-output", type=Path, default=v2_config.V2_MACHINE_PROJECTION_MODEL_PATH)
    parser.add_argument("--human-output", type=Path, default=v2_config.V2_HUMAN_PROJECTION_MODEL_PATH)
    parser.add_argument("--normalized-factor", type=float, default=v2_config.NORMALIZED_FACTOR)
    parser.add_argument("--machine-epochs", type=int, default=80)
    parser.add_argument("--human-epochs", type=int, default=60)
    parser.add_argument("--machine-batch-size", type=int, default=32)
    parser.add_argument("--human-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--target-y-scale", type=float, default=v2_config.PROJECTION_TARGET_Y_SCALE)
    return parser.parse_args()


# hotspot parsing
# each hotspot line is expected to contain a label, bbox coordinates, and target xy.
def parse_hotspot_entries(folder: Path, branch: str):
    # read hotspot text files and keep only the requested projection branch.
    inputs = []
    targets = []
    for path in sorted(folder.glob("*.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                entry = ast.literal_eval(raw)
                is_person = str(entry[0]).strip().lower() == v2_config.PERSON_LABEL
                if branch == "human" and not is_person:
                    continue
                if branch == "machine" and is_person:
                    continue
                inputs.append(entry[1])
                targets.append(entry[2])
    return inputs, targets


# dataset preparation
# bbox inputs are normalized to match the scale used during projection inference.
def make_dataset(inputs, targets, normalized_factor, target_y_scale=1.0):
    # convert hotspot lists into tensors used by the projection networks.
    if not inputs:
        raise ValueError("No hotspot rows were found for this projection branch.")
    input_tensor = torch.tensor(inputs, dtype=torch.float32) / float(normalized_factor)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    target_tensor[:, 1] = target_tensor[:, 1] * float(target_y_scale)
    return ProjectionDataset(input_tensor, target_tensor)


def set_training_seed(random_state):
    # keep model initialization, numpy operations, and pytorch shuffling reproducible.
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def split_dataset(dataset, val_ratio, random_state):
    # use fixed random states so repeated training runs are comparable.
    indices = list(range(len(dataset)))
    trainval_idx, test_idx = train_test_split(indices, test_size=0.05, random_state=random_state)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_ratio, random_state=random_state)
    return train_idx, val_idx, test_idx


# evaluation
# metrics are reported in room-coordinate units to make projection errors interpretable.
def evaluate_model(model, loader, loss_fn):
    # compute average loss and xy error metrics for one dataset split.
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_predictions = model(batch_inputs)
            losses.append(loss_fn(batch_predictions, batch_targets).item())
            predictions.append(batch_predictions.numpy())
            targets.append(batch_targets.numpy())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    errors = predictions - targets
    distances = np.linalg.norm(errors, axis=1)
    return {
        "loss": float(np.mean(losses)),
        "mae_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_xy_error_m": float(np.mean(distances)),
        "p95_xy_error_m": float(np.quantile(distances, 0.95)),
    }


# training pipeline
# one function handles both human and machine branches while preserving branch settings.
def train_projection_model(
    *,
    branch,
    hotspot_dir,
    model,
    output_path,
    metrics_path,
    loss_plot_path,
    normalized_factor,
    epochs,
    batch_size,
    learning_rate,
    loss_fn,
    target_y_scale,
    weight_decay=0.0,
    val_ratio=0.20,
    random_state=DEFAULT_RANDOM_STATE,
):
    # train one projection network and save weights, metrics, and the loss curve.
    set_training_seed(random_state)
    inputs, targets = parse_hotspot_entries(hotspot_dir, branch)
    dataset = make_dataset(inputs, targets, normalized_factor, target_y_scale=target_y_scale)
    train_idx, val_idx, test_idx = split_dataset(dataset, val_ratio=val_ratio, random_state=random_state)

    data_generator = torch.Generator()
    data_generator.manual_seed(random_state)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=data_generator)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    for _epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_inputs, batch_targets in train_loader:
            # update model weights from one mini-batch.
            optimizer.zero_grad()
            batch_predictions = model(batch_inputs)
            loss = loss_fn(batch_predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_losses.append(float(np.mean(epoch_losses)))
        val_losses.append(evaluate_model(model, val_loader, loss_fn)["loss"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch": branch,
            "normalized_factor": normalized_factor,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "target_y_scale": target_y_scale,
            "weight_decay": weight_decay,
        },
        output_path,
    )

    metrics = {
        "branch": branch,
        "hotspot_dir": str(hotspot_dir),
        "model_path": str(output_path),
        "sample_count": len(dataset),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_y_scale": target_y_scale,
        "weight_decay": weight_decay,
        "train": evaluate_model(model, train_loader, loss_fn),
        "val": evaluate_model(model, val_loader, loss_fn),
        "test": evaluate_model(model, test_loader, loss_fn),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_loss_plot(train_losses, val_losses, branch, loss_plot_path)
    return metrics


def save_loss_plot(train_losses, val_losses, branch, output_path):
    # save a compact loss plot for notebook review after training.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="validation")
    ax.set_title(f"v2 {branch} projection training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# branch-specific entry points
# these wrappers keep the original machine and human training choices explicit.
def train_machine_projection(args):
    # train the non-person projection network with MSE loss.
    return train_projection_model(
        branch="machine",
        hotspot_dir=args.machine_hotspot_dir,
        model=MachineNN(),
        output_path=args.machine_output,
        metrics_path=v2_config.V2_MACHINE_TRAINING_METRICS_JSON,
        loss_plot_path=v2_config.V2_MACHINE_TRAINING_LOSS_PNG,
        normalized_factor=args.normalized_factor,
        epochs=args.machine_epochs,
        batch_size=args.machine_batch_size,
        learning_rate=args.learning_rate,
        loss_fn=nn.MSELoss(),
        target_y_scale=args.target_y_scale,
        val_ratio=0.20,
        random_state=args.random_state,
    )


def train_human_projection(args):
    # train the person projection network with smooth l1 loss.
    return train_projection_model(
        branch="human",
        hotspot_dir=args.human_hotspot_dir,
        model=HumanNN(),
        output_path=args.human_output,
        metrics_path=v2_config.V2_HUMAN_TRAINING_METRICS_JSON,
        loss_plot_path=v2_config.V2_HUMAN_TRAINING_LOSS_PNG,
        normalized_factor=args.normalized_factor,
        epochs=args.human_epochs,
        batch_size=args.human_batch_size,
        learning_rate=args.learning_rate,
        loss_fn=nn.SmoothL1Loss(),
        target_y_scale=args.target_y_scale,
        weight_decay=1e-5,
        val_ratio=0.2105,
        random_state=args.random_state,
    )


def main():
    # train both projection models from the command line with default v2 paths.
    args = parse_args()
    v2_config.ensure_v2_output_dirs()
    machine_metrics = train_machine_projection(args)
    human_metrics = train_human_projection(args)
    print(json.dumps({"machine": machine_metrics, "human": human_metrics}, indent=2))


if __name__ == "__main__":
    main()
