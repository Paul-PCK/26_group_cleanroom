import argparse
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LGBM_FEATURE_IMPORTANCE_CSV,
    LGBM_FEATURE_IMPORTANCE_PNG,
    LGBM_PEOPLE_IMPACT_CSV,
    LGBM_PEOPLE_IMPACT_PNG,
    LGBM_PREDICTION_GIF_DIR,
    LGBM_PREDICTIONS_CSV,
    ensure_output_dirs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create plots and GIFs for LGBM temperature predictions.")
    parser.add_argument("--predictions-csv", type=Path, default=LGBM_PREDICTIONS_CSV)
    parser.add_argument("--feature-importance-csv", type=Path, default=LGBM_FEATURE_IMPORTANCE_CSV)
    parser.add_argument("--people-impact-csv", type=Path, default=LGBM_PEOPLE_IMPACT_CSV)
    parser.add_argument("--feature-importance-png", type=Path, default=LGBM_FEATURE_IMPORTANCE_PNG)
    parser.add_argument("--people-impact-png", type=Path, default=LGBM_PEOPLE_IMPACT_PNG)
    parser.add_argument("--gif-dir", type=Path, default=LGBM_PREDICTION_GIF_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--machine-ids", default="")
    parser.add_argument("--machines-per-gif", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=6)
    return parser.parse_args()


def parse_machine_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def configure_matplotlib_cache():
    cache_root = Path(tempfile.gettempdir()) / "cleanroom_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def load_predictions(path: Path = LGBM_PREDICTIONS_CSV):
    if not path.exists():
        raise FileNotFoundError(f"Missing LGBM predictions CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"LGBM predictions CSV is empty: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
    for column in ("source_temp", "target_temp", "prediction", "error"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["timestamp", "target_timestamp", "source_temp", "target_temp", "prediction"])


def plot_feature_importance(
    feature_importance_csv: Path = LGBM_FEATURE_IMPORTANCE_CSV,
    output_png: Path = LGBM_FEATURE_IMPORTANCE_PNG,
    top_n: int = 25,
):
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    if not feature_importance_csv.exists():
        raise FileNotFoundError(f"Missing feature importance CSV: {feature_importance_csv}")
    importance = pd.read_csv(feature_importance_csv).sort_values("importance", ascending=False).head(top_n)
    if importance.empty:
        raise ValueError(f"Feature importance CSV is empty: {feature_importance_csv}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_df = importance.sort_values("importance", ascending=True)
    fig_height = max(5.0, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4c78a8")
    ax.set_xlabel("LightGBM split importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(plot_df)} feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


def plot_people_impact_metrics(
    people_impact_csv: Path = LGBM_PEOPLE_IMPACT_CSV,
    output_png: Path = LGBM_PEOPLE_IMPACT_PNG,
    metric: str = "rmse",
    split: str = "test",
):
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    if not people_impact_csv.exists():
        raise FileNotFoundError(f"Missing people impact CSV: {people_impact_csv}")
    df = pd.read_csv(people_impact_csv)
    if df.empty:
        raise ValueError(f"People impact CSV is empty: {people_impact_csv}")
    plot_df = df[df["split"] == split].copy()
    if plot_df.empty:
        plot_df = df.copy()
    if metric not in plot_df.columns:
        raise ValueError(f"Metric column not found: {metric}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c78a8" if group == "all_rows" else "#f58518" for group in plot_df["people_group"]]
    bars = ax.bar(plot_df["people_group"], plot_df[metric], color=colors)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("People context group")
    ax.set_title(f"{split} prediction error by people context")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, rows in zip(bars, plot_df["rows"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={int(rows)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


def select_machine_groups(df: pd.DataFrame, machine_ids=None, machines_per_gif: int = 2):
    selected = parse_machine_ids(machine_ids)
    if not selected:
        selected = (
            df.groupby("object_id")
            .size()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
    machines_per_gif = max(1, min(int(machines_per_gif), 2))
    return [selected[index : index + machines_per_gif] for index in range(0, len(selected), machines_per_gif)]


def evenly_sample_timestamps(timestamps, max_frames: int):
    timestamps = list(pd.to_datetime(pd.Series(timestamps)).sort_values().drop_duplicates())
    if max_frames is None or max_frames <= 0 or len(timestamps) <= max_frames:
        return timestamps
    indexes = np.linspace(0, len(timestamps) - 1, max_frames).astype(int)
    return [timestamps[index] for index in indexes]


def build_prediction_gif_for_day(
    predictions: pd.DataFrame,
    day,
    machine_ids,
    output_path: Path,
    fps: int = 6,
    max_frames: int = 120,
):
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    day_value = pd.to_datetime(day).date()
    machine_ids = parse_machine_ids(machine_ids)
    day_df = predictions[
        (predictions["timestamp"].dt.date == day_value)
        & (predictions["object_id"].astype(str).isin(machine_ids))
    ].copy()
    if day_df.empty:
        raise ValueError(f"No prediction rows for day={day_value} and machine_ids={machine_ids}")

    frame_times = evenly_sample_timestamps(day_df["timestamp"], max_frames=max_frames)
    y_min = float(np.nanmin([day_df["source_temp"].min(), day_df["target_temp"].min(), day_df["prediction"].min()]))
    y_max = float(np.nanmax([day_df["source_temp"].max(), day_df["target_temp"].max(), day_df["prediction"].max()]))
    y_pad = max(0.5, (y_max - y_min) * 0.15)
    x_min = day_df["timestamp"].min()
    x_max = day_df["target_timestamp"].max()

    fig_height = 3.8 * len(machine_ids)
    fig, axes = plt.subplots(len(machine_ids), 1, figsize=(11, fig_height), sharex=True)
    if len(machine_ids) == 1:
        axes = [axes]

    machine_data = {}
    for machine_id in machine_ids:
        part = day_df[day_df["object_id"].astype(str) == str(machine_id)].sort_values("timestamp")
        label = part["canonical_label"].dropna().mode()
        label_text = label.iloc[0] if not label.empty else ""
        machine_data[str(machine_id)] = (part, label_text)

    def draw(frame_index):
        current_time = frame_times[frame_index]
        for ax, machine_id in zip(axes, machine_ids):
            ax.clear()
            part, label_text = machine_data[str(machine_id)]
            history = part[part["timestamp"] <= current_time]
            current_rows = part[part["timestamp"] == current_time]
            if current_rows.empty:
                current_rows = part[part["timestamp"] <= current_time].tail(1)
            forecast_history = part[part["timestamp"] <= current_time].sort_values("target_timestamp")

            ax.plot(history["timestamp"], history["source_temp"], color="#222222", linewidth=2.0, label="Known GT history")
            if not current_rows.empty:
                current_row = current_rows.iloc[-1]
                ax.scatter(
                    [current_row["timestamp"]],
                    [current_row["source_temp"]],
                    color="#222222",
                    s=28,
                    zorder=4,
                    label="Current known temp",
                )
            ax.plot(
                forecast_history["target_timestamp"],
                forecast_history["prediction"],
                color="#e45756",
                linewidth=2.0,
                label="Forecast history",
            )
            ax.scatter(
                forecast_history["target_timestamp"],
                forecast_history["prediction"],
                color="#e45756",
                s=16,
                alpha=0.75,
                zorder=4,
            )
            if not current_rows.empty:
                current_row = current_rows.iloc[-1]
                ax.scatter(
                    [current_row["target_timestamp"]],
                    [current_row["prediction"]],
                    color="#e45756",
                    edgecolors="#7f1d1d",
                    s=54,
                    zorder=5,
                    label="Current forecast",
                )
            ax.axvline(current_time, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_ylabel("Temp (C)")
            ax.set_title(f"{machine_id} {label_text}".strip())
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"{day_value} forecast view", y=0.995)
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=len(frame_times), interval=1000 / max(fps, 1), repeat=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def build_daily_prediction_gifs(
    predictions_csv: Path = LGBM_PREDICTIONS_CSV,
    output_dir: Path = LGBM_PREDICTION_GIF_DIR,
    split: str = "test",
    machine_ids=None,
    max_machines: int | None = None,
    machines_per_gif: int = 2,
    fps: int = 6,
    max_frames: int = 120,
):
    ensure_output_dirs()
    predictions = load_predictions(predictions_csv)
    predictions = predictions[predictions["split"] == split].copy()
    if predictions.empty:
        raise ValueError(f"No prediction rows for split={split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for day_value, day_df in predictions.groupby(predictions["timestamp"].dt.date):
        selected_machine_ids = parse_machine_ids(machine_ids)
        if not selected_machine_ids:
            selected_machine_ids = (
                day_df.groupby("object_id")
                .size()
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
        if max_machines is not None:
            selected_machine_ids = selected_machine_ids[: int(max_machines)]
        groups = select_machine_groups(day_df, machine_ids=selected_machine_ids, machines_per_gif=machines_per_gif)
        for group in groups:
            group_name = "_".join(safe_filename(machine_id) for machine_id in group)
            output_path = output_dir / f"{split}_{day_value}_{group_name}.gif"
            outputs.append(
                build_prediction_gif_for_day(
                    predictions=predictions,
                    day=day_value,
                    machine_ids=group,
                    output_path=output_path,
                    fps=fps,
                    max_frames=max_frames,
                )
            )
    return outputs


def main():
    args = parse_args()
    feature_png = plot_feature_importance(args.feature_importance_csv, args.feature_importance_png)
    people_png = plot_people_impact_metrics(args.people_impact_csv, args.people_impact_png)
    gif_paths = build_daily_prediction_gifs(
        predictions_csv=args.predictions_csv,
        output_dir=args.gif_dir,
        split=args.split,
        machine_ids=parse_machine_ids(args.machine_ids),
        machines_per_gif=args.machines_per_gif,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"Feature importance plot: {feature_png}")
    print(f"People impact plot: {people_png}")
    for path in gif_paths:
        print(f"Prediction GIF: {path}")


if __name__ == "__main__":
    main()
