import argparse
import gc
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import v2_config


DEFAULT_INPUT_CSV = (
    v2_config.V2_ROOT_DIR
    / "prediction_model"
    / "predictions"
    / "v2_06_prediction_results.csv"
)
DEFAULT_OUTPUT_DIR = v2_config.V2_ROOT_DIR / "prediction_model" / "prediction_gifs"
LABEL_MARKERS = {
    "Machine": "s",
    "Light": "^",
    "Window": "P",
    "Screen": "D",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render one-day object prediction GIF.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--layout", type=Path, default=v2_config.LAYOUT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", type=str, default="2025-04-29")
    parser.add_argument("--labels", nargs="+", default=["Machine"])
    parser.add_argument("--model", choices=("xgboost", "lgbm"), default="xgboost")
    parser.add_argument("--map-horizon", choices=("10min", "20min"), default="10min")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--map-width", type=float, default=v2_config.MAP_WIDTH)
    parser.add_argument("--map-height", type=float, default=v2_config.MAP_HEIGHT)
    parser.add_argument("--gt-smooth-window", type=int, default=5)
    return parser.parse_args()


def normalized_labels(labels):
    return {str(label).strip().lower() for label in labels if str(label).strip()}


def label_text(labels):
    return "+".join(str(label).strip() for label in labels if str(label).strip())


def label_slug(labels):
    return "_".join(str(label).strip().lower().replace("/", "_") for label in labels if str(label).strip())


def horizon_columns(horizon, model):
    return {
        "timestamp": f"{horizon}_target_timestamp",
        "target": f"{horizon}_target_temp",
        "prediction": f"{model}_{horizon}_prediction",
        "actual_minutes": f"{horizon}_actual_minutes_ahead",
    }


def load_predictions(input_csv, model, labels):
    df = pd.read_csv(input_csv)
    wanted_labels = normalized_labels(labels)
    df = df[df["label"].astype(str).str.lower().isin(wanted_labels)].copy()
    if df.empty:
        raise ValueError(f"No rows found for labels: {sorted(wanted_labels)}")

    parts = []
    for horizon in ("10min", "20min"):
        cols = horizon_columns(horizon, model)
        required = [
            "object_id",
            "label",
            "source_timestamp",
            "source_temp",
            "display_x",
            "display_y",
            cols["timestamp"],
            cols["target"],
            cols["prediction"],
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {horizon}: {missing}")

        part = df[required].copy()
        part = part.rename(
            columns={
                cols["timestamp"]: "target_timestamp",
                cols["target"]: "target_temp",
                cols["prediction"]: "prediction",
            }
        )
        part["horizon"] = horizon
        part["source_timestamp"] = pd.to_datetime(part["source_timestamp"], errors="coerce")
        part["target_timestamp"] = pd.to_datetime(part["target_timestamp"], errors="coerce")
        for column in ("source_temp", "target_temp", "prediction", "display_x", "display_y"):
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(
            subset=["object_id", "source_timestamp", "target_timestamp", "prediction", "display_x", "display_y"]
        )
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)
    if long_df.empty:
        raise ValueError("No usable prediction rows found after parsing horizons.")
    return long_df.sort_values(["object_id", "horizon", "source_timestamp"])


def select_day(df, date):
    day = pd.to_datetime(date).date()
    return df[df["source_timestamp"].dt.date.eq(day)].copy()


def build_current_temperature_lines(day_df, smooth_window):
    current = (
        day_df.dropna(subset=["source_temp"])
        .drop_duplicates(subset=["object_id", "source_timestamp"])
        .sort_values(["object_id", "source_timestamp"])
    )
    current["source_temp_smooth"] = (
        current.groupby("object_id")["source_temp"]
        .rolling(window=smooth_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return current


def build_frame_table(day_df, map_horizon):
    frame_df = day_df[day_df["horizon"].eq(map_horizon)].copy()
    if frame_df.empty:
        raise ValueError(f"No rows found for map horizon {map_horizon}.")
    return frame_df.sort_values(["source_timestamp", "object_id"])


def build_object_positions(day_df):
    positions = (
        day_df
        .dropna(subset=["object_id", "display_x", "display_y"])
        .groupby("object_id", as_index=False)
        .agg(
            label=("label", "first"),
            display_x=("display_x", "median"),
            display_y=("display_y", "median"),
        )
        .sort_values("object_id")
    )
    if positions.empty:
        raise ValueError("No object positions available for the selected day.")
    return positions


def make_output_path(args):
    if args.output is not None:
        return args.output
    return args.output_dir / f"{label_slug(args.labels)}_prediction_{args.date}_{args.model}_{args.map_horizon}.gif"


def current_series(df, object_id):
    part = (
        df[df["object_id"].astype(str).eq(str(object_id))]
        .groupby("source_timestamp", as_index=True)["source_temp_smooth"]
        .median()
        .sort_index()
    )
    return part


def prediction_series(df, object_id):
    part = df[df["object_id"].astype(str).eq(str(object_id))].copy()
    if part.empty:
        return part
    return (
        part.groupby("source_timestamp", as_index=False)
        .agg(
            target_timestamp=("target_timestamp", "first"),
            prediction=("prediction", "median"),
        )
        .sort_values("source_timestamp")
    )


def render_prediction_gif(args):
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {args.input_csv}")
    if not args.layout.exists():
        raise FileNotFoundError(f"Missing layout image: {args.layout}")

    labels_display = label_text(args.labels)
    all_predictions = load_predictions(args.input_csv, args.model, args.labels)
    day_df = select_day(all_predictions, args.date)
    if day_df.empty:
        available = sorted(all_predictions["source_timestamp"].dt.date.dropna().astype(str).unique())
        raise ValueError(f"No {labels_display} prediction rows for {args.date}. Available dates: {available}")

    frame_df = build_frame_table(day_df, args.map_horizon)
    object_positions = build_object_positions(day_df)
    timestamps = sorted(frame_df["source_timestamp"].dropna().unique())
    timestamps = timestamps[:: max(1, args.frame_step)]
    if args.max_frames is not None:
        timestamps = timestamps[: args.max_frames]
    if not timestamps:
        raise ValueError("No animation timestamps available.")

    current_temp_df = build_current_temperature_lines(day_df, args.gt_smooth_window)
    object_ids = object_positions["object_id"].astype(str).tolist()
    layout_img = mpimg.imread(args.layout)

    y_values = pd.concat(
        [
            current_temp_df["source_temp_smooth"],
            day_df["prediction"],
        ],
        ignore_index=True,
    ).dropna()
    if y_values.empty:
        raise ValueError("No temperature values available for plotting.")
    y_min = max(0.0, float(y_values.min()) - 0.8)
    y_max = float(y_values.max()) + 0.8

    map_values = frame_df["prediction"].dropna()
    temp_vmin = float(map_values.min())
    temp_vmax = float(map_values.max())
    if np.isclose(temp_vmin, temp_vmax):
        temp_vmin -= 0.5
        temp_vmax += 0.5

    fig_height = max(9, 5.0 + 1.05 * len(object_ids))
    fig = plt.figure(figsize=(13, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        1 + len(object_ids),
        1,
        height_ratios=[1.6] + [0.45] * len(object_ids),
    )
    ax_map = fig.add_subplot(gs[0])
    time_axes = {}
    shared_x_axis = None
    for axis_index, object_id in enumerate(object_ids, start=1):
        ax = fig.add_subplot(gs[axis_index], sharex=shared_x_axis)
        if shared_x_axis is None:
            shared_x_axis = ax
        time_axes[object_id] = ax

    ax_map.imshow(layout_img, extent=[0, args.map_width, 0, args.map_height], origin="upper", alpha=0.95)
    ax_map.set_xlim(0, args.map_width)
    ax_map.set_ylim(0, args.map_height)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(alpha=0.15)

    for object_id, ax in time_axes.items():
        ax.set_xlim(min(timestamps), max(day_df["target_timestamp"].max(), max(timestamps)))
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(object_id, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    if object_ids:
        time_axes[object_ids[-1]].set_xlabel("Target timestamp")
        plt.setp(time_axes[object_ids[-1]].get_xticklabels(), rotation=30, ha="right")
        for object_id in object_ids[:-1]:
            plt.setp(time_axes[object_id].get_xticklabels(), visible=False)

    line_payloads = []
    horizon_styles = {
        "10min": {"color": "red", "linewidth": 1.2, "alpha": 0.65},
        "20min": {"color": "tab:blue", "linewidth": 1.2, "alpha": 0.65},
    }
    for object_id in object_ids:
        ax = time_axes[object_id]
        object_current = current_series(current_temp_df, object_id)
        if object_current.notna().any():
            line, = ax.plot([], [], color="black", linewidth=1.0, alpha=0.65)
            line_payloads.append(
                {
                    "line": line,
                    "x": object_current.index.to_numpy(),
                    "y": object_current.to_numpy(),
                    "reveal_x": object_current.index.to_numpy(),
                }
            )
        for horizon, style in horizon_styles.items():
            object_part = day_df[
                day_df["object_id"].astype(str).eq(object_id)
                & day_df["horizon"].eq(horizon)
            ].sort_values("target_timestamp")
            if object_part.empty:
                continue
            object_prediction = prediction_series(object_part, object_id)
            line, = ax.plot([], [], **style)
            line_payloads.append(
                {
                    "line": line,
                    "x": object_prediction["target_timestamp"].to_numpy(),
                    "y": object_prediction["prediction"].to_numpy(),
                    "reveal_x": object_prediction["source_timestamp"].to_numpy(),
                }
            )

    time_markers = {
        object_id: ax.axvline(timestamps[0], color="black", linestyle="--", linewidth=0.9, alpha=0.75)
        for object_id, ax in time_axes.items()
    }
    if object_ids:
        time_axes[object_ids[0]].legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.6, label=f"T rolling mean ({args.gt_smooth_window})"),
            Line2D([0], [0], color="red", linewidth=1.6, label="10 min prediction"),
            Line2D([0], [0], color="tab:blue", linewidth=1.6, label="20 min prediction"),
        ],
        loc="upper right",
        )

    current_map_artists = []
    scatter_for_colorbar = ax_map.scatter([], [], c=[], cmap="coolwarm", vmin=temp_vmin, vmax=temp_vmax)
    colorbar = fig.colorbar(scatter_for_colorbar, ax=ax_map, fraction=0.035, pad=0.02)
    colorbar.set_label(f"{args.model} {args.map_horizon} prediction (C)")

    def clear_map_artists():
        while current_map_artists:
            current_map_artists.pop().remove()

    def reveal_line(payload, current_timestamp):
        x = payload["x"]
        y = payload["y"]
        keep = payload["reveal_x"] <= current_timestamp
        payload["line"].set_data(x[keep], y[keep])

    def update(frame_index):
        current_timestamp = timestamps[frame_index]
        clear_map_artists()

        current_predictions = frame_df[frame_df["source_timestamp"].eq(current_timestamp)][
            ["object_id", "prediction"]
        ].copy()
        current_rows = object_positions.merge(current_predictions, on="object_id", how="left")
        ax_map.set_title(
            f"{labels_display} prediction map | {args.date} | {args.model} {args.map_horizon} | "
            f"{pd.Timestamp(current_timestamp).strftime('%H:%M:%S')}"
        )
        fig.suptitle(f"{labels_display} temperature predictions | {args.date} | {args.model}", fontsize=13)

        for payload in line_payloads:
            reveal_line(payload, current_timestamp)
        for marker in time_markers.values():
            marker.set_xdata([current_timestamp, current_timestamp])

        inactive_rows = current_rows[current_rows["prediction"].isna()]
        active_rows = current_rows[current_rows["prediction"].notna()]
        if not inactive_rows.empty:
            for label_name, label_rows in inactive_rows.groupby("label"):
                inactive_scatter = ax_map.scatter(
                    label_rows["display_x"],
                    label_rows["display_y"],
                    c="#b8b8b8",
                    s=105,
                    marker=LABEL_MARKERS.get(label_name, "o"),
                    edgecolors="#666666",
                    linewidths=0.8,
                    alpha=0.7,
                    zorder=3,
                )
                current_map_artists.append(inactive_scatter)
        if not active_rows.empty:
            for label_name, label_rows in active_rows.groupby("label"):
                active_scatter = ax_map.scatter(
                    label_rows["display_x"],
                    label_rows["display_y"],
                    c=label_rows["prediction"],
                    cmap="coolwarm",
                    vmin=temp_vmin,
                    vmax=temp_vmax,
                    s=130,
                    marker=LABEL_MARKERS.get(label_name, "o"),
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=4,
                )
                current_map_artists.append(active_scatter)
        for row in current_rows.itertuples(index=False):
            has_prediction = pd.notna(row.prediction)
            label = f"{row.object_id}\n{row.prediction:.1f} C" if has_prediction else f"{row.object_id}\noff"
            text = ax_map.text(
                row.display_x + 0.08,
                row.display_y + 0.08,
                label,
                fontsize=7,
                color="black" if has_prediction else "#555555",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.4},
                zorder=5,
            )
            current_map_artists.append(text)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(timestamps),
        interval=1000 / max(1, args.fps),
        blit=False,
        repeat=False,
    )

    output_path = make_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    del anim
    gc.collect()
    return output_path, len(timestamps), len(object_ids)


def main():
    args = parse_args()
    output_path, frame_count, object_count = render_prediction_gif(args)
    print(f"Saved GIF: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Objects: {object_count}")


if __name__ == "__main__":
    main()
