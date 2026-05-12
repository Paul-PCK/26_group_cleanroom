import argparse
import csv
import os
import gc
from copy import copy
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from config import DAILY_ANIMATION_OUTPUT_DIR, LAYOUT_IMAGE, OBJECT_TIMELINE_CSV, TIMELINE_GIF, ensure_output_dirs


LABEL_MARKERS = {
    "Machine": "s",
    "Screen": "D",
    "Cableduct": "^",
    "Window": "P",
    "Chair": "X",
    "Person": "o",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render timeline map and temperature history GIF.")
    parser.add_argument("--input-csv", type=Path, default=OBJECT_TIMELINE_CSV)
    parser.add_argument("--layout", type=Path, default=LAYOUT_IMAGE)
    parser.add_argument("--output", type=Path, default=TIMELINE_GIF)
    parser.add_argument("--map-width", type=float, default=15.0)
    parser.add_argument("--map-height", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--temperature-column", choices=("temp_mean_c", "temp_max_c"), default="temp_max_c")
    parser.add_argument("--machine-min-observations", type=int, default=2)
    parser.add_argument("--people-min-observations", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Render only one day in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--split-by-day",
        action="store_true",
        help="Render one GIF per day instead of one GIF for the full timeline.",
    )
    parser.add_argument("--daily-output-dir", type=Path, default=DAILY_ANIMATION_OUTPUT_DIR)
    return parser.parse_args()


def parse_timestamp(value: str):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def safe_float(value: str):
    value = str(value).strip()
    return float(value) if value else np.nan


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("timestamp") or "").strip():
                continue
            row["timestamp_dt"] = parse_timestamp(row["timestamp"])
            row["projected_x"] = safe_float(row["projected_x"])
            row["projected_y"] = safe_float(row["projected_y"])
            row["display_x"] = safe_float(row.get("display_x") or row["projected_x"])
            row["display_y"] = safe_float(row.get("display_y") or row["projected_y"])
            row["temp_mean_c"] = safe_float(row["temp_mean_c"])
            row["temp_max_c"] = safe_float(row["temp_max_c"])
            rows.append(row)
    return rows


def filter_rows_by_day(rows, day: str | None):
    if day is None:
        return rows
    return [row for row in rows if row["timestamp_dt"].strftime("%Y-%m-%d") == day]


def available_days(rows):
    return sorted({row["timestamp_dt"].strftime("%Y-%m-%d") for row in rows})


def group_rows_by_frame(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["timestamp_dt"]].append(row)
    timestamps = sorted(grouped)
    return timestamps, grouped


def build_histories(rows, temperature_column):
    histories = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["timestamp_dt"], item["object_id"])):
        temp = row[temperature_column]
        if not np.isnan(temp):
            histories[row["object_id"]].append((row["timestamp_dt"], temp))
    return histories


def eligible_object_ids(rows, people_or_machine, min_observations):
    counts = defaultdict(int)
    for row in rows:
        if row.get("people_or_machine") == people_or_machine:
            counts[row["object_id"]] += 1
    return {object_id for object_id, count in counts.items() if count >= min_observations}


def build_forward_filled_machine_frames(timestamps, frame_rows, eligible_machine_ids):
    last_machine_rows = {}
    filled_frame_rows = {}
    filled_machine_rows = []

    for timestamp in timestamps:
        current_rows = frame_rows[timestamp]
        people_rows = [row for row in current_rows if row.get("people_or_machine") == "person"]
        current_machine_rows = {}
        for row in current_rows:
            if row.get("people_or_machine") != "machine":
                continue
            object_id = row["object_id"]
            if object_id not in eligible_machine_ids:
                continue
            current_row = copy(row)
            current_row["is_forward_filled"] = "false"
            current_machine_rows[object_id] = current_row
            last_machine_rows[object_id] = current_row

        frame_machine_rows = []
        for object_id in sorted(last_machine_rows):
            row = copy(last_machine_rows[object_id])
            if object_id not in current_machine_rows:
                row["timestamp_dt"] = timestamp
                row["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                row["image_name"] = ""
                row["is_forward_filled"] = "true"
            frame_machine_rows.append(row)
            filled_machine_rows.append(row)

        filled_frame_rows[timestamp] = frame_machine_rows + people_rows

    return filled_frame_rows, filled_machine_rows


def assign_object_colors(object_ids, cmap_name):
    object_ids = sorted(object_ids)
    if not object_ids:
        return {}
    cmap = plt.get_cmap(cmap_name, max(20, len(object_ids)))
    return {object_id: cmap(index % cmap.N) for index, object_id in enumerate(object_ids)}


def subset_histories(rows, people_or_machine, min_observations, temperature_column):
    subset = [row for row in rows if row.get("people_or_machine") == people_or_machine]
    counts = defaultdict(int)
    for row in subset:
        counts[row["object_id"]] += 1
    subset = [row for row in subset if counts[row["object_id"]] >= min_observations]
    return subset, build_histories(subset, temperature_column)


def build_animation_from_rows(rows, args):
    rows = [row for row in rows if row.get("merge_role") == "keep"]
    if not rows:
        raise ValueError("No keep rows found for animation.")

    timestamps, frame_rows = group_rows_by_frame(rows)
    if args.max_frames is not None:
        timestamps = timestamps[: args.max_frames]
        keep_timestamps = set(timestamps)
        rows = [row for row in rows if row["timestamp_dt"] in keep_timestamps]
        frame_rows = {timestamp: frame_rows[timestamp] for timestamp in timestamps}
    if not timestamps:
        raise ValueError("No timestamps found for animation.")

    eligible_machine_ids = eligible_object_ids(rows, "machine", args.machine_min_observations)
    frame_rows, filled_machine_rows = build_forward_filled_machine_frames(
        timestamps,
        frame_rows,
        eligible_machine_ids,
    )
    machine_histories = build_histories(filled_machine_rows, args.temperature_column)
    _, people_histories = subset_histories(
        rows, "person", args.people_min_observations, args.temperature_column
    )
    all_temp_values = [
        temp
        for histories in (machine_histories, people_histories)
        for pairs in histories.values()
        for _, temp in pairs
    ]
    if not all_temp_values:
        raise ValueError("No valid temperature values found for animation.")

    temp_min = min(all_temp_values)
    temp_max = max(all_temp_values)
    machine_colors = assign_object_colors(machine_histories.keys(), "tab20")
    people_colors = assign_object_colors(people_histories.keys(), "Set1")
    layout_image = mpimg.imread(args.layout)

    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.1, 1.1])
    ax_map = fig.add_subplot(gs[0])
    ax_machine = fig.add_subplot(gs[1])
    ax_people = fig.add_subplot(gs[2])

    ax_map.imshow(layout_image, extent=[0, args.map_width, 0, args.map_height], origin="upper", alpha=0.95)
    ax_map.set_xlim(0, args.map_width)
    ax_map.set_ylim(0, args.map_height)
    ax_map.set_xlabel("X (meters)")
    ax_map.set_ylabel("Y (meters)")
    ax_map.grid(alpha=0.15)

    for ax, title in (
        (ax_machine, "Machine Temperature History"),
        (ax_people, "People Temperature History"),
    ):
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.set_ylim(temp_min - 0.5, temp_max + 0.5)
        ax.set_ylabel(f"{args.temperature_column} (C)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax_people.set_xlabel("Time")

    def make_line_artists(ax, histories, colors, alpha, width):
        artists = {}
        for object_id, points in histories.items():
            x_values = [timestamp for timestamp, _ in points]
            y_values = [temp for _, temp in points]
            line, = ax.plot([], [], color=colors[object_id], linewidth=width, alpha=alpha, marker="o", markersize=2.5)
            artists[object_id] = {"artist": line, "x": x_values, "y": y_values}
        return artists

    machine_line_artists = make_line_artists(ax_machine, machine_histories, machine_colors, 0.28, 1.6)
    people_line_artists = make_line_artists(ax_people, people_histories, people_colors, 0.28, 1.1)
    machine_time_line = ax_machine.axvline(timestamps[0], color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    people_time_line = ax_people.axvline(timestamps[0], color="black", linestyle="--", linewidth=1.0, alpha=0.8)

    current_map_markers = []
    current_map_texts = []

    def clear_map_artists():
        while current_map_markers:
            current_map_markers.pop().remove()
        while current_map_texts:
            current_map_texts.pop().remove()

    def update_line_artists(artists, current_timestamp):
        active_ids = set()
        for object_id, payload in artists.items():
            count = 0
            while count < len(payload["x"]) and payload["x"][count] <= current_timestamp:
                count += 1
            payload["artist"].set_data(payload["x"][:count], payload["y"][:count])
            payload["artist"].set_alpha(0.28)
            payload["artist"].set_linewidth(1.1)
            if count > 0:
                active_ids.add(object_id)
        for object_id in active_ids:
            artists[object_id]["artist"].set_alpha(0.95)
            artists[object_id]["artist"].set_linewidth(1.9)

    def update(frame_index):
        current_timestamp = timestamps[frame_index]
        current_rows = frame_rows[current_timestamp]
        clear_map_artists()
        update_line_artists(machine_line_artists, current_timestamp)
        update_line_artists(people_line_artists, current_timestamp)
        machine_time_line.set_xdata([current_timestamp, current_timestamp])
        people_time_line.set_xdata([current_timestamp, current_timestamp])
        ax_map.set_title(
            "Projected Objects on 2D Map\n"
            f"Time: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Objects this frame: {len(current_rows)}"
        )

        for row in current_rows:
            px = row["display_x"]
            py = row["display_y"]
            if np.isnan(px) or np.isnan(py):
                continue
            object_id = row["object_id"]
            label = (row.get("canonical_label") or row["label"]) if row["people_or_machine"] == "machine" else row["label"]
            marker = LABEL_MARKERS.get(label, "o")
            if row["people_or_machine"] == "machine":
                color = machine_colors.get(object_id, (0.3, 0.3, 0.3, 0.9))
                alpha = 0.55 if row.get("is_forward_filled") == "true" else 0.95
            else:
                color = people_colors.get(object_id, (0.8, 0.2, 0.2, 0.9))
                alpha = 0.95
            current_map_markers.append(
                ax_map.scatter(
                    [px],
                    [py],
                    s=85,
                    c=[color],
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=alpha,
                    zorder=4,
                )
            )
            current_map_texts.append(
                ax_map.text(
                    px + 0.08,
                    py + 0.08,
                    object_id,
                    fontsize=6.3,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.4},
                    zorder=5,
                )
            )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(timestamps), interval=1000 / max(1, args.fps), blit=False, repeat=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    del anim
    gc.collect()
    return args.output


def build_animation(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {args.input_csv}")
    if not args.layout.exists():
        raise FileNotFoundError(f"Missing layout image: {args.layout}")

    rows = filter_rows_by_day(load_rows(args.input_csv), args.day)
    return build_animation_from_rows(rows, args)


def build_daily_animations(args):
    ensure_output_dirs()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {args.input_csv}")
    if not args.layout.exists():
        raise FileNotFoundError(f"Missing layout image: {args.layout}")

    all_rows = load_rows(args.input_csv)
    output_paths = []
    args.daily_output_dir.mkdir(parents=True, exist_ok=True)
    for day in available_days(all_rows):
        day_rows = filter_rows_by_day(all_rows, day)
        day_args = copy(args)
        day_args.day = day
        day_args.output = args.daily_output_dir / f"object_timeline_dual_linechart_{day}.gif"
        print(f"Rendering {day}: {len({row['timestamp_dt'] for row in day_rows})} frames -> {day_args.output}")
        try:
            output_paths.append(build_animation_from_rows(day_rows, day_args))
        except ValueError as exc:
            print(f"Skipping {day}: {exc}")
    return output_paths


def main():
    args = parse_args()
    if args.split_by_day:
        outputs = build_daily_animations(args)
        print(f"Saved {len(outputs)} daily animations to: {args.daily_output_dir}")
    else:
        output = build_animation(args)
        print(f"Saved animation to: {output}")


if __name__ == "__main__":
    main()
