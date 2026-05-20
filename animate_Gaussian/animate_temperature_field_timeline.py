import argparse
import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from tqdm import tqdm

from animate_people_timeline import (
    DEFAULT_LAYOUT,
    DEFAULT_LOOKUP_CSV,
    DEFAULT_MACHINE_DIR,
    DEFAULT_PEOPLE_DIR,
    DEFAULT_SCALE_BAR,
    DEFAULT_TEMP_BOTTOM,
    DEFAULT_TEMP_TOP,
    DEFAULT_THERMAL_DIR,
    attach_temperatures,
    build_frames,
    ensure_timestamp_lookup,
    mean_temperature,
)


DEFAULT_TEMPERATURE_OUTPUT = (
    Path(__file__).resolve().parent / "people_machine_temperature_field_timeline.gif"
)
TEMPERATURE_DISPLAY_MIN = 18.0
TEMPERATURE_DISPLAY_MAX = 30.0
MARKER_SIZE = 120
TEMPERATURE_LABEL_OFFSET_Y = 0.12
TEMPERATURE_LABEL_FONTSIZE = 7
MARKER_SEQUENCE = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "*", "h", "8"]
LABEL_COLORS = [
    "#ffffff",
    "#4dd0e1",
    "#ffd166",
    "#ef476f",
    "#06d6a0",
    "#f78c6b",
    "#c792ea",
    "#95e06c",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a bounded physical temperature field animation."
    )
    parser.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE_DIR)
    parser.add_argument("--machine-dir", type=Path, default=DEFAULT_MACHINE_DIR)
    parser.add_argument("--thermal-dir", type=Path, default=DEFAULT_THERMAL_DIR)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--lookup-csv", type=Path, default=DEFAULT_LOOKUP_CSV)
    parser.add_argument(
        "--temperature-output",
        "--output",
        dest="temperature_output",
        type=Path,
        default=DEFAULT_TEMPERATURE_OUTPUT,
        help="Output path for the bounded physical temperature field animation.",
    )
    parser.add_argument("--map-width", type=float, default=15.0)
    parser.add_argument("--map-height", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--scale-bar",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=DEFAULT_SCALE_BAR,
    )
    parser.add_argument(
        "--temp-top",
        type=float,
        default=DEFAULT_TEMP_TOP,
        help="Fallback top temperature if OCR fails for a frame.",
    )
    parser.add_argument(
        "--temp-bottom",
        type=float,
        default=DEFAULT_TEMP_BOTTOM,
        help="Fallback bottom temperature if OCR fails for a frame.",
    )
    parser.add_argument(
        "--background-temp",
        type=float,
        default=20.0,
        help="Fixed background temperature of the 2D map in Celsius.",
    )
    parser.add_argument(
        "--people-sigma",
        type=float,
        default=0.65,
        # Larger sigma spreads each person heat source over a wider area.
        help="Gaussian sigma in meters for people heat sources.",
    )
    parser.add_argument(
        "--machine-sigma",
        type=float,
        default=0.45,
        # Larger sigma spreads each machine heat source over a wider area.
        help="Gaussian sigma in meters for machine heat sources.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Global opacity of the rendered field overlay.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap name for the temperature field.",
    )
    parser.add_argument(
        "--rebuild-lookup",
        action="store_true",
        help="Rebuild the timestamp lookup from image metadata even if the CSV already exists.",
    )
    return parser.parse_args()


def precompute_grid(layout_path: Path, map_width: float, map_height: float):
    background = mpimg.imread(layout_path)
    height_px, width_px = background.shape[:2]
    x = np.linspace(0.0, map_width, width_px)
    y = np.linspace(0.0, map_height, height_px)
    grid_x, grid_y = np.meshgrid(x, y)
    return background, grid_x, grid_y


def gaussian_weight(grid_x, grid_y, center_x, center_y, sigma):
    # Unnormalized Gaussian weight centered at one heat source.
    # The weight is highest at the object position and fades with distance.
    squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    variance = sigma ** 2
    return np.exp(-squared_distance / (2.0 * variance))


def iter_heat_sources(frame, people_sigma, machine_sigma):
    # Each detected machine/person with a valid temperature becomes one heat source.
    # Machines and people can use different sigma values.
    for entry in frame["machine_entries"]:
        temperature = entry.get("temperature")
        if temperature is None or math.isnan(temperature):
            continue
        yield entry, float(temperature), machine_sigma

    for entry in frame["people_entries"]:
        temperature = entry.get("temperature")
        if temperature is None or math.isnan(temperature):
            continue
        yield entry, float(temperature), people_sigma


def build_bounded_temperature_field(
    frame,
    grid_x,
    grid_y,
    background_temp,
    people_sigma,
    machine_sigma,
):
    # Accumulate all heat sources as Gaussian-weighted temperature deltas from
    # the fixed background temperature.
    weighted_delta = np.zeros_like(grid_x, dtype=np.float32)
    total_weight = np.zeros_like(grid_x, dtype=np.float32)

    for entry, temperature, sigma in iter_heat_sources(frame, people_sigma, machine_sigma):
        # Build one Gaussian influence map around the source coordinate.
        weight = gaussian_weight(
            grid_x,
            grid_y,
            entry["coord"][0],
            entry["coord"][1],
            sigma,
        ).astype(np.float32)
        # Add this source temperature to the field, weighted by distance.
        weighted_delta += (temperature - background_temp) * weight
        total_weight += weight

    # Normalize by total weight so overlapping sources blend instead of simply
    # increasing without bound.
    return background_temp + weighted_delta / np.maximum(total_weight, 1.0)


def build_all_fields(
    frames,
    grid_x,
    grid_y,
    background_temp,
    people_sigma,
    machine_sigma,
    field_builder,
    desc,
):
    fields = []
    for frame in tqdm(frames, desc=desc, unit="frame"):
        fields.append(
            field_builder(
                frame,
                grid_x,
                grid_y,
                background_temp,
                people_sigma,
                machine_sigma,
            )
        )
    return fields


def save_animation(anim, output_path: Path, fps: int, total_frames: int, desc: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("Output must end with .gif or .mp4")

    with tqdm(total=total_frames, desc=desc, unit="frame") as progress:
        anim.save(
            output_path,
            writer=writer,
            progress_callback=lambda current_frame, _: progress.update(
                current_frame + 1 - progress.n
            ),
        )


def build_label_styles(frames):
    labels = []
    for frame in frames:
        for key in ("people_entries", "machine_entries"):
            for entry in frame[key]:
                label = entry["label"]
                if label not in labels:
                    labels.append(label)

    styles = {}
    for index, label in enumerate(labels):
        styles[label] = {
            "marker": MARKER_SEQUENCE[index % len(MARKER_SEQUENCE)],
            "color": LABEL_COLORS[index % len(LABEL_COLORS)],
        }
    return styles


def build_legend_handles(label_styles):
    handles = []
    for label, style in label_styles.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                label=label,
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
        )
    return handles


def build_animation(
    frames,
    fields,
    background_image,
    map_width,
    map_height,
    background_temp,
    alpha,
    title_text,
    colorbar_label,
    norm,
    colorbar_ticks,
    unit_label,
    cmap_name,
):
    cmap = plt.get_cmap(cmap_name)
    label_styles = build_label_styles(frames)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(
        background_image,
        extent=[0, map_width, 0, map_height],
        origin="upper",
        zorder=0,
    )
    field_artist = ax.imshow(
        fields[0],
        extent=[0, map_width, 0, map_height],
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=1,
    )

    label_scatters = {}
    for label, style in label_styles.items():
        label_scatters[label] = ax.scatter(
            [],
            [],
            s=MARKER_SIZE,
            c=style["color"],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            marker=style["marker"],
            alpha=0.9,
        )

    title = ax.set_title("")
    subtitle = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        zorder=4,
    )

    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=build_legend_handles(label_styles), loc="upper right")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(colorbar_label)
    temperature_texts = []

    def update(frame_index):
        nonlocal temperature_texts
        frame = frames[frame_index]
        field = fields[frame_index]
        field_artist.set_data(field)
        field_artist.set_alpha(alpha)

        for text in temperature_texts:
            text.remove()
        temperature_texts = []

        grouped_coords = {label: [] for label in label_styles}
        for key in ("people_entries", "machine_entries"):
            for entry in frame[key]:
                grouped_coords.setdefault(entry["label"], []).append(entry["coord"])

        for label, scatter in label_scatters.items():
            coords = grouped_coords.get(label, [])
            scatter.set_offsets(coords if coords else np.empty((0, 2)))

        for key in ("people_entries", "machine_entries"):
            for entry in frame[key]:
                temperature = entry.get("temperature")
                if temperature is None or math.isnan(temperature):
                    continue

                x_coord, y_coord = entry["coord"]
                text_y = min(y_coord + TEMPERATURE_LABEL_OFFSET_Y, map_height - 0.05)
                temperature_texts.append(
                    ax.text(
                        x_coord,
                        text_y,
                        f"{temperature:.1f}C",
                        ha="center",
                        va="bottom",
                        fontsize=TEMPERATURE_LABEL_FONTSIZE,
                        color="black",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.7,
                            "edgecolor": "none",
                            "pad": 0.15,
                        },
                        zorder=4,
                    )
                )

        people_mean = mean_temperature(frame["people_entries"])
        machine_mean = mean_temperature(frame["machine_entries"])
        field_mean = float(np.mean(field))
        field_peak = float(np.max(field))
        image_name = frame["image_path"].name if frame["image_path"] is not None else "missing"
        timestamp_text = frame["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        scale_top = frame.get("scale_bar_top")
        scale_bottom = frame.get("scale_bar_bottom")
        scale_source = frame.get("scale_bar_source", "n/a")
        scale_text = "n/a"
        if scale_top is not None and scale_bottom is not None:
            scale_text = f"{scale_top:.1f} -> {scale_bottom:.1f} C"

        title.set_text(title_text)
        subtitle.set_text(
            f"Time: {timestamp_text}\n"
            f"Image: {image_name}\n"
            f"Background temp: {background_temp:.1f} C\n"
            f"Scale estimate: {scale_text} ({scale_source})\n"
            f"People mean: {people_mean:.2f} C | Machines mean: {machine_mean:.2f} C\n"
            f"Field mean: {field_mean:.2f} {unit_label} | Field peak: {field_peak:.2f} {unit_label}"
            if people_mean is not None and machine_mean is not None
            else
            f"Time: {timestamp_text}\n"
            f"Image: {image_name}\n"
            f"Background temp: {background_temp:.1f} C\n"
            f"Scale estimate: {scale_text} ({scale_source})\n"
            f"Field mean: {field_mean:.2f} {unit_label} | Field peak: {field_peak:.2f} {unit_label}"
        )

        return field_artist, *label_scatters.values(), *temperature_texts, title, subtitle

    return fig, animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000,
        blit=False,
        repeat=True,
    )


def main():
    args = parse_args()
    timestamp_to_image, lookup_source = ensure_timestamp_lookup(
        args.thermal_dir,
        args.lookup_csv,
        args.rebuild_lookup,
    )
    frames = build_frames(
        args.people_dir,
        args.machine_dir,
        timestamp_to_image,
        max_frames=args.max_frames,
    )
    attach_temperatures(
        frames,
        args.thermal_dir,
        tuple(args.scale_bar),
        args.temp_top,
        args.temp_bottom,
    )

    background_image, grid_x, grid_y = precompute_grid(
        args.layout,
        args.map_width,
        args.map_height,
    )

    bounded_temperature_fields = build_all_fields(
        frames,
        grid_x,
        grid_y,
        args.background_temp,
        args.people_sigma,
        args.machine_sigma,
        build_bounded_temperature_field,
        "Building bounded temperature fields",
    )

    temperature_norm = Normalize(
        vmin=TEMPERATURE_DISPLAY_MIN,
        vmax=TEMPERATURE_DISPLAY_MAX,
        clip=True,
    )
    temperature_ticks = np.arange(
        TEMPERATURE_DISPLAY_MIN,
        TEMPERATURE_DISPLAY_MAX + 0.1,
        1.0,
    )

    temperature_fig, temperature_anim = build_animation(
        frames,
        bounded_temperature_fields,
        background_image,
        args.map_width,
        args.map_height,
        args.background_temp,
        args.alpha,
        "Bounded Gaussian Temperature Field",
        "Temperature Field (C)",
        temperature_norm,
        temperature_ticks,
        "C",
        args.cmap,
    )
    save_animation(
        temperature_anim,
        args.temperature_output,
        args.fps,
        len(frames),
        "Rendering temperature animation",
    )
    plt.close(temperature_fig)

    print(f"Saved bounded temperature animation to: {args.temperature_output}")
    print(f"Frames: {len(frames)}")
    print(f"Timestamp lookup source: {lookup_source}")
    print(f"Background temperature: {args.background_temp:.1f} C")
    print(f"People sigma: {args.people_sigma:.2f} m")
    print(f"Machine sigma: {args.machine_sigma:.2f} m")


if __name__ == "__main__":
    main()
