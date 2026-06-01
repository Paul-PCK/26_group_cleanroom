import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import v2_config


PEOPLE_CONTEXT_COLUMNS = [
    "people_count_total",
    "people_count_within_1m",
    "people_count_within_2m",
    "people_count_within_3m",
    "nearest_person_distance",
    "mean_person_distance",
    "nearest_person_dx",
    "nearest_person_dy",
]


# command line interface
# notebook usage passes the same fields through a small settings object.
def parse_args():
    # collect v2 object integration input and timeline output paths.
    parser = argparse.ArgumentParser(description="Build v2 object-level temperature time series.")
    parser.add_argument("--input-csv", type=Path, default=v2_config.V2_INTEGRATED_OBJECTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=v2_config.V2_OBJECT_TIMELINE_CSV)
    return parser.parse_args()


def load_integrated_objects(path: Path):
    # load v2_02 output and convert timestamp/numeric columns used by time series aggregation.
    if not path.exists():
        raise FileNotFoundError(f"Missing v2 integrated object CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Integrated object CSV is empty: {path}")

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp_dt", "object_id"])
    df["label"] = df["label"].fillna("").astype(str)

    numeric_columns = [
        "detection_index",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "temp_mean_c",
        "temp_max_c",
        "pixels",
        "confidence",
        "projected_x",
        "projected_y",
        "anchor_x",
        "anchor_y",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("anchor_x", "anchor_y"):
        if column not in df.columns:
            df[column] = np.nan
    df["display_x"] = df["projected_x"]
    df["display_y"] = df["projected_y"]
    machine_mask = ~df["label"].str.lower().eq(v2_config.PERSON_LABEL)
    df.loc[machine_mask, "display_x"] = df.loc[machine_mask, "anchor_x"].fillna(df.loc[machine_mask, "projected_x"])
    df.loc[machine_mask, "display_y"] = df.loc[machine_mask, "anchor_y"].fillna(df.loc[machine_mask, "projected_y"])
    return df


def aggregate_object_timestamp_rows(df: pd.DataFrame):
    # keep one representative detection per object_id and timestamp after anchor assignment.
    group_cols = ["object_id", "timestamp_dt"]
    ranked = df.copy()
    ranked["_confidence_rank"] = ranked["confidence"].fillna(-1.0)
    ranked["_pixels_rank"] = ranked["pixels"].fillna(-1.0)

    timeline = (
        ranked.sort_values(
            [*group_cols, "_confidence_rank", "_pixels_rank"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(group_cols, keep="first")
        .drop(columns=["_confidence_rank", "_pixels_rank"])
    )
    timeline["timestamp"] = timeline["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return timeline.sort_values(["timestamp_dt", "object_id"]).reset_index(drop=True)


def aggregate_people_positions(timeline: pd.DataFrame):
    # build person positions by timestamp for nearby-person context features.
    people = timeline[timeline["label"].str.lower().eq(v2_config.PERSON_LABEL)].copy()
    people = people.dropna(subset=["timestamp_dt", "display_x", "display_y"])
    if people.empty:
        return {}
    return {
        timestamp: group[["display_x", "display_y"]].to_numpy(dtype=float)
        for timestamp, group in people.groupby("timestamp_dt")
    }


def add_people_context_features(timeline: pd.DataFrame):
    # add same-timestamp person counts and distances to each static object row.
    output = timeline.copy()
    for column in PEOPLE_CONTEXT_COLUMNS:
        output[column] = -1.0
    for column in ("people_count_total", "people_count_within_1m", "people_count_within_2m", "people_count_within_3m"):
        output[column] = 0

    people_by_timestamp = aggregate_people_positions(output)
    if not people_by_timestamp:
        return output

    machine_mask = ~output["label"].str.lower().eq(v2_config.PERSON_LABEL)
    for index, row in output[machine_mask].iterrows():
        people_xy = people_by_timestamp.get(row["timestamp_dt"])
        if people_xy is None or len(people_xy) == 0:
            continue
        object_xy = np.asarray([row["display_x"], row["display_y"]], dtype=float)
        if not np.isfinite(object_xy).all():
            continue
        deltas = people_xy - object_xy
        distances = np.linalg.norm(deltas, axis=1)
        nearest_index = int(np.argmin(distances))
        output.at[index, "people_count_total"] = int(len(distances))
        output.at[index, "people_count_within_1m"] = int((distances <= 1.0).sum())
        output.at[index, "people_count_within_2m"] = int((distances <= 2.0).sum())
        output.at[index, "people_count_within_3m"] = int((distances <= 3.0).sum())
        output.at[index, "nearest_person_distance"] = float(distances[nearest_index])
        output.at[index, "mean_person_distance"] = float(np.mean(distances))
        output.at[index, "nearest_person_dx"] = float(deltas[nearest_index, 0])
        output.at[index, "nearest_person_dy"] = float(deltas[nearest_index, 1])
    return output


def build_v2_time_series(args):
    # build the v2 timeline CSV used by later forecasting and animation stages.
    v2_config.ensure_v2_output_dirs()
    df = load_integrated_objects(args.input_csv)
    timeline = aggregate_object_timestamp_rows(df)
    timeline = add_people_context_features(timeline)

    output_columns = [
        "timestamp",
        "object_id",
        "label",
        "image_name",
        "detection_index",
        "display_x",
        "display_y",
        "anchor_x",
        "anchor_y",
        "projected_x",
        "projected_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "temp_mean_c",
        "temp_max_c",
        "confidence",
        "pixels",
        *PEOPLE_CONTEXT_COLUMNS,
    ]
    timeline = timeline[["timestamp_dt", *output_columns]].sort_values(["timestamp_dt", "object_id"])
    output_df = timeline[output_columns].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    return output_df


def main():
    # run v2 time-series generation from the command line.
    args = parse_args()
    timeline = build_v2_time_series(args)
    print(f"V2 timeline rows: {len(timeline)}")
    print(f"V2 timeline CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
