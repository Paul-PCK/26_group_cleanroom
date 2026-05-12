import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LGBM_FEATURE_IMPORTANCE_CSV,
    LGBM_LEARNING_CURVE_PNG,
    LGBM_METRICS_CSV,
    LGBM_MODEL_TXT,
    LGBM_PEOPLE_IMPACT_CSV,
    LGBM_PREDICTIONS_CSV,
    OBJECT_TIMELINE_CSV,
    ensure_output_dirs,
)


# Categorical IDs used by LightGBM.
CATEGORICAL_FEATURES = ["object_id", "canonical_label"]

# Machine location features in the 2D map.
POSITION_FEATURES = ["display_x", "display_y", "anchor_x", "anchor_y"]

# Clock-time features; dataset-relative time is excluded.
TIME_FEATURES = ["hour", "minute", "day_of_week"]

# Nearby-people features for each machine timestamp.
PEOPLE_CONTEXT_FEATURES = [
    "people_count_total",
    "people_count_within_1m",
    "people_count_within_2m",
    "people_count_within_3m",
    "nearest_person_distance",
    "mean_person_distance",
    "nearest_person_dx",
    "nearest_person_dy",
]
# History windows in observation counts, not minutes.
HISTORY_WINDOWS = [1, 3, 5, 10, 15, 20]

# Temperature history features for trend and stability.
LAG_FEATURES = (
    [f"temp_lag_{window}" for window in HISTORY_WINDOWS]
    + [f"temp_roll_mean_{window}" for window in HISTORY_WINDOWS]
    + [f"temp_roll_std_{window}" for window in HISTORY_WINDOWS if window > 1]
    + ["temp_delta_1", "seconds_since_previous"]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM temperature prediction from object timeline CSV.")
    parser.add_argument("--input-csv", type=Path, default=OBJECT_TIMELINE_CSV)
    parser.add_argument("--prediction-csv", type=Path, default=LGBM_PREDICTIONS_CSV)
    parser.add_argument("--feature-importance-csv", type=Path, default=LGBM_FEATURE_IMPORTANCE_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=LGBM_METRICS_CSV)
    parser.add_argument("--people-impact-csv", type=Path, default=LGBM_PEOPLE_IMPACT_CSV)
    parser.add_argument("--learning-curve-png", type=Path, default=LGBM_LEARNING_CURVE_PNG)
    parser.add_argument("--model-path", type=Path, default=LGBM_MODEL_TXT)
    parser.add_argument("--temperature-column", choices=("temp_mean_c", "temp_max_c"), default="temp_mean_c")
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--target-minutes", type=float, default=10.0)
    parser.add_argument("--people-or-machine", default="machine")
    parser.add_argument("--train-dates", default="")
    parser.add_argument("--valid-dates", default="")
    parser.add_argument("--test-dates", default="")
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_date_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [pd.to_datetime(item).date() for item in value if str(item).strip()]
    return [pd.to_datetime(item.strip()).date() for item in str(value).split(",") if item.strip()]


def safe_mode(series):
    values = series.dropna()
    if values.empty:
        return ""
    return values.mode().iloc[0]


def load_timeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Timeline CSV is empty: {path}")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    return df


def aggregate_object_timeline(df: pd.DataFrame, temperature_column: str, people_or_machine: str = "machine"):
    # Convert detections into one object-level row per timestamp.
    work = df.copy()
    work = work[work["people_or_machine"] == people_or_machine].copy()
    work = work[work["merge_role"].fillna("keep") == "keep"].copy()
    numeric_columns = [
        "display_x",
        "display_y",
        "anchor_x",
        "anchor_y",
        "projected_x",
        "projected_y",
        "temp_mean_c",
        "temp_max_c",
    ]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["object_id", "timestamp_dt", temperature_column])

    grouped = work.groupby(["object_id", "timestamp_dt"], as_index=False)
    aggregated = grouped.agg(
        timestamp=("timestamp", "first"),
        canonical_label=("canonical_label", safe_mode),
        clustering_method=("clustering_method", safe_mode),
        static_cluster_id=("static_cluster_id", safe_mode),
        display_x=("display_x", "median"),
        display_y=("display_y", "median"),
        anchor_x=("anchor_x", "median"),
        anchor_y=("anchor_y", "median"),
        projected_x=("projected_x", "median"),
        projected_y=("projected_y", "median"),
        temp_mean_c=("temp_mean_c", "mean"),
        temp_max_c=("temp_max_c", "max"),
        observations_in_frame=("object_id", "size"),
    )
    aggregated = aggregated.sort_values(["object_id", "timestamp_dt"]).reset_index(drop=True)
    return aggregated


def aggregate_people_positions(df: pd.DataFrame):
    # Extract person positions used as machine context.
    people = df.copy()
    people = people[people["people_or_machine"] == "person"].copy()
    people = people[people["merge_role"].fillna("keep") == "keep"].copy()
    for column in ("display_x", "display_y", "projected_x", "projected_y"):
        people[column] = pd.to_numeric(people[column], errors="coerce")

    people["person_x"] = people["display_x"].fillna(people["projected_x"])
    people["person_y"] = people["display_y"].fillna(people["projected_y"])
    people = people.dropna(subset=["timestamp_dt", "person_x", "person_y"])
    if people.empty:
        return people[["timestamp_dt", "person_x", "person_y"]]

    grouped = people.groupby(["object_id", "timestamp_dt"], as_index=False)
    return grouped.agg(
        person_x=("person_x", "median"),
        person_y=("person_y", "median"),
    )


def add_people_context_features(machine_df: pd.DataFrame, raw_df: pd.DataFrame):
    # Add nearby-person counts, distance, and direction per machine row.
    output = machine_df.copy()
    for column in PEOPLE_CONTEXT_FEATURES:
        output[column] = np.nan
    for column in ("people_count_total", "people_count_within_1m", "people_count_within_2m", "people_count_within_3m"):
        output[column] = 0

    people_positions = aggregate_people_positions(raw_df)
    if people_positions.empty:
        return output

    people_by_timestamp = {
        timestamp: group[["person_x", "person_y"]].to_numpy(dtype=float)
        for timestamp, group in people_positions.groupby("timestamp_dt")
    }

    output["machine_x_for_people"] = output["display_x"].fillna(output["projected_x"])
    output["machine_y_for_people"] = output["display_y"].fillna(output["projected_y"])

    for timestamp, row_index in output.groupby("timestamp_dt").groups.items():
        people_xy = people_by_timestamp.get(timestamp)
        if people_xy is None or len(people_xy) == 0:
            continue

        machine_xy = output.loc[row_index, ["machine_x_for_people", "machine_y_for_people"]].to_numpy(dtype=float)
        valid_machine = np.isfinite(machine_xy).all(axis=1)
        if not valid_machine.any():
            continue

        valid_machine_xy = machine_xy[valid_machine]
        distance_vectors = valid_machine_xy[:, None, :] - people_xy[None, :, :]
        distances = np.linalg.norm(distance_vectors, axis=2)
        nearest_index = np.nanargmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(distances)), nearest_index]
        nearest_vectors = distance_vectors[np.arange(len(distance_vectors)), nearest_index]

        valid_output_index = output.index[np.asarray(row_index)[valid_machine]]
        output.loc[valid_output_index, "people_count_total"] = len(people_xy)
        output.loc[valid_output_index, "people_count_within_1m"] = (distances <= 1.0).sum(axis=1)
        output.loc[valid_output_index, "people_count_within_2m"] = (distances <= 2.0).sum(axis=1)
        output.loc[valid_output_index, "people_count_within_3m"] = (distances <= 3.0).sum(axis=1)
        output.loc[valid_output_index, "nearest_person_distance"] = nearest_distance
        output.loc[valid_output_index, "mean_person_distance"] = distances.mean(axis=1)
        output.loc[valid_output_index, "nearest_person_dx"] = nearest_vectors[:, 0]
        output.loc[valid_output_index, "nearest_person_dy"] = nearest_vectors[:, 1]

    output = output.drop(columns=["machine_x_for_people", "machine_y_for_people"])
    return output


def add_time_features(df: pd.DataFrame):
    # Add clock-time columns used by the model.
    output = df.copy()
    first_timestamp = output["timestamp_dt"].min()
    output["hour"] = output["timestamp_dt"].dt.hour
    output["minute"] = output["timestamp_dt"].dt.minute
    output["day_of_week"] = output["timestamp_dt"].dt.dayofweek
    output["minutes_from_start"] = (output["timestamp_dt"] - first_timestamp).dt.total_seconds() / 60.0
    return output


def add_target_by_minutes(output: pd.DataFrame, temperature_column: str, target_minutes: float):
    # Find each machine's future target near the requested minute horizon.
    output["target_temp"] = np.nan
    output["target_timestamp"] = pd.NaT
    target_delta = pd.Timedelta(minutes=target_minutes)

    # Drop targets too far from the requested horizon.
    max_target_delta = pd.Timedelta(minutes=target_minutes * 1.5)

    for _, group in output.groupby("object_id", sort=False):
        group = group.sort_values("timestamp_dt")
        timestamps = group["timestamp_dt"].to_numpy(dtype="datetime64[ns]")
        target_timestamps = (group["timestamp_dt"] + target_delta).to_numpy(dtype="datetime64[ns]")
        target_positions = np.searchsorted(timestamps, target_timestamps, side="left")
        valid = target_positions < len(group)
        if not valid.any():
            continue

        source_index_all = group.index.to_numpy()[valid]
        target_index_all = group.index.to_numpy()[target_positions[valid]]
        actual_delta = output.loc[target_index_all, "timestamp_dt"].to_numpy() - output.loc[source_index_all, "timestamp_dt"].to_numpy()
        within_window = actual_delta <= max_target_delta
        if not within_window.any():
            continue

        source_index = source_index_all[within_window]
        target_index = target_index_all[within_window]
        output.loc[source_index, "target_temp"] = output.loc[target_index, temperature_column].to_numpy()
        output.loc[source_index, "target_timestamp"] = output.loc[target_index, "timestamp_dt"].to_numpy()

    return output


def add_lag_features(df: pd.DataFrame, temperature_column: str, target_horizon: int, target_minutes: float | None = None):
    # Add target, lag, rolling, and delta features per machine.
    output = df.sort_values(["object_id", "timestamp_dt"]).copy()
    grouped = output.groupby("object_id", sort=False)
    if target_minutes is None:
        output["target_temp"] = grouped[temperature_column].shift(-target_horizon)
        output["target_timestamp"] = grouped["timestamp_dt"].shift(-target_horizon)
    else:
        output = add_target_by_minutes(output, temperature_column, target_minutes)
    shifted = grouped[temperature_column].shift(1)
    for window in HISTORY_WINDOWS:
        output[f"temp_lag_{window}"] = grouped[temperature_column].shift(window)
        output[f"temp_roll_mean_{window}"] = (
            shifted.groupby(output["object_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        if window > 1:
            output[f"temp_roll_std_{window}"] = (
                shifted.groupby(output["object_id"])
                .rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )
    output["temp_delta_1"] = output[temperature_column] - output["temp_lag_1"]
    output["seconds_since_previous"] = grouped["timestamp_dt"].diff().dt.total_seconds()
    output["target_minutes_ahead"] = (output["target_timestamp"] - output["timestamp_dt"]).dt.total_seconds() / 60.0
    return output


def build_lgbm_dataset(
    timeline_csv: Path = OBJECT_TIMELINE_CSV,
    temperature_column: str = "temp_mean_c",
    target_horizon: int = 1,
    target_minutes: float | None = 10.0,
    people_or_machine: str = "machine",
):
    # Build the final LightGBM training table and feature list.
    raw_df = load_timeline(timeline_csv)
    object_df = aggregate_object_timeline(raw_df, temperature_column, people_or_machine=people_or_machine)
    if people_or_machine == "machine":
        object_df = add_people_context_features(object_df, raw_df)
    featured = add_time_features(object_df)
    featured = add_lag_features(featured, temperature_column, target_horizon, target_minutes=target_minutes)

    feature_columns = (
        CATEGORICAL_FEATURES
        + POSITION_FEATURES
        + TIME_FEATURES
        + PEOPLE_CONTEXT_FEATURES
        + LAG_FEATURES
        + ["observations_in_frame", "target_minutes_ahead"]
    )
    # Keep rows with target and full requested history.
    model_df = featured.dropna(subset=["target_temp", f"temp_lag_{max(HISTORY_WINDOWS)}"]).copy()
    for column in CATEGORICAL_FEATURES:
        model_df[column] = model_df[column].fillna("").astype("category")
    for column in feature_columns:
        if column not in CATEGORICAL_FEATURES:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    return model_df, feature_columns


def split_by_time(model_df: pd.DataFrame, valid_ratio: float = 0.15, test_ratio: float = 0.15):
    # Split data chronologically by ratio.
    timestamps = np.array(sorted(model_df["timestamp_dt"].unique()))
    if len(timestamps) < 3:
        raise ValueError("Need at least three unique timestamps for train/validation/test split.")
    train_end_index = max(1, int(len(timestamps) * (1.0 - valid_ratio - test_ratio)))
    valid_end_index = max(train_end_index + 1, int(len(timestamps) * (1.0 - test_ratio)))
    train_end = timestamps[train_end_index - 1]
    valid_end = timestamps[valid_end_index - 1]

    train_df = model_df[model_df["timestamp_dt"] <= train_end].copy()
    valid_df = model_df[(model_df["timestamp_dt"] > train_end) & (model_df["timestamp_dt"] <= valid_end)].copy()
    test_df = model_df[model_df["timestamp_dt"] > valid_end].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train, validation, or test set.")
    return train_df, valid_df, test_df


def split_by_dates(model_df: pd.DataFrame, train_dates, valid_dates, test_dates):
    # Split data by explicit train/valid/test dates.
    train_dates = set(parse_date_list(train_dates))
    valid_dates = set(parse_date_list(valid_dates))
    test_dates = set(parse_date_list(test_dates))
    if not train_dates or not valid_dates or not test_dates:
        raise ValueError("train_dates, valid_dates, and test_dates must all be provided for date split.")

    date_values = model_df["timestamp_dt"].dt.date
    train_df = model_df[date_values.isin(train_dates)].copy()
    valid_df = model_df[date_values.isin(valid_dates)].copy()
    test_df = model_df[date_values.isin(test_dates)].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Date split produced an empty train, validation, or test set.")
    return train_df, valid_df, test_df


def train_lgbm_model(train_df, valid_df, feature_columns, random_state=42):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing lightgbm. Install it in the active environment before training.") from exc

    model = lgb.LGBMRegressor(
        objective="regression",

        # Maximum boosting iterations; early stopping may stop earlier.
        n_estimators=1200,

        # Step size for each tree's correction.
        learning_rate=0.03,

        # Tree complexity limit.
        num_leaves=31,

        # Fraction of rows sampled per tree.
        subsample=0.85,

        # Fraction of features sampled per tree.
        colsample_bytree=0.85,

        # Random seed for reproducibility.
        random_state=random_state,

        # Use all available CPU cores.
        n_jobs=-1,
        verbose=-1,
    )
    evals_result = {}
    model.fit(
        train_df[feature_columns],
        train_df["target_temp"],
        eval_set=[
            (train_df[feature_columns], train_df["target_temp"]),
            (valid_df[feature_columns], valid_df["target_temp"]),
        ],
        eval_names=["train", "valid"],
        eval_metric="rmse",
        callbacks=[
            # Record RMSE for the learning curve.
            lgb.record_evaluation(evals_result),

            # Stop when valid RMSE stops improving.
            lgb.early_stopping(80, verbose=False),
        ],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model, evals_result


def regression_metrics(y_true, y_pred):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def predict_with_split(model, train_df, valid_df, test_df, feature_columns, temperature_column):
    # Predict each split and keep source/target timestamps for evaluation.
    outputs = []
    for split_name, split_df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        pred = model.predict(split_df[feature_columns])
        part = split_df[
            [
                "object_id",
                "timestamp",
                "target_timestamp",
                "canonical_label",
                "display_x",
                "display_y",
                temperature_column,
                "target_minutes_ahead",
                *PEOPLE_CONTEXT_FEATURES,
                "target_temp",
            ]
        ].copy()
        part = part.rename(columns={temperature_column: "source_temp"})
        part["split"] = split_name
        part["prediction"] = pred
        part["error"] = part["prediction"] - part["target_temp"]
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def build_people_impact_metrics(predictions):
    # Summarize error by people-nearby groups.
    groups = [
        ("all_rows", pd.Series(True, index=predictions.index)),
        ("no_person_detected", predictions["people_count_total"].fillna(0) == 0),
        ("person_detected", predictions["people_count_total"].fillna(0) > 0),
        ("person_within_1m", predictions["people_count_within_1m"].fillna(0) > 0),
        ("person_within_2m", predictions["people_count_within_2m"].fillna(0) > 0),
        ("person_within_3m", predictions["people_count_within_3m"].fillna(0) > 0),
    ]
    rows = []
    for split_name, split_df in predictions.groupby("split", sort=False):
        for group_name, mask in groups:
            part = split_df[mask.reindex(split_df.index, fill_value=False)]
            if part.empty:
                continue
            metrics = regression_metrics(part["target_temp"], part["prediction"])
            rows.append({"split": split_name, "people_group": group_name, "rows": len(part), **metrics})
    return pd.DataFrame(rows)


def configure_matplotlib_cache():
    cache_root = Path(tempfile.gettempdir()) / "cleanroom_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def save_learning_curve_plot(evals_result, output_path: Path):
    # Plot train/valid RMSE over boosting iterations.
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for split_name, metrics in evals_result.items():
        rmse_values = metrics.get("rmse") or metrics.get("l2") or []
        if not rmse_values:
            continue
        ax.plot(range(1, len(rmse_values) + 1), rmse_values, label=split_name, linewidth=1.8)
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("LGBM learning curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_outputs(model, predictions, feature_columns, metrics_rows, people_impact, args):
    # Save predictions, metrics, feature importance, and model.
    ensure_output_dirs()
    args.prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.prediction_csv, index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.feature_importance_csv, index=False)

    with args.metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mae", "rmse", "r2"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    people_impact.to_csv(args.people_impact_csv, index=False)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_path))
    return importance


def run_lgbm_temperature_pipeline(args):
    # Run the full training and evaluation pipeline.
    model_df, feature_columns = build_lgbm_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        target_horizon=args.target_horizon,
        target_minutes=getattr(args, "target_minutes", None),
        people_or_machine=args.people_or_machine,
    )
    if getattr(args, "train_dates", None) and getattr(args, "valid_dates", None) and getattr(args, "test_dates", None):
        train_df, valid_df, test_df = split_by_dates(model_df, args.train_dates, args.valid_dates, args.test_dates)
    else:
        train_df, valid_df, test_df = split_by_time(model_df, args.valid_ratio, args.test_ratio)
    model, evals_result = train_lgbm_model(train_df, valid_df, feature_columns, random_state=args.random_state)
    predictions = predict_with_split(model, train_df, valid_df, test_df, feature_columns, args.temperature_column)

    metrics_rows = []
    for split_name in ("train", "valid", "test"):
        split_predictions = predictions[predictions["split"] == split_name]
        metrics = regression_metrics(split_predictions["target_temp"], split_predictions["prediction"])
        metrics_rows.append({"split": split_name, **metrics})

    people_impact = build_people_impact_metrics(predictions)
    importance = save_outputs(model, predictions, feature_columns, metrics_rows, people_impact, args)
    learning_curve = save_learning_curve_plot(evals_result, args.learning_curve_png)
    return {
        "model_df": model_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "model": model,
        "predictions": predictions,
        "metrics": metrics_rows,
        "people_impact": people_impact,
        "evals_result": evals_result,
        "learning_curve": learning_curve,
        "feature_importance": importance,
        "feature_columns": feature_columns,
    }


def main():
    args = parse_args()
    result = run_lgbm_temperature_pipeline(args)
    print(f"Rows for modeling: {len(result['model_df'])}")
    print(f"Train/valid/test: {len(result['train_df'])}/{len(result['valid_df'])}/{len(result['test_df'])}")
    print(f"Predictions CSV: {args.prediction_csv}")
    print(f"Feature importance CSV: {args.feature_importance_csv}")
    print(f"Metrics CSV: {args.metrics_csv}")
    print(f"People impact CSV: {args.people_impact_csv}")
    print(f"Learning curve: {args.learning_curve_png}")
    print(f"Model: {args.model_path}")
    for row in result["metrics"]:
        print(f"{row['split']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}")


if __name__ == "__main__":
    main()
