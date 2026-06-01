import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import v2_config


# categorical ids used by the prediction models.
CATEGORICAL_FEATURES = ["object_id"]

# machine location features in the 2D map.
POSITION_FEATURES = ["display_x", "display_y"]

# clock-time features; dataset-relative time is excluded.
TIME_FEATURES = ["hour", "minute", "day_of_week"]

# current known temperature at the source timestamp.
CURRENT_TEMP_FEATURES = ["source_temp"]

# nearby-people features for each machine timestamp.
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
# lag windows in observation counts, not minutes.
LAG_WINDOWS = list(range(1, 31))

# rolling windows summarize short and longer history without adding every window size.
ROLLING_WINDOWS = [5, 10, 20, 30]

# maximum allowed delay after the requested forecast horizon.
TARGET_MATCH_TOLERANCE_MINUTES = 2.0

# temperature history features for trend and stability.
LAG_FEATURES = (
    [f"temp_lag_{window}" for window in LAG_WINDOWS]
    + [f"temp_roll_mean_{window}" for window in ROLLING_WINDOWS]
    + [f"temp_roll_std_{window}" for window in ROLLING_WINDOWS]
    + ["temp_delta_1", "seconds_since_previous"]
)


# model features shared by LGBM and XGBoost.
MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + POSITION_FEATURES
    + TIME_FEATURES
    + CURRENT_TEMP_FEATURES
    + PEOPLE_CONTEXT_FEATURES
    + LAG_FEATURES
)


def parse_args():
    # collect model inputs, split settings, and output locations.
    parser = argparse.ArgumentParser(description="Train LightGBM temperature prediction from object timeline CSV.")
    default_dir = v2_config.V2_LGBM_HORIZON_COMPARE_DIR / "cli"
    parser.add_argument("--input-csv", type=Path, default=v2_config.V2_OBJECT_TIMELINE_CSV)
    parser.add_argument("--prediction-csv", type=Path, default=default_dir / "lgbm_temperature_predictions.csv")
    parser.add_argument("--feature-importance-csv", type=Path, default=default_dir / "lgbm_feature_importance.csv")
    parser.add_argument("--learning-curve-csv", type=Path, default=default_dir / "lgbm_learning_curve.csv")
    parser.add_argument("--temperature-column", choices=("temp_mean_c", "temp_max_c"), default="temp_mean_c")
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--target-minutes", type=float, default=10.0)
    parser.add_argument("--object-role", default="machine", choices=("machine", "person"))
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
    # load timestamped object temperatures and parse timestamps.
    if not path.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Timeline CSV is empty: {path}")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["label"] = df["label"].fillna("").astype(str)
    return df


def aggregate_object_timeline(df: pd.DataFrame, temperature_column: str, object_role: str = "machine"):
    # convert detections into one object-level row per timestamp.
    work = df.copy()
    person_mask = work["label"].str.lower().eq(v2_config.PERSON_LABEL)
    work = work[person_mask if object_role == "person" else ~person_mask].copy()
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
        label=("label", safe_mode),
        display_x=("display_x", "median"),
        display_y=("display_y", "median"),
        anchor_x=("anchor_x", "median"),
        anchor_y=("anchor_y", "median"),
        projected_x=("projected_x", "median"),
        projected_y=("projected_y", "median"),
        temp_mean_c=("temp_mean_c", "mean"),
        temp_max_c=("temp_max_c", "max"),
    )
    aggregated = aggregated.sort_values(["object_id", "timestamp_dt"]).reset_index(drop=True)
    return aggregated


def aggregate_people_positions(df: pd.DataFrame):
    # extract person positions used as machine context.
    people = df.copy()
    people = people[people["label"].str.lower().eq(v2_config.PERSON_LABEL)].copy()
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
    # add nearby-person counts, distance, and direction per machine row.
    output = machine_df.copy()
    for column in PEOPLE_CONTEXT_FEATURES:
        output[column] = -1.0
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
    # add clock-time columns used by the model.
    output = df.copy()
    output["hour"] = output["timestamp_dt"].dt.hour
    output["minute"] = output["timestamp_dt"].dt.minute
    output["day_of_week"] = output["timestamp_dt"].dt.dayofweek
    return output


def add_target_by_minutes(output: pd.DataFrame, temperature_column: str, target_minutes: float):
    # find each machine's future target near the requested minute horizon, resetting each day.
    output["target_temp"] = np.nan
    output["target_timestamp"] = pd.NaT
    if "_sequence_date" not in output.columns:
        output["_sequence_date"] = output["timestamp_dt"].dt.date
    target_delta = pd.Timedelta(minutes=target_minutes)

    # drop targets that are not close to the requested horizon.
    max_target_delta = pd.Timedelta(minutes=target_minutes + TARGET_MATCH_TOLERANCE_MINUTES)

    for _, group in output.groupby(["object_id", "_sequence_date"], sort=False):
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
    # add target, lag, rolling, and delta features per machine, resetting each day.
    output = df.sort_values(["object_id", "timestamp_dt"]).copy()
    output["_sequence_date"] = output["timestamp_dt"].dt.date
    grouped = output.groupby(["object_id", "_sequence_date"], sort=False)
    output["source_temp"] = output[temperature_column]
    if target_minutes is None:
        output["target_temp"] = grouped[temperature_column].shift(-target_horizon)
        output["target_timestamp"] = grouped["timestamp_dt"].shift(-target_horizon)
    else:
        output = add_target_by_minutes(output, temperature_column, target_minutes)
    shifted = grouped[temperature_column].shift(1)
    for window in LAG_WINDOWS:
        output[f"temp_lag_{window}"] = grouped[temperature_column].shift(window)
    for window in ROLLING_WINDOWS:
        output[f"temp_roll_mean_{window}"] = (
            shifted.groupby([output["object_id"], output["_sequence_date"]])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        output[f"temp_roll_std_{window}"] = (
            shifted.groupby([output["object_id"], output["_sequence_date"]])
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )
    output["temp_delta_1"] = output[temperature_column] - output["temp_lag_1"]
    output["seconds_since_previous"] = grouped["timestamp_dt"].diff().dt.total_seconds()
    output["target_minutes_ahead"] = (output["target_timestamp"] - output["timestamp_dt"]).dt.total_seconds() / 60.0
    return output


def prepare_model_features(model_df: pd.DataFrame, feature_columns):
    # keep categorical and numeric feature dtypes consistent across models.
    output = model_df.copy()
    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].fillna("").astype("category")
    for column in feature_columns:
        if column not in CATEGORICAL_FEATURES:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_model_dataset(
    timeline_csv: Path = v2_config.V2_OBJECT_TIMELINE_CSV,
    temperature_column: str = "temp_mean_c",
    target_horizon: int = 1,
    target_minutes: float | None = 10.0,
    object_role: str = "machine",
):
    # build the shared supervised-learning table and feature list.
    raw_df = load_timeline(timeline_csv)
    object_df = aggregate_object_timeline(raw_df, temperature_column, object_role=object_role)
    if object_role == "machine":
        object_df = add_people_context_features(object_df, raw_df)
    featured = add_time_features(object_df)
    featured = add_lag_features(featured, temperature_column, target_horizon, target_minutes=target_minutes)

    feature_columns = list(MODEL_FEATURES)
    # keep rows with target and full requested history.
    model_df = featured.dropna(subset=["target_temp", f"temp_lag_{max(LAG_WINDOWS)}"]).copy()
    model_df = prepare_model_features(model_df, feature_columns)
    return model_df, feature_columns


def build_lgbm_dataset(
    timeline_csv: Path = v2_config.V2_OBJECT_TIMELINE_CSV,
    temperature_column: str = "temp_mean_c",
    target_horizon: int = 1,
    target_minutes: float | None = 10.0,
    object_role: str = "machine",
):
    # keep the existing LGBM entry point while using the shared feature builder.
    return build_model_dataset(
        timeline_csv=timeline_csv,
        temperature_column=temperature_column,
        target_horizon=target_horizon,
        target_minutes=target_minutes,
        object_role=object_role,
    )


def split_by_time(model_df: pd.DataFrame, valid_ratio: float = 0.15, test_ratio: float = 0.15):
    # split data chronologically by ratio.
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
    # split data by explicit train/valid/test dates.
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


def train_lgbm_model(train_df, valid_df, feature_columns, random_state=42, model_params=None):
    # train one LightGBM regressor with optional tuned parameters.
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing lightgbm. Install it in the active environment before training.") from exc

    params = {
        "objective": "regression",

        # maximum boosting iterations; early stopping may stop earlier.
        "n_estimators": 1200,

        # step size for each tree's correction.
        "learning_rate": 0.03,

        # tree complexity limit.
        "num_leaves": 31,

        # fraction of rows sampled per tree.
        "subsample": 0.85,

        # fraction of features sampled per tree.
        "colsample_bytree": 0.85,

        # random seed for reproducibility.
        "random_state": random_state,

        # use all available CPU cores.
        "n_jobs": -1,
        "verbose": -1,
    }
    if model_params:
        params.update(model_params)
    params["objective"] = "regression"
    params["random_state"] = random_state
    params["n_jobs"] = -1
    params["verbose"] = -1

    model = lgb.LGBMRegressor(**params)
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
            # record RMSE for the learning curve.
            lgb.record_evaluation(evals_result),

            # stop when valid RMSE stops improving.
            lgb.early_stopping(80, verbose=False),
        ],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model, evals_result


def regression_metrics(y_true, y_pred):
    # compute standard regression metrics from prediction residuals.
    residual = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def predict_with_split(model, train_df, valid_df, test_df, feature_columns, temperature_column):
    # predict each split and keep source/target timestamps for evaluation.
    outputs = []
    for split_name, split_df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        pred = model.predict(split_df[feature_columns])
        part = split_df[
            [
                "object_id",
                "timestamp",
                "target_timestamp",
                "label",
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


def save_learning_curve_csv(evals_result, output_path: Path):
    # save train/valid RMSE over boosting iterations for downstream comparison.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, metrics in evals_result.items():
        rmse_values = metrics.get("rmse") or metrics.get("l2") or []
        for iteration, rmse in enumerate(rmse_values, start=1):
            rows.append({"iteration": iteration, "split": split_name, "rmse": float(rmse)})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def save_outputs(model, feature_columns, args):
    # save only feature importance; prediction tables are exported by the notebook as test-only CSVs.
    v2_config.ensure_v2_output_dirs()
    args.feature_importance_csv.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.feature_importance_csv, index=False)
    return importance


def run_lgbm_temperature_pipeline(args):
    # run the full training and evaluation pipeline.
    model_df, feature_columns = build_lgbm_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        target_horizon=args.target_horizon,
        target_minutes=getattr(args, "target_minutes", None),
        object_role=args.object_role,
    )
    if getattr(args, "train_dates", None) and getattr(args, "valid_dates", None) and getattr(args, "test_dates", None):
        train_df, valid_df, test_df = split_by_dates(model_df, args.train_dates, args.valid_dates, args.test_dates)
    else:
        train_df, valid_df, test_df = split_by_time(model_df, args.valid_ratio, args.test_ratio)
    model, evals_result = train_lgbm_model(
        train_df,
        valid_df,
        feature_columns,
        random_state=args.random_state,
        model_params=getattr(args, "model_params", None),
    )
    predictions = predict_with_split(model, train_df, valid_df, test_df, feature_columns, args.temperature_column)

    metrics_rows = []
    for split_name in ("train", "valid", "test"):
        split_predictions = predictions[predictions["split"] == split_name]
        metrics = regression_metrics(split_predictions["target_temp"], split_predictions["prediction"])
        metrics_rows.append({"split": split_name, **metrics})

    importance = save_outputs(model, feature_columns, args)
    learning_curve = save_learning_curve_csv(evals_result, args.learning_curve_csv)
    return {
        "model_df": model_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "model": model,
        "predictions": predictions,
        "metrics": metrics_rows,
        "evals_result": evals_result,
        "learning_curve": learning_curve,
        "feature_importance": importance,
        "feature_columns": feature_columns,
    }


def main():
    # run LGBM training as a command line entry point.
    args = parse_args()
    result = run_lgbm_temperature_pipeline(args)
    print(f"Rows for modeling: {len(result['model_df'])}")
    print(f"Train/valid/test: {len(result['train_df'])}/{len(result['valid_df'])}/{len(result['test_df'])}")
    print(f"Feature importance CSV: {args.feature_importance_csv}")
    print(f"Learning curve CSV: {args.learning_curve_csv}")
    for row in result["metrics"]:
        print(f"{row['split']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}")


if __name__ == "__main__":
    main()
