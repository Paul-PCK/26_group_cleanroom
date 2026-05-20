from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np
import pandas as pd

from config import LGBM_MULTIHORIZON_DIR, OBJECT_TIMELINE_CSV, ensure_output_dirs
from lgbm_temperature import (
    CATEGORICAL_FEATURES,
    CURRENT_TEMP_FEATURES,
    LAG_FEATURES,
    PEOPLE_CONTEXT_FEATURES,
    POSITION_FEATURES,
    TIME_FEATURES,
    add_lag_features,
    add_people_context_features,
    add_time_features,
    aggregate_object_timeline,
    load_timeline,
    regression_metrics,
    save_learning_curve_plot,
    split_by_dates,
    split_by_time,
)


DEFAULT_FORECAST_HORIZONS_MIN = [5, 10, 15, 20, 30, 45, 60, 90, 120]


def build_multihorizon_dataset(
    timeline_csv: Path = OBJECT_TIMELINE_CSV,
    temperature_column: str = "temp_mean_c",
    forecast_horizons_min=None,
    people_or_machine: str = "machine",
):
    # Build one row per object, source timestamp, and forecast horizon.
    forecast_horizons_min = list(forecast_horizons_min or DEFAULT_FORECAST_HORIZONS_MIN)
    raw_df = load_timeline(timeline_csv)
    object_df = aggregate_object_timeline(raw_df, temperature_column, people_or_machine=people_or_machine)
    if people_or_machine == "machine":
        object_df = add_people_context_features(object_df, raw_df)

    featured = add_time_features(object_df)

    # Add history features once; the target columns are rebuilt for every horizon below.
    featured = add_lag_features(
        featured,
        temperature_column=temperature_column,
        target_horizon=1,
        target_minutes=min(forecast_horizons_min),
    )
    required_lag = max(int(name.rsplit("_", 1)[-1]) for name in LAG_FEATURES if name.startswith("temp_lag_"))
    featured = featured.dropna(subset=[f"temp_lag_{required_lag}"]).copy()
    featured = featured.drop(columns=["target_temp", "target_timestamp", "target_minutes_ahead"], errors="ignore")

    expanded_parts = []
    for _, group in featured.groupby(["object_id", "_sequence_date"], sort=False):
        group = group.sort_values("timestamp_dt").copy()
        timestamps = group["timestamp_dt"].to_numpy(dtype="datetime64[ns]")
        indices = group.index.to_numpy()

        for horizon_min in forecast_horizons_min:
            source_targets = (group["timestamp_dt"] + pd.Timedelta(minutes=horizon_min)).to_numpy(dtype="datetime64[ns]")
            target_positions = np.searchsorted(timestamps, source_targets, side="left")
            valid = target_positions < len(group)
            if not valid.any():
                continue

            source_index = indices[valid]
            target_index = indices[target_positions[valid]]
            actual_delta = featured.loc[target_index, "timestamp_dt"].to_numpy() - featured.loc[source_index, "timestamp_dt"].to_numpy()
            within_window = actual_delta <= pd.Timedelta(minutes=horizon_min * 1.5)
            if not within_window.any():
                continue

            part = featured.loc[source_index[within_window]].copy()
            final_target_index = target_index[within_window]
            part["target_temp"] = featured.loc[final_target_index, temperature_column].to_numpy()
            part["target_timestamp"] = featured.loc[final_target_index, "timestamp_dt"].to_numpy()
            part["requested_horizon_min"] = float(horizon_min)
            part["target_minutes_ahead"] = (
                part["target_timestamp"] - part["timestamp_dt"]
            ).dt.total_seconds() / 60.0
            expanded_parts.append(part)

    if not expanded_parts:
        raise ValueError("No valid multi-horizon training rows were created.")

    model_df = pd.concat(expanded_parts, ignore_index=True)
    feature_columns = (
        CATEGORICAL_FEATURES
        + POSITION_FEATURES
        + TIME_FEATURES
        + CURRENT_TEMP_FEATURES
        + PEOPLE_CONTEXT_FEATURES
        + LAG_FEATURES
        + ["observations_in_frame", "target_minutes_ahead", "requested_horizon_min"]
    )
    model_df = model_df.dropna(subset=["target_temp"]).copy()
    for column in CATEGORICAL_FEATURES:
        model_df[column] = model_df[column].fillna("").astype("category")
    for column in feature_columns:
        if column not in CATEGORICAL_FEATURES:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    return model_df, feature_columns


def train_forecast_models(train_df, valid_df, feature_columns, random_state=42):
    # Train mean, lower-quantile, and upper-quantile LightGBM models.
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing lightgbm. Install it in the active environment before training.") from exc

    base_params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    fit_params = dict(
        eval_set=[
            (train_df[feature_columns], train_df["target_temp"]),
            (valid_df[feature_columns], valid_df["target_temp"]),
        ],
        eval_names=["train", "valid"],
        categorical_feature=CATEGORICAL_FEATURES,
    )

    mean_evals = {}
    mean_model = lgb.LGBMRegressor(objective="regression", **base_params)
    mean_model.fit(
        train_df[feature_columns],
        train_df["target_temp"],
        eval_metric="rmse",
        callbacks=[
            lgb.record_evaluation(mean_evals),
            lgb.early_stopping(80, verbose=False),
        ],
        **fit_params,
    )

    lower_model = lgb.LGBMRegressor(objective="quantile", alpha=0.025, **base_params)
    lower_model.fit(
        train_df[feature_columns],
        train_df["target_temp"],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(80, verbose=False)],
        **fit_params,
    )

    upper_model = lgb.LGBMRegressor(objective="quantile", alpha=0.975, **base_params)
    upper_model.fit(
        train_df[feature_columns],
        train_df["target_temp"],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(80, verbose=False)],
        **fit_params,
    )
    return {"mean": mean_model, "lower": lower_model, "upper": upper_model}, mean_evals


def predict_forecast_models(models, train_df, valid_df, test_df, feature_columns, temperature_column):
    # Predict every split with mean and 95% quantile interval models.
    outputs = []
    for split_name, split_df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        part = split_df[
            [
                "object_id",
                "timestamp",
                "timestamp_dt",
                "target_timestamp",
                "canonical_label",
                "display_x",
                "display_y",
                temperature_column,
                "requested_horizon_min",
                "target_minutes_ahead",
                *PEOPLE_CONTEXT_FEATURES,
                "target_temp",
            ]
        ].copy()
        part = part.rename(columns={temperature_column: "source_temp"})
        part["split"] = split_name
        part["prediction"] = models["mean"].predict(split_df[feature_columns])
        part["ci_lower"] = models["lower"].predict(split_df[feature_columns])
        part["ci_upper"] = models["upper"].predict(split_df[feature_columns])
        interval_values = np.vstack([part["ci_lower"], part["prediction"], part["ci_upper"]])
        part["ci_lower"] = np.nanmin(interval_values, axis=0)
        part["ci_upper"] = np.nanmax(interval_values, axis=0)
        part["error"] = part["prediction"] - part["target_temp"]
        part["covered_by_95ci"] = (part["target_temp"] >= part["ci_lower"]) & (part["target_temp"] <= part["ci_upper"])
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def build_forecast_metrics(predictions):
    # Summarize mean prediction error and 95% interval coverage.
    rows = []
    for keys, part in predictions.groupby(["split", "requested_horizon_min"], sort=True):
        split_name, horizon_min = keys
        metrics = regression_metrics(part["target_temp"], part["prediction"])
        rows.append(
            {
                "split": split_name,
                "requested_horizon_min": horizon_min,
                "rows": len(part),
                **metrics,
                "mean_error": float(part["error"].mean()),
                "ci_width_mean": float((part["ci_upper"] - part["ci_lower"]).mean()),
                "ci_coverage": float(part["covered_by_95ci"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_forecast_outputs(models, predictions, metrics, feature_columns, args, evals_result):
    # Save multi-horizon predictions, metrics, feature importance, and models.
    ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.prediction_csv, index=False)
    metrics.to_csv(args.metrics_csv, index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": models["mean"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.feature_importance_csv, index=False)

    for name, model in models.items():
        model.booster_.save_model(str(args.output_dir / f"lgbm_multihorizon_{name}.txt"))
    learning_curve = save_learning_curve_plot(evals_result, args.learning_curve_png)
    return importance, learning_curve


def safe_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "object"


def build_group_model_metrics(predictions, group_column):
    # Summarize forecast error and CI coverage for each trained model group.
    rows = []
    group_columns = ["split", group_column, "requested_horizon_min"]
    for keys, part in predictions.groupby(group_columns, sort=True):
        split_name, model_group, horizon_min = keys
        metrics = regression_metrics(part["target_temp"], part["prediction"])
        rows.append(
            {
                "split": split_name,
                group_column: model_group,
                "requested_horizon_min": horizon_min,
                "rows": len(part),
                **metrics,
                "mean_error": float(part["error"].mean()),
                "ci_width_mean": float((part["ci_upper"] - part["ci_lower"]).mean()),
                "ci_coverage": float(part["covered_by_95ci"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_group_forecast_outputs(
    group_models,
    predictions,
    metrics,
    group_metrics,
    feature_columns,
    args,
    evals_by_group,
    group_column,
):
    # Save combined predictions and one model set per group.
    ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.prediction_csv, index=False)
    metrics.to_csv(args.metrics_csv, index=False)

    group_metrics_csv = args.output_dir / "lgbm_multihorizon_group_metrics.csv"
    group_metrics.to_csv(group_metrics_csv, index=False)

    importance_rows = []
    models_dir = args.output_dir / "per_group_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_group, models in group_models.items():
        group_dir = models_dir / safe_filename(model_group)
        group_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            model.booster_.save_model(str(group_dir / f"{name}.txt"))
        for feature, importance in zip(feature_columns, models["mean"].feature_importances_):
            importance_rows.append(
                {
                    group_column: model_group,
                    "feature": feature,
                    "importance": int(importance),
                }
            )

    importance = pd.DataFrame(importance_rows).sort_values([group_column, "importance"], ascending=[True, False])
    importance.to_csv(args.feature_importance_csv, index=False)

    first_evals = next(iter(evals_by_group.values()), None)
    learning_curve = None
    if first_evals is not None:
        learning_curve = save_learning_curve_plot(first_evals, args.learning_curve_png)
    return importance, learning_curve, group_metrics_csv


def run_grouped_multihorizon_forecast_pipeline(args):
    # Train one independent multi-horizon forecast model per group.
    model_df, feature_columns = build_multihorizon_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        forecast_horizons_min=args.forecast_horizons_min,
        people_or_machine=args.people_or_machine,
    )
    group_column = getattr(args, "group_model_column", "canonical_label")
    if group_column not in model_df.columns:
        raise ValueError(f"Missing group model column: {group_column}")

    prediction_parts = []
    group_models = {}
    evals_by_group = {}
    split_parts = {"train": [], "valid": [], "test": []}
    skipped_rows = []

    for model_group, group_df in model_df.groupby(group_column, sort=True):
        group_df = group_df.sort_values(["object_id", "timestamp_dt"]).copy()
        try:
            if getattr(args, "train_dates", None) and getattr(args, "valid_dates", None) and getattr(args, "test_dates", None):
                train_df, valid_df, test_df = split_by_dates(group_df, args.train_dates, args.valid_dates, args.test_dates)
            else:
                train_df, valid_df, test_df = split_by_time(group_df, args.valid_ratio, args.test_ratio)
            models, evals_result = train_forecast_models(train_df, valid_df, feature_columns, random_state=args.random_state)
            group_predictions = predict_forecast_models(models, train_df, valid_df, test_df, feature_columns, args.temperature_column)
        except Exception as exc:
            skipped_rows.append({group_column: model_group, "reason": str(exc), "rows": len(group_df)})
            continue

        group_predictions["model_scope"] = f"per_{group_column}"
        group_predictions["model_group_column"] = group_column
        group_predictions["model_group"] = model_group
        prediction_parts.append(group_predictions)
        group_models[model_group] = models
        evals_by_group[model_group] = evals_result
        split_parts["train"].append(train_df)
        split_parts["valid"].append(valid_df)
        split_parts["test"].append(test_df)

    if not prediction_parts:
        raise ValueError("No grouped models were trained. Check date splits and group row counts.")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = build_forecast_metrics(predictions)
    group_metrics = build_group_model_metrics(predictions, group_column)
    importance, learning_curve, group_metrics_csv = save_group_forecast_outputs(
        group_models,
        predictions,
        metrics,
        group_metrics,
        feature_columns,
        args,
        evals_by_group,
        group_column,
    )
    skipped_groups = pd.DataFrame(skipped_rows)
    if not skipped_groups.empty:
        skipped_groups.to_csv(args.output_dir / "lgbm_multihorizon_skipped_groups.csv", index=False)

    return {
        "model_df": model_df,
        "train_df": pd.concat(split_parts["train"], ignore_index=True) if split_parts["train"] else pd.DataFrame(),
        "valid_df": pd.concat(split_parts["valid"], ignore_index=True) if split_parts["valid"] else pd.DataFrame(),
        "test_df": pd.concat(split_parts["test"], ignore_index=True) if split_parts["test"] else pd.DataFrame(),
        "models": group_models,
        "predictions": predictions,
        "metrics": metrics,
        "group_metrics": group_metrics,
        "group_metrics_csv": group_metrics_csv,
        "skipped_groups": skipped_groups,
        "feature_importance": importance,
        "learning_curve": learning_curve,
        "evals_result": evals_by_group,
        "feature_columns": feature_columns,
    }


def run_multihorizon_forecast_pipeline(args):
    # Run the full multi-horizon forecast and 95% CI pipeline.
    if getattr(args, "group_models", False):
        return run_grouped_multihorizon_forecast_pipeline(args)

    model_df, feature_columns = build_multihorizon_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        forecast_horizons_min=args.forecast_horizons_min,
        people_or_machine=args.people_or_machine,
    )
    if getattr(args, "train_dates", None) and getattr(args, "valid_dates", None) and getattr(args, "test_dates", None):
        train_df, valid_df, test_df = split_by_dates(model_df, args.train_dates, args.valid_dates, args.test_dates)
    else:
        train_df, valid_df, test_df = split_by_time(model_df, args.valid_ratio, args.test_ratio)

    models, evals_result = train_forecast_models(train_df, valid_df, feature_columns, random_state=args.random_state)
    predictions = predict_forecast_models(models, train_df, valid_df, test_df, feature_columns, args.temperature_column)
    metrics = build_forecast_metrics(predictions)
    importance, learning_curve = save_forecast_outputs(models, predictions, metrics, feature_columns, args, evals_result)
    return {
        "model_df": model_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "models": models,
        "predictions": predictions,
        "metrics": metrics,
        "feature_importance": importance,
        "learning_curve": learning_curve,
        "evals_result": evals_result,
        "feature_columns": feature_columns,
    }


def default_multihorizon_args():
    return SimpleNamespace(
        input_csv=OBJECT_TIMELINE_CSV,
        output_dir=LGBM_MULTIHORIZON_DIR,
        prediction_csv=LGBM_MULTIHORIZON_DIR / "lgbm_multihorizon_predictions.csv",
        metrics_csv=LGBM_MULTIHORIZON_DIR / "lgbm_multihorizon_metrics.csv",
        feature_importance_csv=LGBM_MULTIHORIZON_DIR / "lgbm_multihorizon_feature_importance.csv",
        learning_curve_png=LGBM_MULTIHORIZON_DIR / "lgbm_multihorizon_learning_curve.png",
        temperature_column="temp_mean_c",
        forecast_horizons_min=DEFAULT_FORECAST_HORIZONS_MIN,
        people_or_machine="machine",
        train_dates=[],
        valid_dates=[],
        test_dates=[],
        valid_ratio=0.15,
        test_ratio=0.15,
        group_models=True,
        group_model_column="canonical_label",
        random_state=42,
    )
