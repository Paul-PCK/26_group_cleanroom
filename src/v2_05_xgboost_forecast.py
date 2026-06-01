from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np
import pandas as pd

import v2_config
from v2_04_lgbm_temperature import (
    CATEGORICAL_FEATURES,
    PEOPLE_CONTEXT_FEATURES,
    build_model_dataset,
    regression_metrics,
    split_by_dates,
    split_by_time,
)


DEFAULT_FORECAST_HORIZONS_MIN = [5, 10, 15, 20, 30, 45, 60, 90, 120]


def save_learning_curve_csv(evals_result, output_path: Path):
    # save train/valid RMSE over boosting iterations for downstream comparison.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, metrics in evals_result.items():
        rmse_values = metrics.get("rmse") or []
        for iteration, rmse in enumerate(rmse_values, start=1):
            rows.append({"iteration": iteration, "split": split_name, "rmse": float(rmse)})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_multihorizon_dataset(
    timeline_csv: Path = v2_config.V2_OBJECT_TIMELINE_CSV,
    temperature_column: str = "temp_mean_c",
    forecast_horizons_min=None,
    object_role: str = "machine",
):
    # build one shared feature table per forecast horizon.
    forecast_horizons_min = list(forecast_horizons_min or DEFAULT_FORECAST_HORIZONS_MIN)
    expanded_parts = []
    feature_columns = None
    for horizon_min in forecast_horizons_min:
        part, feature_columns = build_model_dataset(
            timeline_csv=timeline_csv,
            temperature_column=temperature_column,
            target_horizon=1,
            target_minutes=float(horizon_min),
            object_role=object_role,
        )
        part = part.copy()
        part["requested_horizon_min"] = float(horizon_min)
        expanded_parts.append(part)

    if not expanded_parts:
        raise ValueError("No valid multi-horizon training rows were created.")

    model_df = pd.concat(expanded_parts, ignore_index=True)
    return model_df, feature_columns


def train_forecast_models(train_df, valid_df, feature_columns, random_state=42, model_params=None):
    # train one XGBoost mean regression model for point prediction.
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing xgboost. Install xgboost>=2.0 in the active environment before training.") from exc

    base_params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=80,
    )
    if model_params:
        base_params.update(model_params)
    base_params["random_state"] = random_state
    base_params["n_jobs"] = -1
    base_params["tree_method"] = "hist"
    base_params["enable_categorical"] = True
    fit_params = dict(
        eval_set=[
            (train_df[feature_columns], train_df["target_temp"]),
            (valid_df[feature_columns], valid_df["target_temp"]),
        ],
        verbose=False,
    )

    mean_model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **base_params)
    mean_model.fit(
        train_df[feature_columns],
        train_df["target_temp"],
        **fit_params,
    )
    mean_evals_raw = mean_model.evals_result()
    mean_evals = {
        "train": mean_evals_raw.get("validation_0", {}),
        "valid": mean_evals_raw.get("validation_1", {}),
    }

    return {"mean": mean_model}, mean_evals


def predict_forecast_models(models, train_df, valid_df, test_df, feature_columns, temperature_column):
    # predict every split with the mean regression model.
    outputs = []
    for split_name, split_df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        part = split_df[
            [
                "object_id",
                "timestamp",
                "timestamp_dt",
                "target_timestamp",
                "label",
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
        part["error"] = part["prediction"] - part["target_temp"]
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def build_forecast_metrics(predictions):
    # summarize point-prediction error.
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
            }
        )
    return pd.DataFrame(rows)


def save_forecast_outputs(models, predictions, metrics, feature_columns, args, evals_result):
    # save feature importance and learning curve; the notebook exports test-only predictions.
    v2_config.ensure_v2_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": models["mean"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.feature_importance_csv, index=False)

    learning_curve = save_learning_curve_csv(evals_result, args.learning_curve_csv)
    return importance, learning_curve


def safe_filename(value):
    # create filesystem-safe names for object or group identifiers.
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "object"


def build_group_model_metrics(predictions, group_column):
    # summarize forecast error for each trained model group.
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
    # save grouped feature importance and one learning curve for optional grouped runs.
    v2_config.ensure_v2_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    importance_rows = []
    for model_group, models in group_models.items():
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
        learning_curve = save_learning_curve_csv(first_evals, args.learning_curve_csv)
    return importance, learning_curve


def run_grouped_multihorizon_forecast_pipeline(args):
    # train one independent multi-horizon forecast model per group.
    model_df, feature_columns = build_multihorizon_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        forecast_horizons_min=args.forecast_horizons_min,
        object_role=args.object_role,
    )
    group_column = getattr(args, "group_model_column", "label")
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
            models, evals_result = train_forecast_models(
                train_df,
                valid_df,
                feature_columns,
                random_state=args.random_state,
                model_params=getattr(args, "model_params", None),
            )
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
    importance, learning_curve = save_group_forecast_outputs(
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

    return {
        "model_df": model_df,
        "train_df": pd.concat(split_parts["train"], ignore_index=True) if split_parts["train"] else pd.DataFrame(),
        "valid_df": pd.concat(split_parts["valid"], ignore_index=True) if split_parts["valid"] else pd.DataFrame(),
        "test_df": pd.concat(split_parts["test"], ignore_index=True) if split_parts["test"] else pd.DataFrame(),
        "models": group_models,
        "predictions": predictions,
        "metrics": metrics,
        "group_metrics": group_metrics,
        "skipped_groups": skipped_groups,
        "feature_importance": importance,
        "learning_curve": learning_curve,
        "evals_result": evals_by_group,
        "feature_columns": feature_columns,
    }


def run_multihorizon_forecast_pipeline(args):
    # run the full multi-horizon point-forecast pipeline.
    if getattr(args, "group_models", False):
        return run_grouped_multihorizon_forecast_pipeline(args)

    model_df, feature_columns = build_multihorizon_dataset(
        timeline_csv=args.input_csv,
        temperature_column=args.temperature_column,
        forecast_horizons_min=args.forecast_horizons_min,
        object_role=args.object_role,
    )
    if getattr(args, "train_dates", None) and getattr(args, "valid_dates", None) and getattr(args, "test_dates", None):
        train_df, valid_df, test_df = split_by_dates(model_df, args.train_dates, args.valid_dates, args.test_dates)
    else:
        train_df, valid_df, test_df = split_by_time(model_df, args.valid_ratio, args.test_ratio)

    models, evals_result = train_forecast_models(
        train_df,
        valid_df,
        feature_columns,
        random_state=args.random_state,
        model_params=getattr(args, "model_params", None),
    )
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
    # provide default paths and settings for notebook execution.
    output_dir = v2_config.V2_XGBOOST_HORIZON_COMPARE_DIR / "cli"
    return SimpleNamespace(
        input_csv=v2_config.V2_OBJECT_TIMELINE_CSV,
        output_dir=output_dir,
        feature_importance_csv=output_dir / "xgboost_multihorizon_feature_importance.csv",
        learning_curve_csv=output_dir / "xgboost_multihorizon_learning_curve.csv",
        temperature_column="temp_mean_c",
        forecast_horizons_min=DEFAULT_FORECAST_HORIZONS_MIN,
        object_role="machine",
        train_dates=[],
        valid_dates=[],
        test_dates=[],
        valid_ratio=0.15,
        test_ratio=0.15,
        group_models=True,
        group_model_column="label",
        random_state=42,
    )
