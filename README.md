# 2026 Group Cleanroom

## 1. Project Purpose

This folder contains the cleanroom thermal image pipeline.

The workflow includes:

- thermal image preprocessing
- YOLO detection
- 2D projection
- object integration
- object timeline generation
- temperature prediction
- model comparison

## 2. Folder Structure

```text
2026_group_cleanroom/
  models/
  src/
  notebooks/
  docs/
  tmp/
  thermal_images/
```

## 3. Environment Setup

Create the environment from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ITP
```

## 4. Required Inputs

```text
thermal_images/
models/pck_yolo_best.pt
models/yolov8n.pt
models/yolov8n-seg.pt
models/pck_human_projection_nn_model.pth
models/pck_machine_projection_nn_model.pth
thermal_image_timestamp_lookup.csv
used_scale_labels.csv
feynman_room_layout_without_axis.png
```

## 5. Notebook Workflow

Run these notebooks for the main image-to-timeline pipeline:

```text
notebooks/01_preprocessing.ipynb
notebooks/03_yolo_apply.ipynb
notebooks/04_projection.ipynb
notebooks/05_final_table_and_object_integration.ipynb
notebooks/06_generation.ipynb
```

Use this notebook only when retraining YOLO:

```text
notebooks/02_yolo_training.ipynb
```

Run these notebooks for baseline prediction outputs:

```text
notebooks/07_lgbm_temperature_prediction.ipynb
notebooks/07_1_xgboost_multihorizon_forecast.ipynb
```

Run these notebooks for hyperparameter tuning:

```text
notebooks/08_lgbm_hyperparameter_tuning.ipynb
notebooks/08_1_xgboost_hyperparameter_tuning.ipynb
```

After running notebook 08, rerun notebook 07 with tuned parameters enabled to generate tuned LGBM prediction CSV files.

After running notebook 08_1, rerun notebook 07_1 with tuned parameters enabled to generate tuned XGBoost prediction CSV files.

Run this notebook after the LGBM and XGBoost prediction CSV files exist:

```text
notebooks/09_model_comparison_lgbm_xgboost.ipynb
```

## 6. Main Outputs

Notebook 05 generates the combined detection, projection, temperature, and object integration outputs.

```text
tmp/final/final_detection_projection_temperature.csv
```

Contains detection rows with projected 2D positions and temperature values.

```text
tmp/integration/integrated_projected_objects.csv
```

Contains object-level rows after assigning detections to learned object anchors.

```text
tmp/integration/static_object_registry.csv
```

Contains the learned object anchors and object IDs used by later notebooks.

Notebook 06 generates the object timeline used by animation and prediction notebooks.

```text
tmp/timeline/object_timeline_temperature.csv
```

Contains timestamped object temperature records.

## 7. Prediction Result CSV Files

Notebook 07 generates LGBM prediction CSV files.

```text
tmp/lgbm/horizon_compare/05min/lgbm_temperature_predictions.csv
tmp/lgbm/horizon_compare/10min/lgbm_temperature_predictions.csv
```

These are baseline LGBM prediction CSV files.

```text
tmp/lgbm/horizon_compare/05min/lgbm_temperature_predictions_test.csv
tmp/lgbm/horizon_compare/10min/lgbm_temperature_predictions_test.csv
```

These are baseline LGBM test-only prediction CSV files.

```text
tmp/lgbm/horizon_compare/tuned/05min/lgbm_temperature_predictions.csv
tmp/lgbm/horizon_compare/tuned/10min/lgbm_temperature_predictions.csv
```

These are tuned LGBM prediction CSV files.

```text
tmp/lgbm/horizon_compare/tuned/05min/lgbm_temperature_predictions_test.csv
tmp/lgbm/horizon_compare/tuned/10min/lgbm_temperature_predictions_test.csv
```

These are tuned LGBM test-only prediction CSV files.

Notebook 07_1 generates XGBoost prediction CSV files.

```text
tmp/xgboost/horizon_compare/05min/xgboost_temperature_predictions.csv
tmp/xgboost/horizon_compare/10min/xgboost_temperature_predictions.csv
```

These are baseline XGBoost prediction CSV files.

```text
tmp/xgboost/horizon_compare/05min/xgboost_temperature_predictions_test.csv
tmp/xgboost/horizon_compare/10min/xgboost_temperature_predictions_test.csv
```

These are baseline XGBoost test-only prediction CSV files.

```text
tmp/xgboost/horizon_compare/tuned/05min/xgboost_temperature_predictions.csv
tmp/xgboost/horizon_compare/tuned/10min/xgboost_temperature_predictions.csv
```

These are tuned XGBoost prediction CSV files.

```text
tmp/xgboost/horizon_compare/tuned/05min/xgboost_temperature_predictions_test.csv
tmp/xgboost/horizon_compare/tuned/10min/xgboost_temperature_predictions_test.csv
```

These are tuned XGBoost test-only prediction CSV files.

`*_predictions.csv` contains train, valid, and test prediction rows.

`*_predictions_test.csv` contains test prediction rows only.

## 8. Hyperparameter Tuning Outputs

Notebook 08 generates LGBM tuning outputs.

```text
tmp/lgbm/tuning/best_lgbm_params.json
```

This file is loaded by notebook 07 when tuned parameters are enabled.

Notebook 08_1 generates XGBoost tuning outputs.

```text
tmp/xgboost/tuning/best_xgboost_params.json
```

This file is loaded by notebook 07_1 when tuned parameters are enabled.

## 9. Model Comparison Outputs

Notebook 09 reads existing LGBM and XGBoost prediction CSV files.

```text
tmp/model_comparison/combined_test_predictions.csv
```

Contains combined test prediction rows from the selected LGBM and XGBoost outputs.

```text
tmp/model_comparison/model_metrics.csv
tmp/model_comparison/category_metrics.csv
tmp/model_comparison/object_metrics.csv
```

Contain comparison metrics generated from the combined test prediction rows.

