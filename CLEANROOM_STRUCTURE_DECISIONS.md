# Cleanroom Structure Decisions

This document records the agreed structure decisions for the new `2026_group_cleanroom` project.
It is meant to be read at the start of future sessions before changing files.

## Project Direction

The new main project will be `2026_group_cleanroom`.

The old `ITP` folder may be used only as a reference source. Files from `ITP` should not be copied blindly, because many of them are legacy, debug, duplicated, or transition files.

The goal is to build a clean, minimal, restartable workflow that can run from raw thermal images to final outputs.

Forecasting is not part of the current focus.

## Top-Level Layout

The intended layout is:

```text
thermal_images/
2026_group_cleanroom/
  models/
  src/
  notebooks/
  docs/
  tmp/
```

## `thermal_images/`

`thermal_images/` is the raw input image folder.

Decision:
- It should be treated as the original thermal image input folder.
- It should contain raw `.jpg` thermal images such as `IR_57327.jpg`.
- It should not contain generated outputs, annotations, CSV files, notebooks, models, or debug images.

Current note:
- The intended design is that `thermal_images/` is the input image folder at the same level as `2026_group_cleanroom/`.
- If the current workspace still has `thermal_images/` inside `2026_group_cleanroom/`, treat that as a temporary location. Code should use a configurable path instead of hard-coding old `ITP` paths.

## `2026_group_cleanroom/models/`

`models/` should contain only model files that are actually useful for the cleanroom workflow.

Decision:
- Keep the trained YOLO detection model for normal inference.
- Keep the trained human and machine projection models for projection inference.
- Also keep the YOLO base models because the cleanroom project should preserve the ability to retrain YOLO.

Files to keep:

```text
models/
  pck_yolo_best.pt
  pck_human_projection_nn_model.pth
  pck_machine_projection_nn_model.pth
  yolov8n.pt
  yolov8n-seg.pt
```

Model roles:

```text
pck_yolo_best.pt
```

The trained YOLO detection model used by the main pipeline to produce labels, confidence scores, and bounding boxes.

```text
pck_human_projection_nn_model.pth
```

The trained projection model used for `Person` detections.

```text
pck_machine_projection_nn_model.pth
```

The trained projection model used for non-`Person` detections.

```text
yolov8n.pt
```

YOLOv8 nano detection base model. Keep this so YOLO detection can be retrained from a base model if needed.

```text
yolov8n-seg.pt
```

YOLOv8 nano segmentation base model. Keep this as an available base model, even though segmentation is not the current main pipeline.

## `2026_group_cleanroom/src/`

`src/` should contain only the main Python code used by the cleanroom pipeline.

Decision:
- Keep only pipeline code.
- Do not place notebooks, models, raw images, output CSV files, output images, debug images, old test scripts, or temporary experiment scripts here.
- Each file should own one clear pipeline responsibility.
- Paths and fixed settings should be centralized in `config.py` instead of hard-coded across scripts.

Planned files:

```text
src/
  config.py
  scale_bar.py
  preprocessing.py
  yolo_train.py
  yolo_apply.py
  projection.py
  final_table.py
  object_integration.py
  timeline.py
  animate_timeline.py
  run_pipeline.py
```

File roles:

```text
config.py
```

Centralizes project paths and fixed settings, including:
- `PROJECT_ROOT`
- `THERMAL_IMAGES_DIR`
- `MODELS_DIR`
- `TMP_DIR`
- YOLO model path
- projection model paths
- output CSV paths

```text
scale_bar.py
```

Handles thermal image scale-bar logic, including:
- reading scale-bar colors
- detecting or assigning top and bottom temperature labels
- fallback temperature rules
- color-to-temperature mapping

```text
preprocessing.py
```

Converts raw thermal images into YOLO-ready images, including:
- reading `.jpg` input images
- scale-bar temperature normalization
- rotating images 180 degrees
- padding images to 640x640
- writing preprocessed images and temperature maps

```text
yolo_train.py
```

Preserves the ability to retrain YOLO, including:
- building or reading a YOLO dataset
- using `yolov8n.pt` as the main detection base model
- keeping `yolov8n-seg.pt` available for future segmentation work
- exporting newly trained weights

```text
yolo_apply.py
```

Runs object detection using `pck_yolo_best.pt`, producing:
- labels
- confidence scores
- bounding boxes

```text
projection.py
```

Projects bounding boxes to the 2D room map, including:
- `HumanNN`
- `MachineNN`
- `Person` detections use the human projection model
- non-`Person` detections use the machine projection model

```text
final_table.py
```

Merges detection results, projection coordinates, and temperature statistics into the main final table.

```text
object_integration.py
```

Merges duplicate detections within the same frame and links the same static object across time.

```text
timeline.py
```

Builds timeline data from integrated object results.

```text
animate_timeline.py
```

Generates the preferred timeline animation or GIF output.

```text
run_pipeline.py
```

Runs the full cleanroom pipeline from beginning to end.

Intended command:

```bash
python src/run_pipeline.py
```

## `2026_group_cleanroom/notebooks/`

`notebooks/` should contain the primary interactive execution workflow for operating, checking, and presenting the cleanroom workflow.

Decision:
- Do not keep old notebook history versions here.
- Do not keep multiple notebooks for the same old workflow.
- Do not place output CSV files, models, raw images, or generated images here.
- Notebooks should contain the runnable workflow body: parameter setup, function calls, previews, checks, and visual review.
- `src/` should provide reusable functions and optional command-line entry points. It should not make notebooks into empty shells.
- The notebook structure should be separated by workflow stage: processing, training, application/conversion, final table/object integration, and generation.

Planned files:

```text
notebooks/
  00_overview.ipynb
  01_preprocessing.ipynb
  02_yolo_training.ipynb
  03_yolo_apply.ipynb
  04_projection.ipynb
  05_final_table_and_object_integration.ipynb
  06_generation.ipynb
```

Notebook roles:

```text
00_overview.ipynb
```

Project overview and workflow entry point. It should include:
- project path checks
- model file checks
- input image folder checks
- notebook execution order
- full pipeline command

It should not contain heavy implementation details.

```text
01_preprocessing.ipynb
```

Handles raw thermal image preprocessing by calling reusable functions from `src/`. It should include:
- reading `thermal_images/`
- scale-bar temperature normalization
- image rotation and 640x640 padding
- visual checks of preprocessed images
- checks of temperature maps

Related `src/` files:
- `scale_bar.py`
- `preprocessing.py`

```text
02_yolo_training.ipynb
```

Preserves YOLO retraining capability by calling reusable functions from `src/`. It should include:
- building or checking the YOLO dataset
- training from `yolov8n.pt`
- optionally using `yolov8n-seg.pt` if segmentation work is needed later
- evaluating training results
- exporting new trained weights

Related `src/` file:
- `yolo_train.py`

```text
03_yolo_apply.ipynb
```

Runs trained YOLO inference by calling reusable functions from `src/`. It should include:
- loading `pck_yolo_best.pt`
- running detection on preprocessed images
- writing `detections.csv`
- showing annotated detection images

Related `src/` file:
- `yolo_apply.py`

```text
04_projection.ipynb
```

Projects bounding boxes to the 2D room map by calling reusable functions from `src/`. It should include:
- loading human and machine projection models
- routing `Person` detections to the human model
- routing non-`Person` detections to the machine model
- writing projected coordinates
- checking projected positions

Related `src/` file:
- `projection.py`

```text
05_final_table_and_object_integration.ipynb
```

Builds the final table and integrates objects by calling reusable functions from `src/`. It should include:
- merging YOLO detections
- merging projection coordinates
- computing bbox temperature mean and max
- creating the final detection/projection/temperature table
- cleaning same-frame duplicate detections
- linking the same static object across time
- assigning object IDs
- checking the integration table

Related `src/` files:
- `final_table.py`
- `object_integration.py`

```text
06_generation.ipynb
```

Generates final timeline and animation outputs by calling reusable functions from `src/`. It should include:
- building the object timeline CSV
- generating GIF or animation output
- checking final visualization outputs

Related `src/` files:
- `timeline.py`
- `animate_timeline.py`

## `2026_group_cleanroom/docs/`

`docs/` should contain the cleanroom project documentation.

Decision:
- Do not use `docs/` as an old-file backup folder.
- Use it to help a future session, teammate, or notebook user understand the current workflow quickly.
- Keep formal project documentation here.

Planned files:

```text
docs/
  README.md
  PROJECT_CONTEXT.md
  PIPELINE.md
  FILE_REGISTRY.md
  OUTPUTS.md
  DECISIONS.md
```

## `2026_group_cleanroom/tmp/`

`tmp/` should contain only generated outputs.

Decision:
- All pipeline outputs should be written under `tmp/`.
- `tmp/` can be deleted and regenerated.
- Do not place source code, notebooks, models, or raw images here.

Planned folders:

```text
tmp/
  preprocessed/
  yolo/
  projection/
  final/
  integration/
  timeline/
  animation/
  yolo_training/
```

## Static Object Position Decision

Background objects should not be forced to move every frame just because the YOLO bbox shifts.

Current decision:
- Static object positions should be learned from detections, not manually predefined.
- `05_final_table_and_object_integration.ipynb` supports `STATIC_CLUSTERING = 'dbscan'`, `'kmeans'`, or `'gmm'`.
- All three modes cluster non-person detections globally by projected position.
- Each learned cluster gets `anchor_x` / `anchor_y` from the median projected position.
- Each learned cluster gets `canonical_label` from majority vote inside the cluster.
- `display_x` / `display_y` use the learned cluster anchor so the animation is stable.

Unsupervised modes:
- `dbscan`: does not require a fixed number of clusters and can leave outliers as noise.
- `kmeans`: requires `KMEANS_CLUSTERS`.
- `gmm`: tests 1..`GMM_MAX_COMPONENTS` and selects the component count by BIC.

## Finalized Layout

The cleanroom project is now expected to use this layout:

```text
thermal_images/
2026_group_cleanroom/
  models/
  src/
  notebooks/
  docs/
  tmp/
```
