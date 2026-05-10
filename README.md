# 2026 Group Cleanroom

## 1. Project Purpose

The pipeline takes thermal images, applies YOLO detection, projects detections to a 2D floor map, joins temperature values, and learns stable object positions with unsupervised clustering. The learned positions are then used to build object timelines and daily animations.

## 2. Folder Structure

```text
2026_group_cleanroom/
  models/       trained weights used by the pipeline
  src/          reusable Python functions
  notebooks/    main execution notebooks
  docs/         project notes and decisions
  tmp/          regenerated pipeline outputs
  thermal_images/  local input image folder
```

## 3. Environment Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ITP
```


## 4. Required Inputs

The pipeline expects these inputs:

```text
thermal_images/
models/pck_yolo_best.pt
models/pck_human_projection_nn_model.pth
models/pck_machine_projection_nn_model.pth
thermal_image_timestamp_lookup.csv
used_scale_labels.csv
feynman_room_layout_without_axis.png
```

For YOLO retraining, the base weights are kept in `models/`:

```text
models/yolov8n.pt
models/yolov8n-seg.pt
```

## 5. Notebook Workflow

Run the notebooks in this order for new images without retraining:

```text
01_preprocessing.ipynb
03_yolo_apply.ipynb
04_projection.ipynb
05_final_table_and_object_integration.ipynb
06_generation.ipynb
```

Use this notebook only when retraining YOLO:

```text
02_yolo_training.ipynb
```

## 6. Object Integration / Learned Anchors

`05_final_table_and_object_integration.ipynb` learns stable object positions from detections with unsupervised clustering.

Available methods:

```text
DBSCAN
KMeans
GMM
```

Each method clusters all non-person detections by projected 2D position. Each learned cluster produces:

```text
object_id
canonical_label
anchor_x / anchor_y
display_x / display_y
label_counts
observations
```

`canonical_label` is decided by majority vote inside the cluster. `display_x` and `display_y` are the stable positions used by the timeline and animation.

## 7. Outputs

Main generated outputs:

```text
tmp/final/final_detection_projection_temperature.csv
tmp/integration/integrated_projected_objects.csv
tmp/integration/static_object_registry.csv
tmp/timeline/object_timeline_temperature.csv
tmp/animation/daily/
```

`tmp/integration/integrated_projected_objects.csv` is the key object-level output from notebook 05. Notebook 06 reads it and generates the timeline and daily GIFs.
