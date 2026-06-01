from pathlib import Path

# project paths
# keep v2 path resolution independent from the original pipeline config.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

# shared folders
# all generated v2 artifacts are written under tmp/v2 by stage.
MODELS_DIR = PROJECT_ROOT / "models"
TMP_DIR = PROJECT_ROOT / "tmp"
V2_ROOT_DIR = TMP_DIR / "v2"
V2_INPUT_DIR = V2_ROOT_DIR / "input"
V2_MODELS_DIR = V2_ROOT_DIR / "models"
V2_PROJECTION_DIR = V2_ROOT_DIR / "projection"
V2_INTEGRATION_DIR = V2_ROOT_DIR / "integration"
V2_TIMELINE_DIR = V2_ROOT_DIR / "timeline"
V2_ANIMATION_DIR = V2_ROOT_DIR / "animation"
V2_LGBM_OUTPUT_DIR = V2_ROOT_DIR / "lgbm"
V2_LGBM_HORIZON_COMPARE_DIR = V2_LGBM_OUTPUT_DIR / "horizon_compare"
V2_XGBOOST_OUTPUT_DIR = V2_ROOT_DIR / "xgboost"
V2_XGBOOST_HORIZON_COMPARE_DIR = V2_XGBOOST_OUTPUT_DIR / "horizon_compare"
V2_MODEL_COMPARISON_DIR = V2_ROOT_DIR / "model_comparison"
LAYOUT_IMAGE = PROJECT_ROOT / "feynman_room_layout_without_axis.png"

# projection training inputs and model outputs
# hotspot folders provide bbox-to-map supervision for the v2 projection NN.
HOTSPOT_MACHINES_DIR = PROJECT_ROOT / "hotspot_machines"
HOTSPOT_PEOPLE_DIR = PROJECT_ROOT / "hotspot_people_loc"
V2_HUMAN_PROJECTION_MODEL_PATH = V2_MODELS_DIR / "v2_human_projection_nn_model.pth"
V2_MACHINE_PROJECTION_MODEL_PATH = V2_MODELS_DIR / "v2_machine_projection_nn_model.pth"
V2_HUMAN_TRAINING_METRICS_JSON = V2_MODELS_DIR / "v2_human_projection_training_metrics.json"
V2_MACHINE_TRAINING_METRICS_JSON = V2_MODELS_DIR / "v2_machine_projection_training_metrics.json"
V2_HUMAN_TRAINING_LOSS_PNG = V2_MODELS_DIR / "v2_human_projection_training_loss.png"
V2_MACHINE_TRAINING_LOSS_PNG = V2_MODELS_DIR / "v2_machine_projection_training_loss.png"

# active projection weights
# v2_01 uses these aliases so the notebook always reads v2-trained weights.
HUMAN_PROJECTION_MODEL_PATH = V2_HUMAN_PROJECTION_MODEL_PATH
MACHINE_PROJECTION_MODEL_PATH = V2_MACHINE_PROJECTION_MODEL_PATH

# stage outputs
# each path maps to a concrete notebook stage in the v2 workflow.
V2_RAW_BBOX_CSV = V2_ROOT_DIR / "new_thermal_bbox.csv"
V2_STANDARDIZED_BBOX_CSV = V2_INPUT_DIR / "v2_thermal_bbox_standardized.csv"
V2_PROJECTED_DETECTIONS_CSV = V2_PROJECTION_DIR / "v2_projected_detections.csv"
V2_PROJECTION_PREVIEW_PNG = V2_PROJECTION_DIR / "v2_projected_points_by_label.png"
V2_INTEGRATED_OBJECTS_CSV = V2_INTEGRATION_DIR / "v2_integrated_objects.csv"
V2_STATIC_OBJECT_REGISTRY_CSV = V2_INTEGRATION_DIR / "v2_static_object_registry.csv"
V2_DBSCAN_LAYOUT_PREVIEW_PNG = V2_INTEGRATION_DIR / "v2_dbscan_anchor_layout_preview.png"
V2_DBSCAN_OUTLIER_SUMMARY_PNG = V2_INTEGRATION_DIR / "v2_dbscan_outlier_summary_by_label.png"
V2_OBJECT_TIMELINE_CSV = V2_TIMELINE_DIR / "v2_object_timeline_temperature.csv"
V2_TIMELINE_GIF = V2_ANIMATION_DIR / "v2_object_timeline.gif"
V2_DAILY_ANIMATION_OUTPUT_DIR = V2_ANIMATION_DIR / "daily"

# projection constants
# bbox coordinates are normalized before NN inference, then mapped to room units.
NORMALIZED_FACTOR = 640.0
PERSON_LABEL = "person"
MAP_WIDTH = 15.0
MAP_HEIGHT = 8.0
PROJECTION_TARGET_Y_SOURCE_HEIGHT = 8.5
PROJECTION_TARGET_Y_TRAINING_HEIGHT = 6
PROJECTION_TARGET_Y_SCALE = PROJECTION_TARGET_Y_TRAINING_HEIGHT / PROJECTION_TARGET_Y_SOURCE_HEIGHT


def ensure_v2_output_dirs():
    # create the v2 output folders required by the active notebook stages.
    for path in (
        V2_INPUT_DIR,
        V2_MODELS_DIR,
        V2_PROJECTION_DIR,
        V2_INTEGRATION_DIR,
        V2_TIMELINE_DIR,
        V2_ANIMATION_DIR,
        V2_LGBM_OUTPUT_DIR,
        V2_LGBM_HORIZON_COMPARE_DIR,
        V2_XGBOOST_OUTPUT_DIR,
        V2_XGBOOST_HORIZON_COMPARE_DIR,
        V2_MODEL_COMPARISON_DIR,
        V2_DAILY_ANIMATION_OUTPUT_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
