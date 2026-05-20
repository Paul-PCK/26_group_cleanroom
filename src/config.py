from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

SIBLING_THERMAL_IMAGES_DIR = WORKSPACE_ROOT / "thermal_images"
IN_PROJECT_THERMAL_IMAGES_DIR = PROJECT_ROOT / "thermal_images"
THERMAL_IMAGES_DIR = (
    SIBLING_THERMAL_IMAGES_DIR
    if SIBLING_THERMAL_IMAGES_DIR.exists()
    else IN_PROJECT_THERMAL_IMAGES_DIR
)

MODELS_DIR = PROJECT_ROOT / "models"
TMP_DIR = PROJECT_ROOT / "tmp"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TIMESTAMP_LOOKUP_CSV = PROJECT_ROOT / "thermal_image_timestamp_lookup.csv"
SCALE_LABELS_CSV = PROJECT_ROOT / "used_scale_labels.csv"
LAYOUT_IMAGE = PROJECT_ROOT / "feynman_room_layout_without_axis.png"
STATIC_OBJECT_ANCHORS_CSV = PROJECT_ROOT / "static_object_anchors.csv"

YOLO_MODEL_PATH = MODELS_DIR / "pck_yolo_best.pt"
YOLO_BASE_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_SEG_BASE_MODEL_PATH = MODELS_DIR / "yolov8n-seg.pt"
HUMAN_PROJECTION_MODEL_PATH = MODELS_DIR / "pck_human_projection_nn_model.pth"
MACHINE_PROJECTION_MODEL_PATH = MODELS_DIR / "pck_machine_projection_nn_model.pth"

PREPROCESSED_DIR = TMP_DIR / "preprocessed"
PREPROCESSED_IMAGES_DIR = PREPROCESSED_DIR / "images"
PREPROCESSED_TEMPERATURE_MAPS_DIR = PREPROCESSED_DIR / "temperature_maps"
PREPROCESSING_SUMMARY_CSV = PREPROCESSED_DIR / "preprocessing_summary.csv"

YOLO_OUTPUT_DIR = TMP_DIR / "yolo"
YOLO_ANNOTATED_DIR = YOLO_OUTPUT_DIR / "annotated_images"
YOLO_DETECTIONS_CSV = YOLO_OUTPUT_DIR / "detections.csv"

PROJECTION_OUTPUT_DIR = TMP_DIR / "projection"
PROJECTED_DETECTIONS_CSV = PROJECTION_OUTPUT_DIR / "projected_detections.csv"

FINAL_OUTPUT_DIR = TMP_DIR / "final"
FINAL_TABLE_CSV = FINAL_OUTPUT_DIR / "final_detection_projection_temperature.csv"

INTEGRATION_OUTPUT_DIR = TMP_DIR / "integration"
INTEGRATED_OBJECTS_CSV = INTEGRATION_OUTPUT_DIR / "integrated_projected_objects.csv"
STATIC_OBJECT_REGISTRY_CSV = INTEGRATION_OUTPUT_DIR / "static_object_registry.csv"

TIMELINE_OUTPUT_DIR = TMP_DIR / "timeline"
OBJECT_TIMELINE_CSV = TIMELINE_OUTPUT_DIR / "object_timeline_temperature.csv"

LGBM_OUTPUT_DIR = TMP_DIR / "lgbm"
LGBM_HORIZON_COMPARE_DIR = LGBM_OUTPUT_DIR / "horizon_compare"
LGBM_MULTIHORIZON_DIR = LGBM_OUTPUT_DIR / "multihorizon"
LGBM_PREDICTIONS_CSV = LGBM_OUTPUT_DIR / "lgbm_temperature_predictions.csv"
LGBM_FEATURE_IMPORTANCE_CSV = LGBM_OUTPUT_DIR / "lgbm_feature_importance.csv"
LGBM_METRICS_CSV = LGBM_OUTPUT_DIR / "lgbm_metrics.csv"
LGBM_MODEL_TXT = LGBM_OUTPUT_DIR / "lgbm_temperature_model.txt"
LGBM_PEOPLE_IMPACT_CSV = LGBM_OUTPUT_DIR / "lgbm_people_impact_metrics.csv"
LGBM_LEARNING_CURVE_PNG = LGBM_OUTPUT_DIR / "lgbm_learning_curve.png"
LGBM_FEATURE_IMPORTANCE_PNG = LGBM_OUTPUT_DIR / "lgbm_feature_importance.png"
LGBM_PEOPLE_IMPACT_PNG = LGBM_OUTPUT_DIR / "lgbm_people_impact_metrics.png"
LGBM_PREDICTION_GIF_DIR = LGBM_OUTPUT_DIR / "prediction_gifs"

ANIMATION_OUTPUT_DIR = TMP_DIR / "animation"
TIMELINE_GIF = ANIMATION_OUTPUT_DIR / "object_timeline_dual_linechart.gif"
DAILY_ANIMATION_OUTPUT_DIR = ANIMATION_OUTPUT_DIR / "daily"

YOLO_TRAINING_DIR = TMP_DIR / "yolo_training"
YOLO_TRAINING_PROJECT_DIR = YOLO_TRAINING_DIR / "runs"

DEFAULT_IMAGE_GLOBS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
NORMALIZED_FACTOR = 640.0
PERSON_LABEL = "person"


def ensure_output_dirs():
    for path in (
        PREPROCESSED_IMAGES_DIR,
        PREPROCESSED_TEMPERATURE_MAPS_DIR,
        YOLO_ANNOTATED_DIR,
        PROJECTION_OUTPUT_DIR,
        FINAL_OUTPUT_DIR,
        INTEGRATION_OUTPUT_DIR,
        TIMELINE_OUTPUT_DIR,
        LGBM_OUTPUT_DIR,
        LGBM_HORIZON_COMPARE_DIR,
        LGBM_MULTIHORIZON_DIR,
        LGBM_PREDICTION_GIF_DIR,
        ANIMATION_OUTPUT_DIR,
        DAILY_ANIMATION_OUTPUT_DIR,
        YOLO_TRAINING_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def list_thermal_images(thermal_dir: Path = THERMAL_IMAGES_DIR, max_images: int | None = None):
    image_paths = []
    for pattern in DEFAULT_IMAGE_GLOBS:
        image_paths.extend(sorted(Path(thermal_dir).glob(pattern)))
    image_paths = sorted({path.resolve(): path for path in image_paths}.values(), key=lambda p: p.name)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return image_paths
