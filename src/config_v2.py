from pathlib import Path


# resolve project paths without depending on the original config module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

# locate thermal images from the expected workspace layout.
SIBLING_THERMAL_IMAGES_DIR = WORKSPACE_ROOT / "thermal_images"
IN_PROJECT_THERMAL_IMAGES_DIR = PROJECT_ROOT / "thermal_images"
THERMAL_IMAGES_DIR = (
    SIBLING_THERMAL_IMAGES_DIR
    if SIBLING_THERMAL_IMAGES_DIR.exists()
    else IN_PROJECT_THERMAL_IMAGES_DIR
)

# define shared v2 project folders.
MODELS_DIR = PROJECT_ROOT / "models"
TMP_DIR = PROJECT_ROOT / "tmp"
V2_ROOT_DIR = TMP_DIR / "v2"
V2_INPUT_DIR = V2_ROOT_DIR / "input"
V2_PROJECTION_DIR = V2_ROOT_DIR / "projection"

# define projection model weights.
HUMAN_PROJECTION_MODEL_PATH = MODELS_DIR / "pck_human_projection_nn_model.pth"
MACHINE_PROJECTION_MODEL_PATH = MODELS_DIR / "pck_machine_projection_nn_model.pth"

# define v2 source input and projection-stage outputs.
V2_RAW_BBOX_CSV = V2_ROOT_DIR / "new_thermal_bbox.csv"
V2_STANDARDIZED_BBOX_CSV = V2_INPUT_DIR / "v2_thermal_bbox_standardized.csv"
V2_PROJECTED_DETECTIONS_CSV = V2_PROJECTION_DIR / "v2_projected_detections.csv"

# define projection constants.
NORMALIZED_FACTOR = 640.0
PERSON_LABEL = "person"

# control whether v2_01 replaces CSV timestamps with image EXIF timestamps.
USE_IMAGE_EXIF_TIMESTAMP = True


def ensure_v2_output_dirs():
    # create only the v2 folders currently used by v2_01.
    for path in (
        V2_INPUT_DIR,
        V2_PROJECTION_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
