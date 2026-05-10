from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import SCALE_LABELS_CSV, THERMAL_IMAGES_DIR


DEFAULT_SCALE_BAR = (618, 47, 633, 431)
DEFAULT_FALLBACK_TOP = 33.1
DEFAULT_FALLBACK_BOTTOM = 19.0
DEFAULT_NORMALIZE_MIN = 18.0
DEFAULT_NORMALIZE_MAX = 50.0

TOP_TEXT_REGION = (560, 20, 640, 70)
BOTTOM_TEXT_REGION = (560, 410, 640, 460)
TOP_DIGIT_BOXES = ((18, 0, 34, 18), (33, 0, 49, 18), (58, 0, 73, 18))
BOTTOM_DIGIT_BOXES = ((20, 34, 30, 50), (34, 34, 49, 50), (57, 34, 72, 50))
TOP_TEMPERATURE_RANGE = (20.0, 40.0)
BOTTOM_TEMPERATURE_RANGE = (18.0, 22.0)

REFERENCE_SCALE_TEXT = {
    "IR_60330.jpg": {"top": "33.1", "bottom": "19.0"},
    "IR_57327.jpg": {"top": "34.0", "bottom": "19.5"},
    "IR_57328.jpg": {"top": "32.6", "bottom": "20.1"},
    "IR_57507.jpg": {"top": "26.8", "bottom": "19.1"},
    "IR_58430.jpg": {"top": "25.1", "bottom": "18.9"},
    "IR_59883.jpg": {"top": "23.9", "bottom": "18.5"},
}

PATCH_SIZE = (24, 32)
TRANSLATION_OFFSETS = (-1, 0, 1)
DISTANCE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class ScaleDigitLayout:
    top_region: tuple[int, int, int, int]
    bottom_region: tuple[int, int, int, int]
    top_digit_boxes: tuple[tuple[int, int, int, int], ...]
    bottom_digit_boxes: tuple[tuple[int, int, int, int], ...]
    top_valid_range: tuple[float, float]
    bottom_valid_range: tuple[float, float]

    def region_for(self, position: str):
        return self.top_region if position == "top" else self.bottom_region

    def boxes_for(self, position: str):
        return self.top_digit_boxes if position == "top" else self.bottom_digit_boxes

    def valid_range_for(self, position: str):
        return self.top_valid_range if position == "top" else self.bottom_valid_range


DIGIT_LAYOUT = ScaleDigitLayout(
    top_region=TOP_TEXT_REGION,
    bottom_region=BOTTOM_TEXT_REGION,
    top_digit_boxes=TOP_DIGIT_BOXES,
    bottom_digit_boxes=BOTTOM_DIGIT_BOXES,
    top_valid_range=TOP_TEMPERATURE_RANGE,
    bottom_valid_range=BOTTOM_TEMPERATURE_RANGE,
)


@dataclass
class SlotModel:
    labels: np.ndarray
    features: np.ndarray


def crop_digit_patch(image_array, region, digit_box):
    x0, y0, _, _ = region
    dx0, dy0, dx1, dy1 = digit_box
    patch = image_array[y0 + dy0 : y0 + dy1, x0 + dx0 : x0 + dx1]
    if patch.size == 0:
        raise ValueError("Digit patch is empty.")
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    return gray


def normalize_patch(gray_patch):
    patch = gray_patch.astype(np.float32)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    mean = float(patch.mean())
    std = float(patch.std()) or 1.0
    if std < 1e-6:
        std = 1.0
    return ((patch - mean) / std).reshape(-1)


def augment_patch(gray_patch):
    height, width = gray_patch.shape
    padded = cv2.copyMakeBorder(gray_patch, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
    augmented = []
    for dx in TRANSLATION_OFFSETS:
        for dy in TRANSLATION_OFFSETS:
            view = padded[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]
            augmented.append(view)
            augmented.append(cv2.GaussianBlur(view, (3, 3), 0.4))
    return augmented


class ScaleDigitClassifier:
    def __init__(
        self,
        thermal_dir: Path,
        reference_scale_text: dict[str, dict[str, str]],
        layout: ScaleDigitLayout = DIGIT_LAYOUT,
    ):
        self.thermal_dir = Path(thermal_dir)
        self.reference_scale_text = reference_scale_text
        self.layout = layout
        self.slot_models = self._build_slot_models()

    def _build_slot_models(self):
        slot_samples: dict[tuple[str, int], list[np.ndarray]] = {}
        slot_labels: dict[tuple[str, int], list[str]] = {}

        for filename, labels in self.reference_scale_text.items():
            image_path = self.thermal_dir / filename
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                image_array = np.array(image.convert("RGB"))

            for position, value in labels.items():
                digits = value.replace(".", "")
                boxes = self.layout.boxes_for(position)
                if len(digits) != len(boxes):
                    raise ValueError(
                        f"Expected {len(boxes)} digits for {filename} {position}, got {len(digits)}."
                    )
                for digit_index, (digit, digit_box) in enumerate(zip(digits, boxes)):
                    gray_patch = crop_digit_patch(
                        image_array, self.layout.region_for(position), digit_box
                    )
                    for augmented in augment_patch(gray_patch):
                        slot_key = (position, digit_index)
                        slot_samples.setdefault(slot_key, []).append(normalize_patch(augmented))
                        slot_labels.setdefault(slot_key, []).append(digit)

        slot_models = {}
        for slot_key, features in slot_samples.items():
            slot_models[slot_key] = SlotModel(
                labels=np.array(slot_labels[slot_key]),
                features=np.vstack(features).astype(np.float32),
            )
        if not slot_models:
            raise ValueError("No scale digit training samples were built.")
        return slot_models

    def digit_candidates(self, image_array, position: str, digit_index: int):
        slot_key = (position, digit_index)
        model = self.slot_models.get(slot_key)
        if model is None:
            raise ValueError(f"No digit model for slot {slot_key}.")
        gray_patch = crop_digit_patch(
            image_array,
            self.layout.region_for(position),
            self.layout.boxes_for(position)[digit_index],
        )
        feature = normalize_patch(gray_patch)
        distances = np.mean((model.features - feature[None, :]) ** 2, axis=1)

        best_per_digit = {}
        for label, distance in zip(model.labels, distances):
            distance = float(distance)
            if label not in best_per_digit or distance < best_per_digit[label]:
                best_per_digit[label] = distance
        return sorted(best_per_digit.items(), key=lambda item: item[1])

    def read_temperature(self, image_array, position: str):
        valid_range = self.layout.valid_range_for(position)
        candidate_sets = [
            self.digit_candidates(image_array, position, digit_index)
            for digit_index in range(3)
        ]

        best_value = None
        best_score = float("inf")
        for digit_a, score_a in candidate_sets[0][:4]:
            for digit_b, score_b in candidate_sets[1][:4]:
                for digit_c, score_c in candidate_sets[2][:4]:
                    value = float(f"{digit_a}{digit_b}.{digit_c}")
                    score = score_a + score_b + score_c
                    if valid_range[0] <= value <= valid_range[1] and score < best_score:
                        best_value = value
                        best_score = score

        if best_value is None:
            digit_a, score_a = candidate_sets[0][0]
            digit_b, score_b = candidate_sets[1][0]
            digit_c, score_c = candidate_sets[2][0]
            best_value = float(f"{digit_a}{digit_b}.{digit_c}")
            best_score = score_a + score_b + score_c
        return best_value, best_score

    def resolve_temperatures(self, image_array, fallback_top, fallback_bottom):
        try:
            top_value, top_score = self.read_temperature(image_array, "top")
            bottom_value, bottom_score = self.read_temperature(image_array, "bottom")
            return {
                "top": top_value,
                "bottom": bottom_value,
                "source": "digit_classifier",
                "ocr_score": max(top_score, bottom_score),
            }
        except Exception:
            return {
                "top": fallback_top,
                "bottom": fallback_bottom,
                "source": "fallback",
                "ocr_score": float("nan"),
            }


def load_reference_scale_text(labels_csv: Path = SCALE_LABELS_CSV):
    reference_scale_text = dict(REFERENCE_SCALE_TEXT)
    labels_csv = Path(labels_csv)
    if labels_csv.exists():
        with labels_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_name = row.get("image_name", "").strip()
                top = row.get("top", "").strip()
                bottom = row.get("bottom", "").strip()
                if image_name and top and bottom:
                    reference_scale_text[image_name] = {"top": top, "bottom": bottom}
    return reference_scale_text


def build_scale_digit_classifier(
    thermal_dir: Path = THERMAL_IMAGES_DIR,
    labels_csv: Path = SCALE_LABELS_CSV,
):
    return ScaleDigitClassifier(thermal_dir, load_reference_scale_text(labels_csv), DIGIT_LAYOUT)


def build_scale_bar_mapping(image_array, scale_bar, temp_top, temp_bottom):
    x0, y0, x1, y1 = scale_bar
    crop = image_array[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Scale bar crop is empty. Check scale-bar coordinates.")
    palette_rgb = crop.mean(axis=1).astype(np.float32)
    palette_temp = np.linspace(temp_top, temp_bottom, len(palette_rgb)).astype(np.float32)
    return palette_rgb, palette_temp


def estimate_temperature_map(image_array, palette_rgb, palette_temp):
    flat_pixels = image_array.reshape(-1, 3).astype(np.float32)
    unique_pixels, inverse = np.unique(flat_pixels, axis=0, return_inverse=True)
    unique_temperatures = np.empty(len(unique_pixels), dtype=np.float32)

    for start in range(0, len(unique_pixels), DISTANCE_CHUNK_SIZE):
        end = min(start + DISTANCE_CHUNK_SIZE, len(unique_pixels))
        chunk = unique_pixels[start:end]
        distances = ((chunk[:, None, :] - palette_rgb[None, :, :]) ** 2).sum(axis=2)
        unique_temperatures[start:end] = palette_temp[distances.argmin(axis=1)]

    return unique_temperatures[inverse].reshape(image_array.shape[:2]).astype(np.float32)


def transform_to_bbox_space(array, rotate_180=True, pad_height=640, pad_value=0.0):
    transformed = np.rot90(array, k=2) if rotate_180 else array.copy()
    height, width = transformed.shape
    if pad_height > height:
        padded = np.full((pad_height, width), pad_value, dtype=transformed.dtype)
        padded[:height, :] = transformed
        transformed = padded
    return transformed


def normalize_temperature_map(temperature_map, normalize_min, normalize_max):
    denominator = normalize_max - normalize_min
    if denominator <= 0:
        raise ValueError("normalize_max must be greater than normalize_min.")
    return np.clip((temperature_map - normalize_min) / denominator, 0.0, 1.0)


def image_from_normalized(normalized_image):
    scaled = np.clip(np.round(normalized_image * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def estimate_image_temperature_map(image_path: Path, classifier: ScaleDigitClassifier, args):
    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"))
    scale_info = classifier.resolve_temperatures(
        image_array,
        args.fallback_top,
        args.fallback_bottom,
    )
    palette_rgb, palette_temp = build_scale_bar_mapping(
        image_array,
        tuple(args.scale_bar),
        scale_info["top"],
        scale_info["bottom"],
    )
    temperature_map = estimate_temperature_map(image_array, palette_rgb, palette_temp)
    return temperature_map, scale_info
