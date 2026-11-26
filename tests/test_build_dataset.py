import cv2
import numpy as np
import pandas as pd

from src.data import build_dataset
from src.data import preprocess


def test_scan_source_includes_heic_and_resizes_for_analysis(tmp_path, monkeypatch):
    source_dir = tmp_path / "recolectado"
    image_path = (
        source_dir
        / "Good_Quality_Fruits"
        / "Apple_good"
        / "photo.HEIC"
    )
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    monkeypatch.setattr(build_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_dataset,
        "read_image",
        lambda _: np.zeros((1600, 1200, 3), dtype=np.uint8),
    )
    analyzed_sizes = []

    def fake_mask_and_size(image):
        analyzed_sizes.append(image.shape[:2])
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.5, "mediano"

    monkeypatch.setattr(
        build_dataset, "get_fruit_mask_and_size", fake_mask_and_size
    )

    records = build_dataset.scan_source(source_dir, "recolectado")

    assert len(records) == 1
    assert records[0]["filepath"].endswith("photo.HEIC")
    assert records[0]["fruit"] == "apple"
    assert records[0]["quality"] == "good"
    assert max(analyzed_sizes[0]) == build_dataset.ANALYSIS_MAX_DIMENSION


def test_resize_for_analysis_keeps_small_images_unchanged():
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    resized = build_dataset.resize_for_analysis(image)

    assert resized is image


def test_resize_for_analysis_preserves_aspect_ratio():
    image = np.zeros((1600, 1200, 3), dtype=np.uint8)

    resized = build_dataset.resize_for_analysis(image)

    assert resized.shape == (768, 576, 3)


def test_read_image_uses_reduced_decode_for_jpeg(tmp_path, monkeypatch):
    image_path = tmp_path / "photo.jpg"
    expected_image = np.zeros((100, 100, 3), dtype=np.uint8)
    calls = []

    def fake_imread(path, flag):
        calls.append((path, flag))
        return expected_image

    monkeypatch.setattr(build_dataset.cv2, "imread", fake_imread)

    image = build_dataset.read_image(image_path)

    assert image is expected_image
    assert calls == [(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)]


def test_prepare_ml_split_loads_heic_rows(tmp_path, monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(preprocess, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "read_image", lambda _: image)
    monkeypatch.setattr(
        preprocess,
        "extract_ml_features",
        lambda _: np.array([1.0], dtype=np.float32),
    )
    rows = pd.DataFrame(
        [{"filepath": "photo.HEIC", "quality": "good", "size_label": "mediano"}]
    )

    features, quality, size = preprocess.prepare_ml_split(rows)

    assert features.shape == (1, 1)
    assert quality.tolist() == [preprocess.QUALITY_TO_IDX["good"]]
    assert size.tolist() == [preprocess.SIZE_TO_IDX["mediano"]]


def test_prepare_cnn_split_loads_heic_rows(tmp_path, monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(preprocess, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "read_image", lambda _: image)
    monkeypatch.setattr(
        preprocess,
        "preprocess_image_for_cnn",
        lambda image, target_size: np.zeros((*target_size, 3), dtype=np.float32),
    )
    rows = pd.DataFrame(
        [{"filepath": "photo.HEIC", "quality": "bad", "size_label": "grande"}]
    )

    images, quality, size = preprocess.prepare_cnn_split(rows, target_size=(8, 8))

    assert images.shape == (1, 8, 8, 3)
    assert quality.tolist() == [preprocess.QUALITY_TO_IDX["bad"]]
    assert size.tolist() == [preprocess.SIZE_TO_IDX["grande"]]
