
"""Pruebas del preprocesamiento de imágenes."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    load_image_rgb,
    resize_with_padding,
    resize_and_normalize,
    segment_fruit_uniform_background,
    crop_to_foreground,
    extract_color_histogram,
    extract_combined_features,
    build_feature_dataset,
    augment_training_image,
)


def save_rgb(path, image):
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )
    assert cv2.imwrite(str(path), image_bgr)


def example_fruit():
    # Fondo claro
    image = np.full(
        (150, 220, 3),
        245,
        dtype=np.uint8
    )

    # Fruta roja artificial
    cv2.circle(
        image,
        (110, 75),
        47,
        (195, 35, 30),
        -1
    )

    # Mancha oscura
    cv2.circle(
        image,
        (95, 67),
        8,
        (65, 20, 15),
        -1
    )

    return image


def test_load_image_rgb_preserves_rgb_order(tmp_path):
    image = np.array(
        [[[230, 20, 5], [10, 225, 40]]],
        dtype=np.uint8
    )

    path = tmp_path / "example.png"
    save_rgb(path, image)

    result = load_image_rgb(path)

    assert np.array_equal(result, image)


def test_load_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_rgb(tmp_path / "missing.png")


def test_load_image_corrupt_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("Esto no es una imagen")

    with pytest.raises(ValueError):
        load_image_rgb(path)


def test_resize_with_padding_preserves_proportion():
    image = np.full(
        (50, 100, 3),
        (10, 10, 10),
        dtype=np.uint8
    )

    image[:, 25:75, :] = (230, 30, 20)

    result = resize_with_padding(
        image,
        (128, 128),
        (240, 240, 240)
    )

    assert result.shape == (128, 128, 3)

    assert np.array_equal(
        result[0, 0],
        np.array((240, 240, 240), dtype=np.uint8)
    )

    assert np.array_equal(
        result[32, 0],
        image[0, 0]
    )

    assert np.array_equal(
        result[95, 127],
        image[-1, -1]
    )


def test_resize_and_normalize_output(tmp_path):
    path = tmp_path / "fruit.png"
    save_rgb(path, example_fruit())

    result = resize_and_normalize(path)

    assert result.shape == (128, 128, 3)
    assert result.dtype == np.float32

    assert 0 <= result.min()
    assert result.max() <= 1


def test_histogram_sum_and_length():
    image = np.zeros(
        (10, 20, 3),
        dtype=np.uint8
    )

    image[:, :10] = (255, 0, 0)
    image[:, 10:] = (0, 0, 255)

    hist = extract_color_histogram(image)

    assert hist.shape == (512,)
    assert hist.dtype == np.float32
    assert hist.sum() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 2

    assert sorted(
        hist[hist > 0].tolist()
    ) == pytest.approx([0.5, 0.5])


def test_histogram_ignores_background():
    image = np.full(
        (20, 20, 3),
        (255, 255, 255),
        dtype=np.uint8
    )

    image[5:15, 5:15] = (190, 30, 15)

    mask = np.zeros(
        (20, 20),
        dtype=np.uint8
    )

    mask[5:15, 5:15] = 255

    full_hist = extract_color_histogram(image)

    fruit_hist = extract_color_histogram(
        image,
        mask=mask
    )

    assert np.count_nonzero(full_hist) == 2
    assert np.count_nonzero(fruit_hist) == 1
    assert fruit_hist.sum() == pytest.approx(1.0)


def test_segment_fruit_and_fill_spot():
    image = example_fruit()

    mask = segment_fruit_uniform_background(
        image
    )

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8

    # Centro de la fruta
    assert mask[75, 110] == 255

    # Mancha dentro de la fruta
    assert mask[67, 95] == 255

    # Fondo
    assert mask[0, 0] == 0

    assert 0.17 < np.mean(mask > 0) < 0.25

    cropped, cropped_mask = crop_to_foreground(
        image,
        mask
    )

    assert cropped.shape[:2] == cropped_mask.shape
    assert cropped.shape[0] < image.shape[0]
    assert cropped.shape[1] < image.shape[1]


def test_segmentation_rejects_two_fruits():
    image = np.full(
        (140, 300, 3),
        245,
        dtype=np.uint8
    )

    cv2.circle(
        image,
        (80, 70),
        42,
        (205, 35, 20),
        -1
    )

    cv2.circle(
        image,
        (220, 70),
        42,
        (205, 35, 20),
        -1
    )

    with pytest.raises(
        ValueError,
        match="varias regiones"
    ):
        segment_fruit_uniform_background(image)


def test_segmentation_rejects_uniform_image():
    image = np.full(
        (64, 64, 3),
        240,
        dtype=np.uint8
    )

    with pytest.raises(ValueError):
        segment_fruit_uniform_background(image)


def test_existing_apis_build_x_and_y(tmp_path):
    path = tmp_path / "fruit.png"
    save_rgb(path, example_fruit())

    vector = extract_combined_features(path)

    assert vector.shape == (512,)
    assert vector.sum() == pytest.approx(1)

    df = pd.DataFrame({
        "path": [str(path)],
        "quality": ["Good"]
    })

    X, y = build_feature_dataset(df)

    assert X.shape == (1, 512)
    assert y.tolist() == ["Good"]


def test_skip_invalid_images_keeps_labels_aligned(tmp_path):
    path = tmp_path / "fruit.png"
    save_rgb(path, example_fruit())

    df = pd.DataFrame({
        "path": [
            str(tmp_path / "missing.jpg"),
            str(path)
        ],
        "quality": ["Bad", "Good"]
    })

    with pytest.warns(
        RuntimeWarning,
        match="Se omitieron 1 imágenes"
    ):
        X, y = build_feature_dataset(
            df,
            on_error="skip"
        )

    assert X.shape == (1, 512)
    assert y.tolist() == ["Good"]


def test_augmentation_reproducible():
    image = example_fruit()

    a = augment_training_image(
        image,
        rng=np.random.default_rng(42)
    )

    b = augment_training_image(
        image,
        rng=np.random.default_rng(42)
    )

    assert np.array_equal(a, b)
    assert a.dtype == np.uint8
    assert a.shape == image.shape
