from io import BytesIO

import pytest
from PIL import Image, UnidentifiedImageError

from image_preprocessing import preprocess_image


def test_preprocess_converts_image_to_rgb(sample_rgb_image: Image.Image) -> None:
    grayscale = sample_rgb_image.convert("L")
    encoded = BytesIO()
    grayscale.save(encoded, format="PNG")

    processed = preprocess_image(encoded.getvalue())

    assert processed.mode == "RGB"
    assert processed.size == sample_rgb_image.size


def test_preprocess_rejects_invalid_file() -> None:
    with pytest.raises(UnidentifiedImageError):
        preprocess_image(b"this is not an image")


def test_preprocess_accepts_image_path(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("L", (20, 10), color=128).save(image_path)

    processed = preprocess_image(image_path)

    assert processed.mode == "RGB"
    assert processed.size == (20, 10)


def test_preprocess_rejects_corrupt_image() -> None:
    valid_image = Image.new("RGB", (32, 32), color="black")
    encoded = BytesIO()
    valid_image.save(encoded, format="PNG")
    corrupt_bytes = encoded.getvalue()[:-12]

    with pytest.raises(OSError):
        preprocess_image(corrupt_bytes)
