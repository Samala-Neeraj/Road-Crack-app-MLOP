from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.model
def test_model_loads_successfully(loaded_model) -> None:
    assert loaded_model is not None
    assert callable(loaded_model.predict)


@pytest.mark.model
def test_inference_on_sample_image_has_expected_result_structure(
    loaded_model, sample_rgb_image: Image.Image
) -> None:
    results = loaded_model.predict(
        source=sample_rgb_image, imgsz=320, device="cpu", verbose=False
    )

    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert hasattr(result, "boxes")
    assert hasattr(result, "names")
    assert result.orig_shape == (sample_rgb_image.height, sample_rgb_image.width)
    if result.boxes is not None and len(result.boxes) > 0:
        assert result.boxes.xyxy.shape[1] == 4
        assert result.boxes.cls.shape[0] == len(result.boxes)
        assert result.boxes.conf.shape[0] == len(result.boxes)


@pytest.mark.model
def test_real_sample_image_inference_returns_confidences_in_range(
    loaded_model, real_sample_image: Path
) -> None:
    results = loaded_model.predict(
        source=str(real_sample_image), imgsz=320, device="cpu", verbose=False
    )

    assert len(results) == 1
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        assert bool((boxes.conf >= 0).all())
        assert bool((boxes.conf <= 1).all())
