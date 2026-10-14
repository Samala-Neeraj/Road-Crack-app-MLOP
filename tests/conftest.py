import os
from pathlib import Path
from typing import Iterator

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_path(project_root: Path) -> Path:
    configured_path = Path(os.getenv("MODEL_PATH", "best.pt")).expanduser()
    path = (
        configured_path
        if configured_path.is_absolute()
        else project_root / configured_path
    ).resolve()
    if not path.is_file():
        pytest.skip(f"YOLO checkpoint is not available: {path}")
    return path


@pytest.fixture(scope="session")
def loaded_model(model_path: Path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


@pytest.fixture
def sample_rgb_image() -> Iterator[Image.Image]:
    image = Image.new("RGB", (320, 240), color=(80, 80, 80))
    try:
        yield image
    finally:
        image.close()


@pytest.fixture(scope="session")
def data_yaml_path(project_root: Path) -> Path:
    configured_path = os.getenv("DATA_YAML")
    candidates = (
        [Path(configured_path).expanduser()]
        if configured_path
        else [project_root / "data.yaml", project_root / "dataset" / "data.yaml"]
    )
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else project_root / candidate
        if path.is_file():
            return path.resolve()
    pytest.skip(
        "Roboflow data.yaml is not present. Set DATA_YAML to its local path "
        "to run this check."
    )


@pytest.fixture(scope="session")
def real_sample_image(project_root: Path) -> Path:
    configured_path = os.getenv("SAMPLE_IMAGE")
    candidates = (
        [Path(configured_path).expanduser()]
        if configured_path
        else [
            project_root / "sample.jpg",
            project_root / "sample.png",
            project_root / "dataset" / "valid" / "images",
        ]
    )
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else project_root / candidate
        if path.is_file():
            return path.resolve()
        if path.is_dir():
            image_files = sorted(
                item
                for item in path.iterdir()
                if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if image_files:
                return image_files[0]
    pytest.skip(
        "No real road-image sample was found. Set SAMPLE_IMAGE or add one under "
        "dataset\\valid\\images to run this check."
    )
