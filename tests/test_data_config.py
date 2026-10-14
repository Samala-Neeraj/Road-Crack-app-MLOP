import os
from pathlib import Path

import pytest
import yaml


EXPECTED_CLASSES = ["Longitudinal", "Transverse", "Edge Crack", "Alligator"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_DATA_YAML = os.getenv("DATA_YAML")
DATA_YAML_CANDIDATES = (
    [Path(CONFIGURED_DATA_YAML).expanduser()]
    if CONFIGURED_DATA_YAML
    else [PROJECT_ROOT / "data.yaml", PROJECT_ROOT / "dataset" / "data.yaml"]
)
DATA_YAML_AVAILABLE = any(
    (
        candidate
        if candidate.is_absolute()
        else PROJECT_ROOT / candidate
    ).is_file()
    for candidate in DATA_YAML_CANDIDATES
)


@pytest.mark.skipif(
    not DATA_YAML_AVAILABLE,
    reason="No project data.yaml is present; configure DATA_YAML to enable.",
)
def test_data_yaml_has_four_expected_classes(data_yaml_path: Path) -> None:
    with data_yaml_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert isinstance(config, dict)
    assert config.get("nc") == 4
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    assert names == EXPECTED_CLASSES
