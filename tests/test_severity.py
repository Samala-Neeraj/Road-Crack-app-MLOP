import pytest

from severity import severity_from_detection_count


@pytest.mark.parametrize(
    ("detection_count", "expected"),
    [
        (0, "Low"),
        (1, "Medium"),
        (3, "Medium"),
        (4, "High"),
        (100, "High"),
    ],
)
def test_severity_thresholds(detection_count: int, expected: str) -> None:
    assert severity_from_detection_count(detection_count) == expected


@pytest.mark.parametrize("detection_count", [-1, -5])
def test_negative_detection_count_is_rejected(detection_count: int) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        severity_from_detection_count(detection_count)


@pytest.mark.parametrize("detection_count", [1.5, "2", None, True])
def test_non_integer_detection_count_is_rejected(detection_count: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        severity_from_detection_count(detection_count)  # type: ignore[arg-type]
