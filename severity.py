def severity_from_detection_count(detection_count: int) -> str:
    if isinstance(detection_count, bool) or not isinstance(detection_count, int):
        raise TypeError("detection_count must be an integer.")
    if detection_count < 0:
        raise ValueError("detection_count cannot be negative.")
    if detection_count == 0:
        return "Low"
    if detection_count <= 3:
        return "Medium"
    return "High"
