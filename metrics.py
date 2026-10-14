import math
import threading

from prometheus_client import Counter, Gauge, Histogram

from severity import severity_from_detection_count


INFERENCE_REQUESTS = Counter(
    "road_crack_inference_requests_total",
    "Number of road crack inference requests started.",
)
INFERENCE_ERRORS = Counter(
    "road_crack_inference_errors_total",
    "Number of failed road crack inference requests.",
)
INFERENCE_LATENCY = Histogram(
    "road_crack_inference_latency_seconds",
    "Time spent processing road crack inference requests.",
)
DETECTIONS_PER_CLASS = Counter(
    "road_crack_detections_total",
    "Number of detected cracks grouped by class.",
    labelnames=("class_name",),
)
CONFIDENCE_SCORES = Histogram(
    "road_crack_confidence_score",
    "Confidence scores of detected cracks.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
SEVERITY_DISTRIBUTION = Counter(
    "road_crack_severity_images_total",
    "Successfully processed images grouped by detection-count severity.",
    labelnames=("severity",),
)
AVERAGE_CRACK_COUNT = Gauge(
    "road_crack_average_count_per_image",
    "Running average number of detected cracks per successful image.",
)
MODEL_VERSION_INFO = Gauge(
    "road_crack_model_version_info",
    "Currently selected model version; the active version has value 1.",
    labelnames=("model_version",),
)

for _severity in ("Low", "Medium", "High"):
    SEVERITY_DISTRIBUTION.labels(severity=_severity).inc(0)

_average_lock = threading.Lock()
_processed_image_count = 0
_total_detection_count = 0


def record_request_started() -> None:
    INFERENCE_REQUESTS.inc()


def record_request_failed() -> None:
    INFERENCE_ERRORS.inc()


def record_inference_latency(seconds: float) -> None:
    INFERENCE_LATENCY.observe(seconds)


def record_model_version(version: str) -> None:
    MODEL_VERSION_INFO.clear()
    MODEL_VERSION_INFO.labels(model_version=version).set(1)


def record_successful_inference(
    detections: list[tuple[str, float]],
) -> str:
    global _processed_image_count, _total_detection_count

    detection_count = len(detections)
    severity = severity_from_detection_count(detection_count)
    for class_name, confidence in detections:
        if not class_name:
            raise ValueError("class_name cannot be empty.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")

    for class_name, confidence in detections:
        DETECTIONS_PER_CLASS.labels(class_name=class_name).inc()
        CONFIDENCE_SCORES.observe(confidence)
    SEVERITY_DISTRIBUTION.labels(severity=severity).inc()

    with _average_lock:
        _processed_image_count += 1
        _total_detection_count += detection_count
        AVERAGE_CRACK_COUNT.set(
            _total_detection_count / _processed_image_count
        )
    return severity
