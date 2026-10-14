import pytest
import socket
import urllib.request

from prometheus_client import start_http_server

from metrics import (
    record_inference_latency,
    record_model_version,
    record_request_failed,
    record_request_started,
    record_successful_inference,
)


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_metrics_endpoint_exposes_inference_metrics() -> None:
    record_request_started()
    record_request_failed()
    record_inference_latency(0.25)
    record_model_version("pytest-model")
    record_successful_inference([("Longitudinal", 0.75)])

    server, thread = start_http_server(
        port=_available_port(), addr="127.0.0.1"
    )
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/metrics", timeout=5
        ) as response:
            body = response.read().decode("utf-8")
        assert response.status == 200
        for metric in (
            "road_crack_inference_requests_total",
            "road_crack_inference_errors_total",
            "road_crack_inference_latency_seconds_bucket",
            "road_crack_detections_total",
            "road_crack_confidence_score_bucket",
            "road_crack_severity_images_total",
            "road_crack_average_count_per_image",
            "road_crack_model_version_info",
        ):
            assert metric in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_invalid_detection_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        record_successful_inference([("Longitudinal", confidence)])
