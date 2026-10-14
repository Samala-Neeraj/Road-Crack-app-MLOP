import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from prometheus_client import start_http_server
from requests.exceptions import RequestException
from ultralytics import YOLO
from lineage_utils import (
    file_dataset,
    input_dataset,
    lineage_job,
    output_dataset,
)
from metrics import (
    record_inference_latency,
    record_model_version,
    record_request_failed,
    record_request_started,
    record_successful_inference,
)
from image_preprocessing import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

_configured_model_path = Path(os.getenv("MODEL_PATH", "best.pt")).expanduser()
MODEL_PATH = (
    _configured_model_path
    if _configured_model_path.is_absolute()
    else PROJECT_ROOT / _configured_model_path
).resolve()
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "path").strip().lower()
METRICS_PORT = int(os.getenv("PROMETHEUS_METRICS_PORT", "8000"))


@st.cache_resource
def start_metrics_server() -> object:
    return start_http_server(port=METRICS_PORT, addr="127.0.0.1")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_model_path() -> tuple[Path, str, str | None]:
    if MODEL_SOURCE == "path":
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Set MODEL_PATH in .env."
            )
        return MODEL_PATH, str(MODEL_PATH), None

    if MODEL_SOURCE != "registry":
        raise ValueError(
            f"MODEL_SOURCE must be 'path' or 'registry', not {MODEL_SOURCE!r}."
        )

    model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME", "road-crack-yolo11m"
    )
    alias = os.getenv("MODEL_REGISTRY_ALIAS", "production")
    try:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)
        version = MlflowClient().get_model_version_by_alias(model_name, alias)
        artifact_uri = f"{version.source.rstrip('/')}/best.pt"
        checkpoint = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        ).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Registry alias {model_name}@{alias} did not provide best.pt."
            )
        return (
            checkpoint,
            f"{model_name}@{alias} (version {version.version})",
            None,
        )
    except (MlflowException, OSError, RequestException) as error:
        if MODEL_PATH.is_file():
            return (
                MODEL_PATH,
                str(MODEL_PATH),
                f"Could not load MLflow model {model_name}@{alias}: {error}. "
                f"Using the configured local checkpoint {MODEL_PATH} instead.",
            )
        raise RuntimeError(
            f"Could not load MLflow model {model_name}@{alias}, and the "
            f"configured fallback checkpoint does not exist: {MODEL_PATH}. "
            "Start MLflow or set MODEL_SOURCE=path."
        ) from error


st.set_page_config(page_title="Road Crack Detection")
start_metrics_server()
st.title("Road Crack Detection")
st.write("Upload a road image to detect cracks.")

if MODEL_SOURCE not in {"path", "registry"}:
    st.error("Set MODEL_SOURCE in .env to either 'path' or 'registry'.")
    st.stop()

confidence = st.sidebar.slider(
    "Confidence threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05
)
uploaded_file = st.file_uploader(
    "Choose a road image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    if st.button("Run detection", type="primary"):
        record_request_started()
        inference_started_at = time.perf_counter()
        try:
            configured_model = (
                f"{os.getenv('MLFLOW_REGISTERED_MODEL_NAME', 'road-crack-yolo11m')}"
                f"@{os.getenv('MODEL_REGISTRY_ALIAS', 'production')}"
                if MODEL_SOURCE == "registry"
                else str(MODEL_PATH)
            )
            with lineage_job(
                "streamlit-inference",
                [
                    file_dataset(MODEL_PATH)
                    if MODEL_SOURCE == "path"
                    else input_dataset("mlflow-registry", configured_model),
                    input_dataset("streamlit-upload", "uploaded-road-image"),
                ],
                outputs=[
                    output_dataset("streamlit-inference", "crack-detections")
                ],
            ):
                image = preprocess_image(uploaded_file.getvalue())
                model_path, model_label, fallback_message = resolve_model_path()
                if fallback_message:
                    st.warning(fallback_message)
                model = load_model(str(model_path))
                if MODEL_SOURCE == "registry" and "(" in model_label:
                    model_version = model_label.rsplit("(version ", 1)[-1].rstrip(")")
                else:
                    model_version = model_path.name
                record_model_version(model_version)
                result = model.predict(
                    source=image, conf=confidence, verbose=False
                )[0]
                boxes = result.boxes
                detections: list[tuple[str, float]] = []
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        detections.append(
                            (
                                str(result.names[class_id]),
                                float(box.conf[0].item()),
                            )
                        )
                severity = record_successful_inference(detections)
        except Exception as error:
            record_request_failed()
            st.error(f"Prediction failed: {error}")
        else:
            st.caption(f"Model: {model_label}")
            st.caption(f"Image severity: {severity}")
            original_column, result_column = st.columns(2)
            with original_column:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with result_column:
                st.image(
                    result.plot()[:, :, ::-1],
                    caption="Detection result",
                    use_container_width=True,
                )

            if not detections:
                st.info("No cracks detected above the selected confidence threshold.")
            else:
                st.subheader(f"Detections: {len(detections)}")
                for label, score in detections:
                    st.write(f"{label}: {score:.1%}")
        finally:
            record_inference_latency(
                time.perf_counter() - inference_started_at
            )