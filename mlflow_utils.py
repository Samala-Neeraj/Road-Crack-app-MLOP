import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


PROJECT_ROOT = Path(__file__).resolve().parent

PLOT_PATTERNS = (
    "confusion_matrix*.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
)

METRIC_NAME_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/precision": "precision",
    "metrics/recall(B)": "recall",
    "metrics/recall": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "metrics/mAP50-95": "map50_95",
}


def configure_tracking() -> str:
    """Load project settings and select the configured MLflow tracking server."""
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def mlflow_metric_name(name: str) -> str:
    """Map YOLO CSV labels to MLflow-safe metric names."""
    if name in METRIC_NAME_ALIASES:
        return METRIC_NAME_ALIASES[name]
    safe_name = re.sub(r"[^A-Za-z0-9_. /-]", "_", name).strip()
    if not safe_name:
        raise ValueError(f"Metric name cannot be empty: {name!r}")
    return safe_name


def env_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the project when it is not absolute."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping and fail clearly for missing or malformed files."""
    import yaml

    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_path}")
    return value


def mlflow_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return "None"
    return str(value)


def numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            converted[str(key)] = number
    return converted


def log_results_csv(csv_path: str | Path) -> dict[str, float]:
    """Log each numeric results.csv column at its original epoch step."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO results.csv not found: {path}")

    last_metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not rows.fieldnames:
            raise ValueError(f"YOLO results.csv has no header: {path}")
        for row_index, row in enumerate(rows, start=1):
            if not row:
                continue
            step_value = row.get("epoch")
            try:
                step = int(float(step_value)) if step_value else row_index
            except ValueError:
                step = row_index

            metrics: dict[str, float] = {}
            for column, raw_value in row.items():
                if column is None or column.strip().lower() in {"epoch", "time"}:
                    continue
                try:
                    number = float(raw_value) if raw_value is not None else math.nan
                except ValueError:
                    continue
                if math.isfinite(number):
                    metric_name = mlflow_metric_name(column.strip())
                    metrics[metric_name] = number

            if metrics:
                mlflow.log_metrics(metrics, step=step)
                last_metrics = metrics

    if not last_metrics:
        raise ValueError(f"No numeric epoch metrics found in {path}")
    return last_metrics


def log_best_model_and_training_artifacts(
    best_model_path: str | Path,
    run_directory: str | Path,
    data_yaml_path: str | Path | None = None,
    args_yaml_path: str | Path | None = None,
) -> None:
    """Log the YOLO checkpoint and available result, config, and diagnostic files."""
    best_path = Path(best_model_path)
    run_dir = Path(run_directory)
    if not best_path.is_file():
        raise FileNotFoundError(f"Best model checkpoint not found: {best_path}")
    mlflow.log_artifact(str(best_path), artifact_path="best_model")

    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        mlflow.log_artifact(str(results_csv), artifact_path="training")

    for label, candidate in (
        ("dataset", Path(data_yaml_path) if data_yaml_path else None),
        ("training", Path(args_yaml_path) if args_yaml_path else None),
    ):
        if candidate is not None and candidate.is_file():
            mlflow.log_artifact(str(candidate), artifact_path=label)

    found_plots: set[Path] = set()
    for pattern in PLOT_PATTERNS:
        found_plots.update(run_dir.rglob(pattern))
    for plot_path in sorted(found_plots):
        mlflow.log_artifact(str(plot_path), artifact_path="plots")


def log_per_class_results(metrics: Any, names: Any) -> None:
    """Log final YOLO validation precision, recall, F1, and AP for each class."""
    class_result = getattr(metrics, "class_result", None)
    if not callable(class_result) or not isinstance(names, (dict, list, tuple)):
        raise RuntimeError(
            "Ultralytics did not expose final per-class validation metrics and class names."
        )

    items = sorted(names.items()) if isinstance(names, dict) else enumerate(names)
    logged = False
    for class_index, class_name in items:
        values = class_result(int(class_index))
        if len(values) < 5:
            raise RuntimeError(
                f"Ultralytics returned incomplete metrics for class {class_name!r}."
            )
        precision, recall, f1, map50, map50_95 = (float(value) for value in values[:5])
        safe_name = str(class_name).strip().lower().replace(" ", "_")
        mlflow.log_metrics(
            {
                f"class/{safe_name}/precision": precision,
                f"class/{safe_name}/recall": recall,
                f"class/{safe_name}/f1": f1,
                f"class/{safe_name}/map50": map50,
                f"class/{safe_name}/map50_95": map50_95,
            }
        )
        logged = True
    if not logged:
        raise RuntimeError("No per-class validation metrics were available to log.")


def log_per_class_metrics(trainer: Any) -> None:
    """Log the final YOLO validation precision, recall, F1, and AP for each class."""
    validator = getattr(trainer, "validator", None)
    metrics = getattr(validator, "metrics", None)
    data = getattr(trainer, "data", {})
    names = data.get("names") if isinstance(data, dict) else None
    log_per_class_results(metrics, names)


def register_best_model(
    run_id: str,
    registered_model_name: str,
    promote_to_production: bool = False,
) -> int:
    """Register the logged checkpoint and assign staging and optional production aliases."""
    client = MlflowClient()
    try:
        client.get_registered_model(registered_model_name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(registered_model_name)

    run = client.get_run(run_id)
    model_source = f"{run.info.artifact_uri.rstrip('/')}/best_model"
    registered = client.create_model_version(
        name=registered_model_name,
        source=model_source,
        run_id=run_id,
        await_creation_for=300,
    )
    version = int(registered.version)
    client.set_registered_model_alias(registered_model_name, "staging", str(version))
    if promote_to_production:
        client.set_registered_model_alias(
            registered_model_name, "production", str(version)
        )
    return version
