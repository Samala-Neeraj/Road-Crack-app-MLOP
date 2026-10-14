import argparse
import os
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow_utils import (
    configure_tracking,
    env_path,
    log_best_model_and_training_artifacts,
    log_per_class_results,
    log_results_csv,
    mlflow_value,
    register_best_model,
    yaml_mapping,
)
from lineage_utils import (
    file_dataset,
    lineage_job,
    output_dataset,
    roboflow_dataset,
    split_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import an existing Ultralytics Colab run using results.csv, "
            "best.pt, and args.yaml."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Folder containing the Colab results.csv, best.pt, and args.yaml.",
    )
    parser.add_argument("--data-yaml", help="Optional dataset data.yaml to log.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--roboflow-version", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--run-name", default="colab-road-crack-yolo11m")
    parser.add_argument(
        "--evaluate-per-class",
        action="store_true",
        help="Run local validation to compute per-class metrics from --data-yaml.",
    )
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Assign both staging and production aliases to this imported version.",
    )
    return parser.parse_args()


def _read_args_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Ultralytics args.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def main() -> None:
    args = parse_args()
    run_dir = env_path(args.run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Colab run directory not found: {run_dir}")

    results_csv = run_dir / "results.csv"
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.is_file():
        best_model_path = run_dir / "best.pt"
    args_yaml_path = run_dir / "args.yaml"
    training_args = _read_args_yaml(args_yaml_path)
    if not results_csv.is_file():
        raise FileNotFoundError(f"YOLO results.csv not found: {results_csv}")
    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Colab best.pt not found in {run_dir}\\weights or {run_dir}"
        )

    tracking_uri = configure_tracking()
    dataset_name = (
        args.dataset_name
        or os.getenv("DATASET_NAME")
        or "road-crack-detection-severity"
    )
    roboflow_version = (
        args.roboflow_version or os.getenv("ROBOFLOW_VERSION") or "1"
    )
    model_type = args.model_type or os.getenv("MODEL_TYPE") or "yolo11m"
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME", "road-crack-yolo11m"
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "road-crack-yolo11m"
    )
    data_yaml = env_path(args.data_yaml) if args.data_yaml else None
    if args.evaluate_per_class and data_yaml is None:
        raise ValueError("--evaluate-per-class requires --data-yaml.")

    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking server: {tracking_uri}")
    print(f"Importing Colab run from: {run_dir}")

    lineage_inputs = [
        roboflow_dataset(dataset_name, str(roboflow_version)),
        file_dataset(run_dir),
        file_dataset(results_csv),
        file_dataset(args_yaml_path),
    ]
    if data_yaml is not None:
        lineage_inputs.extend(
            [
                file_dataset(data_yaml),
                *split_datasets(data_yaml, yaml_mapping(data_yaml)),
            ]
        )
    lineage_outputs = [
        output_dataset("file", best_model_path.resolve().as_posix()),
        output_dataset(
            "mlflow-registry", f"{registered_model_name}@staging"
        ),
    ]
    if args.promote_to_production:
        lineage_outputs.append(
            output_dataset(
                "mlflow-registry", f"{registered_model_name}@production"
            )
        )

    with lineage_job(
        "yolo-training-colab-retroactive",
        lineage_inputs,
        outputs=lineage_outputs,
    ):
        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.set_tags(
                {
                    "source": "google-colab-retroactive",
                    "dataset_name": dataset_name,
                    "roboflow_version": str(roboflow_version),
                    "model_type": model_type,
                }
            )
            params = {
                f"colab.{key}": mlflow_value(value)[:500]
                for key, value in training_args.items()
            }
            params.update(
                {
                    "dataset_name": dataset_name,
                    "roboflow_version": str(roboflow_version),
                    "model_type": model_type,
                }
            )
            mlflow.log_params(params)
            final_metrics = log_results_csv(results_csv)
            if args.evaluate_per_class:
                from ultralytics import YOLO

                validation_model = YOLO(str(best_model_path))
                validation_results = validation_model.val(
                    data=str(data_yaml),
                    plots=True,
                    project=str(run_dir / "mlflow_revalidation"),
                    name="per_class_metrics",
                )
                log_per_class_results(validation_results, validation_model.names)
            log_best_model_and_training_artifacts(
                best_model_path=best_model_path,
                run_directory=run_dir,
                data_yaml_path=data_yaml,
                args_yaml_path=args_yaml_path,
            )

            version = register_best_model(
                run_id=run.info.run_id,
                registered_model_name=registered_model_name,
                promote_to_production=args.promote_to_production,
            )
            missing_plots = [
                pattern
                for pattern in ("confusion_matrix.png", "BoxPR_curve.png")
                if not list(run_dir.rglob(pattern))
            ]

    print(f"Imported MLflow run ID: {run.info.run_id}")
    print(f"Imported {len(final_metrics)} final-epoch metric(s) from results.csv.")
    print(f"Registered model: {registered_model_name}, version {version}")
    print(f"Model alias: {registered_model_name}@staging")
    if args.promote_to_production:
        print(f"Promoted alias: {registered_model_name}@production")
    if missing_plots:
        print(
            "Not found in the Colab run folder (not fabricated): "
            + ", ".join(missing_plots)
        )
    if data_yaml is None:
        print(
            "No data.yaml was logged. Pass --data-yaml with the local Roboflow "
            "config if you want it attached to this imported run."
        )


if __name__ == "__main__":
    main()
