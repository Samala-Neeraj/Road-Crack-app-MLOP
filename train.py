import argparse
import os
from pathlib import Path
from typing import Any

import mlflow

from mlflow_utils import (
    PROJECT_ROOT,
    configure_tracking,
    env_path,
    log_best_model_and_training_artifacts,
    log_per_class_metrics,
    mlflow_metric_name,
    mlflow_value,
    numeric_metrics,
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
        description="Train YOLO11m and record its metrics and model in local MLflow."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the Roboflow-exported data.yaml file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Initial Ultralytics checkpoint (default: yolo11m.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache training images in memory (default: enabled).",
    )
    parser.add_argument("--device", default=None, help="Optional Ultralytics device.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--name", default="road_crack_yolo11m")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--roboflow-version", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--promote-to-production", action="store_true")
    return parser.parse_args()


def _record_epoch_metrics(trainer: Any) -> None:
    validation_metrics = numeric_metrics(getattr(trainer, "metrics", {}) or {})
    losses = getattr(trainer, "tloss", None)
    training_losses: dict[str, float] = {}
    if losses is not None:
        if isinstance(losses, dict):
            training_losses = numeric_metrics(
                {
                    key if "/" in key else f"train/{key}": value
                    for key, value in losses.items()
                }
            )
        else:
            label_loss_items = getattr(trainer, "label_loss_items", None)
            if callable(label_loss_items):
                training_losses = numeric_metrics(
                    label_loss_items(losses, prefix="train")
                )
    loss_names = list(getattr(trainer, "loss_names", ()) or ())
    if isinstance(losses, (list, tuple)) and loss_names:
        training_losses.update(
            numeric_metrics(
                {
                    f"train/{name}": value
                    for name, value in zip(loss_names, losses)
                }
            )
        )

    values = {
        mlflow_metric_name(key): value
        for key, value in {**training_losses, **validation_metrics}.items()
    }
    aliases = {
        "precision": ("precision", "metrics/precision(B)", "metrics/precision"),
        "recall": ("recall", "metrics/recall(B)", "metrics/recall"),
        "map50": ("map50", "metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": (
            "map50_95",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ),
        "box_loss": ("train/box_loss",),
        "cls_loss": ("train/cls_loss",),
        "dfl_loss": ("train/dfl_loss",),
    }
    missing = [
        name
        for name, candidates in aliases.items()
        if not any(candidate in values for candidate in candidates)
    ]
    if missing:
        raise RuntimeError(
            "Cannot log required per-epoch YOLO metrics: "
            + ", ".join(missing)
            + ". Inspect the installed Ultralytics metrics API."
        )

    step = int(getattr(trainer, "epoch", 0)) + 1
    mlflow.log_metrics(values, step=step)


def _log_effective_training_parameters(
    trainer: Any,
    requested_args: argparse.Namespace,
    dataset_name: str,
    roboflow_version: str,
    model_type: str,
) -> None:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("MLflow run ended before training parameters were recorded.")

    effective_args = vars(getattr(trainer, "args", object()))
    params = {
        f"yolo.{key}": mlflow_value(value)
        for key, value in effective_args.items()
    }
    params.update(
        {
            "model_type": model_type,
            "dataset_name": dataset_name,
            "roboflow_version": roboflow_version,
        }
    )
    for key, value in vars(requested_args).items():
        params.setdefault(f"requested.{key}", mlflow_value(value))

    mlflow.log_params({key: value[:500] for key, value in params.items()})
    mlflow.set_tags(
        {
            "dataset_name": dataset_name,
            "roboflow_version": roboflow_version,
            "model_type": model_type,
        }
    )


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking()
    data_yaml = env_path(args.data)
    data_config = yaml_mapping(data_yaml)

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

    if not data_config.get("train") or not data_config.get("val"):
        raise ValueError(
            f"{data_yaml} must define at least the 'train' and 'val' dataset paths."
        )
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must all be positive integers.")

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "cache": args.cache,
        "project": str(env_path(args.project)),
        "name": args.name,
        "plots": True,
    }
    if args.device is not None:
        train_args["device"] = args.device

    lineage_inputs = [
        roboflow_dataset(dataset_name, str(roboflow_version)),
        file_dataset(data_yaml),
        *split_datasets(data_yaml, data_config),
    ]
    lineage_outputs = []
    with lineage_job("yolo-training", lineage_inputs, outputs=lineage_outputs):
        import ultralytics
        from ultralytics import YOLO

        mlflow.set_experiment(experiment_name)
        print(f"MLflow tracking server: {tracking_uri}")
        print(f"Dataset config: {data_yaml}")
        print(f"Ultralytics version: {ultralytics.__version__}")

        with mlflow.start_run(run_name=args.name) as run:
            mlflow.set_tags(
                {
                    "dataset_name": dataset_name,
                    "roboflow_version": str(roboflow_version),
                    "model_type": model_type,
                }
            )
            model = YOLO(args.model)
            model.add_callback("on_fit_epoch_end", _record_epoch_metrics)
            model.train(**train_args)
            trainer = model.trainer
            _log_effective_training_parameters(
                trainer, args, dataset_name, str(roboflow_version), model_type
            )
            log_per_class_metrics(trainer)

            run_directory = Path(trainer.save_dir)
            best_model_path = Path(trainer.best)
            if not best_model_path.is_absolute():
                best_model_path = (run_directory / best_model_path).resolve()
            if not best_model_path.is_file():
                raise FileNotFoundError(
                    "Training finished but best.pt was not found at "
                    f"{best_model_path}"
                )

            log_best_model_and_training_artifacts(
                best_model_path=best_model_path,
                run_directory=run_directory,
                data_yaml_path=data_yaml,
                args_yaml_path=run_directory / "args.yaml",
            )
            version = register_best_model(
                run_id=run.info.run_id,
                registered_model_name=registered_model_name,
                promote_to_production=args.promote_to_production,
            )
            lineage_outputs.extend(
                [
                    output_dataset("file", run_directory.as_posix()),
                    output_dataset("file", best_model_path.as_posix()),
                    output_dataset(
                        "mlflow-registry",
                        f"{registered_model_name}@staging",
                    ),
                ]
            )
            if args.promote_to_production:
                lineage_outputs.append(
                    output_dataset(
                        "mlflow-registry",
                        f"{registered_model_name}@production",
                    )
                )

    print(f"MLflow run ID: {run.info.run_id}")
    print(f"Registered model: {registered_model_name}, version {version}")
    print(f"Model alias: {registered_model_name}@staging")
    if args.promote_to_production:
        print(f"Promoted alias: {registered_model_name}@production")
    print(f"Training artifacts: {run_directory}")


if __name__ == "__main__":
    main()
