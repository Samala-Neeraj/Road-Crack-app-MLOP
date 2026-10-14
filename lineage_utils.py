import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Generator
from uuid import uuid4

from dotenv import load_dotenv
from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import Job, Run, RunEvent, RunState
from openlineage.client.generated.base import InputDataset, OutputDataset
from openlineage.client.generated.error_message_run import ErrorMessageRunFacet
from openlineage.client.transport.file import FileConfig, FileTransport


PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCER = (
    "https://github.com/OpenLineage/OpenLineage/tree/1.54.0/client/python"
)
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def _client() -> OpenLineageClient:
    event_directory = _resolve_path(
        os.getenv("OPENLINEAGE_EVENT_DIR", "lineage/events")
    )
    event_directory.mkdir(parents=True, exist_ok=True)
    event_file = event_directory / "openlineage.jsonl"
    return OpenLineageClient(
        transport=FileTransport(
            FileConfig(log_file_path=str(event_file), append=True)
        )
    )


def file_dataset(path: str | Path) -> InputDataset:
    resolved_path = _resolve_path(path)
    return InputDataset(namespace="file", name=resolved_path.as_posix())


def input_dataset(namespace: str, name: str) -> InputDataset:
    return InputDataset(namespace=namespace, name=name)


def output_dataset(namespace: str, name: str) -> OutputDataset:
    return OutputDataset(namespace=namespace, name=name)


def roboflow_dataset(
    dataset_name: str | None = None,
    version: str | None = None,
) -> InputDataset:
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "neeraj-samala")
    dataset_name = dataset_name or os.getenv(
        "DATASET_NAME", "road-crack-detection-severity"
    )
    version = version or os.getenv("ROBOFLOW_VERSION", "1")
    return InputDataset(
        namespace="roboflow",
        name=f"{workspace}/{dataset_name}/version/{version}",
    )


def split_datasets(
    data_yaml: str | Path, data_config: dict[str, object]
) -> list[InputDataset]:
    yaml_path = _resolve_path(data_yaml)
    configured_root = data_config.get("path")
    if isinstance(configured_root, str):
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent

    datasets: list[InputDataset] = []
    for split in ("train", "val", "test"):
        split_paths = data_config.get(split)
        if isinstance(split_paths, str):
            values = [split_paths]
        elif isinstance(split_paths, list):
            values = [value for value in split_paths if isinstance(value, str)]
        else:
            continue

        for value in values:
            split_path = Path(value).expanduser()
            if not split_path.is_absolute():
                split_path = root / split_path
            datasets.append(
                InputDataset(
                    namespace="file",
                    name=f"{split}:{split_path.resolve().as_posix()}",
                )
            )
    return datasets


def _emit_event(
    *,
    event_type: RunState,
    run_id: str,
    job_name: str,
    inputs: list[InputDataset],
    outputs: list[OutputDataset],
    error: Exception | None = None,
) -> None:
    facets = {}
    if error is not None:
        facets["errorMessage"] = ErrorMessageRunFacet(
            message=str(error),
            programmingLanguage="Python",
            stackTrace="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            producer=PRODUCER,
        )

    event = RunEvent(
        eventTime=(
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
        producer=PRODUCER,
        eventType=event_type,
        run=Run(runId=run_id, facets=facets),
        job=Job(
            namespace=os.getenv(
                "OPENLINEAGE_NAMESPACE", "road-crack-detection"
            ),
            name=job_name,
        ),
        inputs=inputs,
        outputs=outputs,
    )
    _client().emit(event)


@contextmanager
def lineage_job(
    job_name: str,
    inputs: list[InputDataset],
    outputs: list[OutputDataset] | None = None,
) -> Generator[str, None, None]:
    run_id = str(uuid4())
    final_outputs = outputs if outputs is not None else []
    _emit_event(
        event_type=RunState.START,
        run_id=run_id,
        job_name=job_name,
        inputs=inputs,
        outputs=[],
    )
    try:
        yield run_id
    except Exception as error:
        try:
            _emit_event(
                event_type=RunState.FAIL,
                run_id=run_id,
                job_name=job_name,
                inputs=inputs,
                outputs=[],
                error=error,
            )
        except Exception as lineage_error:
            error.add_note(
                f"Could not write the OpenLineage FAIL event: {lineage_error}"
            )
        raise
    else:
        _emit_event(
            event_type=RunState.COMPLETE,
            run_id=run_id,
            job_name=job_name,
            inputs=inputs,
            outputs=final_outputs,
        )
