from __future__ import annotations

import json
import os
import platform
import socket
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel

from llmbench import APP_VERSION, RESULT_SCHEMA_VERSION
from llmbench.config import ResolvedExperiment, benchmark_protocol_version
from llmbench.models import NormalizedSampleBase


class RunCase(Protocol):
    @property
    def case_id(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class RunPaths:
    root: Path
    raw: Path
    raw_cases: Path
    logs: Path
    manifest: Path
    environment: Path
    events: Path
    samples: Path
    summary: Path
    combined_raw: Path
    stdout_log: Path
    stderr_log: Path


def paths_for_run(root: Path) -> RunPaths:
    return RunPaths(
        root=root,
        raw=root / "raw",
        raw_cases=root / "raw" / "cases",
        logs=root / "logs",
        manifest=root / "manifest.json",
        environment=root / "environment.json",
        events=root / "events.jsonl",
        samples=root / "samples.jsonl",
        summary=root / "summary.json",
        combined_raw=root / "raw" / "llama-bench.json",
        stdout_log=root / "logs" / "stdout.log",
        stderr_log=root / "logs" / "stderr.log",
    )


def new_run_id(model_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{model_id}-{uuid4().hex[:8]}"


def create_run(
    experiment: ResolvedExperiment,
    cases: Sequence[RunCase],
    commands: Mapping[str, list[str]],
) -> tuple[str, RunPaths]:
    run_id = new_run_id(experiment.model.id)
    paths = paths_for_run(experiment.benchmark.output_directory / run_id)
    paths.raw_cases.mkdir(parents=True, exist_ok=False)
    paths.logs.mkdir(parents=True, exist_ok=False)

    write_json(
        paths.manifest,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark_track": experiment.track,
            "benchmark_protocol_version": benchmark_protocol_version(experiment.track),
            "application_version": APP_VERSION,
            "run_id": run_id,
            "created_at": utc_now(),
            "experiment": {
                "id": experiment.id,
                "source": str(experiment.experiment_path),
                "model_source": str(experiment.model_path),
                "runtime_source": str(experiment.runtime_path),
                "model": experiment.model.model_dump(mode="json"),
                "runtime": experiment.runtime.model_dump(mode="json"),
                "benchmark": experiment.benchmark.model_dump(mode="json"),
            },
            "cases": [{**case.to_dict(), "command": commands[case.case_id]} for case in cases],
        },
    )
    write_json(paths.environment, collect_environment())
    append_event(paths, "run_created", {"case_count": len(cases)})
    return run_id, paths


def collect_environment() -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "application_version": APP_VERSION,
    }


def write_json(path: Path, value: Any) -> None:
    write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, values: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for value in values:
            payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            output.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        values.append(value)
    return values


def append_event(paths: RunPaths, event_type: str, data: dict[str, Any] | None = None) -> None:
    append_jsonl(
        paths.events,
        [
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "timestamp": utc_now(),
                "event": event_type,
                "data": data or {},
            }
        ],
    )


def append_process_log(path: Path, case_id: str, contents: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"===== {case_id} =====\n")
        output.write(contents)
        if contents and not contents.endswith("\n"):
            output.write("\n")


def store_case_samples(paths: RunPaths, samples: Iterable[NormalizedSampleBase]) -> None:
    append_jsonl(paths.samples, samples)


def terminal_case_ids(paths: RunPaths) -> set[str]:
    return {str(row["case_id"]) for row in read_jsonl(paths.samples)}


def rebuild_combined_raw(paths: RunPaths) -> None:
    combined: list[dict[str, Any]] = []
    for path in sorted(paths.raw_cases.glob("*.json")):
        try:
            value = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            combined.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            combined.append(value)
    write_json(paths.combined_raw, combined)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
