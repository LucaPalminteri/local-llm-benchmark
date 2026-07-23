from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llmbench.config import ResolvedExperiment


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str


def run_doctor(experiment: ResolvedExperiment) -> list[DoctorCheck]:
    checks = [
        _file_check(
            "llama-bench executable",
            experiment.runtime.llama_bench_path,
            expected_suffix=".exe" if os.name == "nt" else None,
        ),
        _file_check("GGUF model", experiment.model.gguf_path, expected_suffix=".gguf"),
        _output_check(experiment.benchmark.output_directory),
    ]

    if experiment.runtime.llama_bench_path.is_file():
        checks.append(
            _device_check(
                experiment.runtime.llama_bench_path,
                experiment.runtime.backend,
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="device discovery",
                passed=False,
                detail="not attempted because llama-bench was not found",
            )
        )

    if experiment.runtime.expected_commit:
        checks.append(
            DoctorCheck(
                name="llama.cpp commit",
                passed=True,
                detail=(
                    f"expected {experiment.runtime.expected_commit}; "
                    "the first benchmark result will verify it"
                ),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="llama.cpp commit",
                passed=True,
                detail="not pinned; set expected_commit for strict reproducibility",
            )
        )
    return checks


def _file_check(name: str, path: Path, *, expected_suffix: str | None = None) -> DoctorCheck:
    if not path.is_file():
        return DoctorCheck(name=name, passed=False, detail=f"not found: {path}")
    if expected_suffix and path.suffix.lower() != expected_suffix:
        return DoctorCheck(
            name=name,
            passed=False,
            detail=f"expected a {expected_suffix} file: {path}",
        )
    return DoctorCheck(name=name, passed=True, detail=str(path))


def _output_check(path: Path) -> DoctorCheck:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK)
    return DoctorCheck(
        name="output directory",
        passed=writable,
        detail=(
            f"{path} (nearest existing directory: {ancestor})"
            if writable
            else f"not writable: {path}"
        ),
    )


def _device_check(executable: Path, backend: str) -> DoctorCheck:
    try:
        result = subprocess.run(
            [str(executable), "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return DoctorCheck("device discovery", False, str(error))

    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        return DoctorCheck(
            "device discovery",
            False,
            f"llama-bench exited with {result.returncode}: {combined}",
        )
    backend_found = backend.lower() in combined.lower()
    return DoctorCheck(
        "device discovery",
        backend_found,
        combined or f"no devices reported; expected backend {backend}",
    )
