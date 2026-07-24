from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llmbench.config import (
    ConfigError,
    InteractiveBenchmarkConfig,
    ResolvedExperiment,
    load_experiment,
)
from llmbench.models import InteractiveNormalizedSample
from llmbench.planner import expand_interactive_cases


def _benchmark_document() -> dict[str, Any]:
    return {
        "track": "interactive",
        "mode": "full",
        "server": {
            "host": "127.0.0.1",
            "port": "auto",
            "readiness_timeout_seconds": 300,
            "request_timeout_seconds": 300,
            "context_size": 16384,
        },
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 40,
            "seed": 42,
        },
        "workloads": [
            {"id": "small", "size": "small", "target_prompt_tokens": 500},
            {"id": "medium", "size": "medium", "target_prompt_tokens": 4000},
            {"id": "large", "size": "large", "target_prompt_tokens": 12000},
        ],
        "requested_output_tokens": 128,
        "warmup_requests": 1,
        "repetitions": 10,
        "measure_cold": True,
        "cold_workload_id": "small",
        "measure_warm_uncached": True,
        "measure_warm_cached": True,
        "cached_prefix_ratio": 0.75,
        "output_directory": "../../runs",
    }


def test_committed_interactive_configs_load_and_expand() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke = load_experiment(root / "configs/experiments/interactive-smoke.yaml")
    full = load_experiment(root / "configs/experiments/interactive-full.yaml")

    assert smoke.runtime.llama_server_path is not None
    assert smoke.runtime.llama_server_path.name == "llama-server.exe"
    assert [case.case_id for case in expand_interactive_cases(smoke)] == [
        "small-cold",
        "small-warmup",
        "small-warm-uncached",
        "small-warm-cached",
    ]
    assert [case.repetitions for case in expand_interactive_cases(smoke)] == [1, 1, 2, 2]

    full_cases = expand_interactive_cases(full)
    assert len(full_cases) == 10
    assert sum(case.repetitions for case in full_cases) == 64
    assert [case.phase for case in full_cases[:4]] == [
        "cold",
        "warmup",
        "warm_uncached",
        "warm_cached",
    ]
    assert {case.workload_size for case in full_cases} == {"small", "medium", "large"}
    assert {case.requested_output_tokens for case in full_cases} == {128}


def test_smoke_mode_permits_fewer_than_ten_repetitions() -> None:
    document = _benchmark_document()
    document["mode"] = "smoke"
    document["workloads"] = [document["workloads"][0]]
    document["repetitions"] = 2

    benchmark = InteractiveBenchmarkConfig.model_validate(document)

    assert benchmark.mode == "smoke"
    assert benchmark.repetitions == 2


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"repetitions": 9}, "at least 10 repetitions"),
        ({"measure_warm_uncached": False, "measure_warm_cached": False}, "warm measurement"),
        ({"cold_workload_id": "missing"}, "cold_workload_id"),
        ({"requested_output_tokens": 5000}, "exceeds server context"),
    ],
)
def test_rejects_invalid_interactive_combinations(
    update: dict[str, Any],
    message: str,
) -> None:
    document = _benchmark_document()
    document.update(update)

    with pytest.raises(ValidationError, match=message):
        InteractiveBenchmarkConfig.model_validate(document)


def test_rejects_invalid_port_and_duplicate_workload_size() -> None:
    invalid_port = _benchmark_document()
    invalid_port["server"]["port"] = 0
    with pytest.raises(ValidationError, match="server port"):
        InteractiveBenchmarkConfig.model_validate(invalid_port)

    duplicate_size = deepcopy(_benchmark_document())
    duplicate_size["workloads"][1]["size"] = "small"
    with pytest.raises(ValidationError, match="workload sizes must be unique"):
        InteractiveBenchmarkConfig.model_validate(duplicate_size)


def test_rejects_invalid_timeouts_and_measurement_phase() -> None:
    invalid_timeout = _benchmark_document()
    invalid_timeout["server"]["request_timeout_seconds"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        InteractiveBenchmarkConfig.model_validate(invalid_timeout)

    with pytest.raises(ValidationError, match="warmup"):
        InteractiveNormalizedSample(
            run_id="run",
            case_id="small-invalid",
            repetition=1,
            status="completed",
            model_id="model",
            runtime_id="runtime",
            phase="invalid",  # type: ignore[arg-type]
            workload_id="small",
            workload_size="small",
            target_prompt_tokens=500,
            actual_prompt_tokens=498,
            requested_output_tokens=128,
        )


def test_interactive_experiment_requires_server_executable(
    resolved_experiment: ResolvedExperiment,
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "runtime.yaml"
    runtime_lines = [
        line
        for line in resolved_experiment.runtime_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("llama_server_path:")
    ]
    runtime_path.write_text("\n".join(runtime_lines), encoding="utf-8")
    experiment_path = tmp_path / "interactive.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: missing-server",
                f"model: '{resolved_experiment.model_path}'",
                f"runtime: '{runtime_path}'",
                "benchmark:",
                "  track: interactive",
                "  mode: smoke",
                "  server:",
                "    context_size: 4096",
                "  workloads:",
                "    - id: small",
                "      size: small",
                "      target_prompt_tokens: 500",
                "  requested_output_tokens: 128",
                "  repetitions: 2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="requires runtime llama_server_path"):
        load_experiment(experiment_path)
