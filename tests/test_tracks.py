from __future__ import annotations

from pathlib import Path

import pytest

from llmbench import (
    INTERACTIVE_BENCHMARK_PROTOCOL_VERSION,
    RAW_BENCHMARK_PROTOCOL_VERSION,
)
from llmbench.config import (
    ConfigError,
    InteractiveBenchmarkConfig,
    RawBenchmarkConfig,
    ResolvedExperiment,
    load_experiment,
)
from llmbench.planner import expand_cases
from llmbench.reporting import build_summary, compare_summaries
from llmbench.storage import create_run, read_json, write_json


def _write_experiment(
    path: Path,
    resolved_experiment: ResolvedExperiment,
    benchmark_lines: list[str],
) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"id: {path.stem}",
                f"model: '{resolved_experiment.model_path}'",
                f"runtime: '{resolved_experiment.runtime_path}'",
                "benchmark:",
                *[f"  {line}" for line in benchmark_lines],
            ]
        ),
        encoding="utf-8",
    )


def _interactive_benchmark_lines(output_directory: str) -> list[str]:
    return [
        "track: interactive",
        "mode: smoke",
        "server:",
        "  host: 127.0.0.1",
        "  port: auto",
        "  context_size: 4096",
        "workloads:",
        "  - id: small",
        "    size: small",
        "    target_prompt_tokens: 500",
        "requested_output_tokens: 128",
        "warmup_requests: 1",
        "repetitions: 2",
        "cold_workload_id: small",
        f"output_directory: {output_directory}",
    ]


def test_legacy_experiment_defaults_to_raw_track(
    resolved_experiment: ResolvedExperiment,
) -> None:
    assert resolved_experiment.track == "raw"
    assert isinstance(resolved_experiment.benchmark, RawBenchmarkConfig)


def test_loads_minimal_interactive_experiment(
    resolved_experiment: ResolvedExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "interactive.yaml"
    _write_experiment(
        path,
        resolved_experiment,
        _interactive_benchmark_lines("runs"),
    )

    experiment = load_experiment(path)

    assert experiment.track == "interactive"
    assert isinstance(experiment.benchmark, InteractiveBenchmarkConfig)
    assert experiment.benchmark.output_directory == (tmp_path / "runs").resolve()


@pytest.mark.parametrize(
    ("benchmark_lines", "message"),
    [
        (["track: unsupported"], "union_tag_invalid"),
        (["track: interactive", "prompt_tokens: [512]"], "prompt_tokens"),
    ],
)
def test_rejects_unknown_tracks_and_mixed_track_fields(
    resolved_experiment: ResolvedExperiment,
    tmp_path: Path,
    benchmark_lines: list[str],
    message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    _write_experiment(path, resolved_experiment, benchmark_lines)

    with pytest.raises(ConfigError, match=message):
        load_experiment(path)


def test_manifests_store_track_specific_protocols(
    resolved_experiment: ResolvedExperiment,
    tmp_path: Path,
) -> None:
    raw_case = expand_cases(resolved_experiment)[0]
    _, raw_paths = create_run(
        resolved_experiment,
        [raw_case],
        {raw_case.case_id: ["llama-bench.exe"]},
    )
    raw_manifest = read_json(raw_paths.manifest)
    assert raw_manifest["benchmark_track"] == "raw"
    assert raw_manifest["benchmark_protocol_version"] == RAW_BENCHMARK_PROTOCOL_VERSION

    path = tmp_path / "interactive.yaml"
    _write_experiment(
        path,
        resolved_experiment,
        _interactive_benchmark_lines("interactive-runs"),
    )
    interactive = load_experiment(path)
    _, interactive_paths = create_run(interactive, [], {})
    interactive_manifest = read_json(interactive_paths.manifest)
    assert interactive_manifest["benchmark_track"] == "interactive"
    assert (
        interactive_manifest["benchmark_protocol_version"] == INTERACTIVE_BENCHMARK_PROTOCOL_VERSION
    )
    assert interactive_manifest["cases"] == []


def test_old_raw_manifest_remains_reportable(
    resolved_experiment: ResolvedExperiment,
) -> None:
    case = expand_cases(resolved_experiment)[0]
    _, paths = create_run(
        resolved_experiment,
        [case],
        {case.case_id: ["llama-bench.exe"]},
    )
    manifest = read_json(paths.manifest)
    manifest.pop("benchmark_track")
    manifest["experiment"]["benchmark"].pop("track")
    write_json(paths.manifest, manifest)

    summary = build_summary(paths)

    assert summary["benchmark_track"] == "raw"
    assert summary["benchmark_protocol_version"] == RAW_BENCHMARK_PROTOCOL_VERSION


def test_raw_comparison_rejects_interactive_track() -> None:
    raw = {
        "benchmark_track": "raw",
        "benchmark_protocol_version": RAW_BENCHMARK_PROTOCOL_VERSION,
        "cases": [],
    }
    interactive = {
        "benchmark_track": "interactive",
        "benchmark_protocol_version": INTERACTIVE_BENCHMARK_PROTOCOL_VERSION,
        "cases": [],
    }

    with pytest.raises(ValueError, match="different benchmark tracks"):
        compare_summaries([raw, interactive])
