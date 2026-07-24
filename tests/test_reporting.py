from __future__ import annotations

import pytest

from llmbench.config import ResolvedExperiment
from llmbench.llama_bench import failure_sample
from llmbench.models import NormalizedSample
from llmbench.planner import expand_cases
from llmbench.reporting import build_summary, compare_summaries, refresh_reports
from llmbench.storage import create_run, store_case_samples


def test_calculates_statistics_from_individual_samples(
    resolved_experiment: ResolvedExperiment,
) -> None:
    case = expand_cases(resolved_experiment)[0]
    command = ["llama-bench.exe", "-p", "512"]
    _, paths = create_run(
        resolved_experiment,
        [case],
        {case.case_id: command},
    )
    samples = [
        NormalizedSample(
            run_id=paths.root.name,
            case_id=case.case_id,
            repetition=index,
            status="completed",
            test_type=case.test_type,
            prompt_tokens=case.prompt_tokens,
            generation_tokens=case.generation_tokens,
            context_depth=case.context_depth,
            tokens_per_second=value,
            duration_ns=100,
            model_id=resolved_experiment.model.id,
            runtime_id=resolved_experiment.runtime.id,
            kv_cache_key="f16",
            kv_cache_value="f16",
            gpu_layers=99,
        )
        for index, value in enumerate([10.0, 20.0, 30.0], 1)
    ]
    assert all(sample.track == "raw" for sample in samples)
    store_case_samples(paths, samples)

    summary = refresh_reports(paths)
    result = summary["cases"][0]

    assert result["median_tokens_per_second"] == 20.0
    assert result["mean_tokens_per_second"] == 20.0
    assert result["stddev_tokens_per_second"] == 10.0
    assert result["minimum_tokens_per_second"] == 10.0
    assert result["maximum_tokens_per_second"] == 30.0
    assert paths.summary.is_file()
    assert (paths.root / "report.md").is_file()
    assert (paths.root / "report.csv").is_file()


def test_failed_case_remains_in_summary(
    resolved_experiment: ResolvedExperiment,
) -> None:
    case = expand_cases(resolved_experiment)[0]
    _, paths = create_run(
        resolved_experiment,
        [case],
        {case.case_id: ["llama-bench.exe"]},
    )
    store_case_samples(
        paths,
        [
            failure_sample(
                run_id=paths.root.name,
                experiment=resolved_experiment,
                case=case,
                kind="timeout",
                message="timed out",
            )
        ],
    )

    summary = build_summary(paths)

    assert summary["status"] == "completed_with_failures"
    assert summary["failed_case_count"] == 1
    assert summary["cases"][0]["failure_kind"] == "timeout"


def test_refuses_different_protocol_versions() -> None:
    first = {"benchmark_protocol_version": "raw-v1", "cases": []}
    second = {"benchmark_protocol_version": "raw-v2", "cases": []}

    with pytest.raises(ValueError, match="different benchmark protocol"):
        compare_summaries([first, second])


def test_refuses_different_case_matrices() -> None:
    first = {
        "benchmark_protocol_version": "raw-v1",
        "cases": [
            {
                "test_type": "generation",
                "prompt_tokens": 0,
                "generation_tokens": 128,
                "context_depth": 0,
            }
        ],
    }
    second = {
        "benchmark_protocol_version": "raw-v1",
        "cases": [
            {
                "test_type": "generation",
                "prompt_tokens": 0,
                "generation_tokens": 256,
                "context_depth": 0,
            }
        ],
    }

    with pytest.raises(ValueError, match="different benchmark case matrices"):
        compare_summaries([first, second])
