from __future__ import annotations

import json

import pytest

from llmbench.config import ResolvedExperiment
from llmbench.llama_bench import (
    LlamaBenchParseError,
    normalize_records,
    parse_json_output,
)
from llmbench.planner import expand_cases


def _generation_record() -> dict[str, object]:
    return {
        "build_commit": "abc123",
        "build_number": 6000,
        "cpu_info": "Test CPU",
        "gpu_info": "AMD Radeon RX 6700 XT",
        "backends": "Vulkan",
        "model_filename": "model.gguf",
        "model_type": "test 7B Q4_K_M",
        "model_size": 4_000_000_000,
        "model_n_params": 7_000_000_000,
        "n_batch": 512,
        "n_ubatch": 256,
        "n_threads": 4,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 99,
        "flash_attn": 1,
        "n_prompt": 0,
        "n_gen": 128,
        "n_depth": 0,
        "test_time": "2026-07-23T12:00:00Z",
        "samples_ns": [5_000_000_000, 5_100_000_000, 4_900_000_000],
        "samples_ts": [25.6, 25.1, 26.1],
    }


def test_parses_json_and_normalizes_individual_repetitions(
    resolved_experiment: ResolvedExperiment,
) -> None:
    records = parse_json_output(json.dumps([_generation_record()]))
    generation_case = expand_cases(resolved_experiment)[1]

    samples = normalize_records(
        records,
        run_id="run-1",
        experiment=resolved_experiment,
        case=generation_case,
    )

    assert len(samples) == 3
    assert [sample.repetition for sample in samples] == [1, 2, 3]
    assert [sample.tokens_per_second for sample in samples] == [25.6, 25.1, 26.1]
    assert samples[0].build_commit == "abc123"
    assert samples[0].backend == "Vulkan"


def test_parses_jsonl() -> None:
    rows = [_generation_record(), {**_generation_record(), "n_depth": 2048}]
    output = "\n".join(json.dumps(row) for row in rows)

    assert parse_json_output(output) == rows


def test_rejects_aggregate_without_individual_samples(
    resolved_experiment: ResolvedExperiment,
) -> None:
    record = _generation_record()
    del record["samples_ts"]
    generation_case = expand_cases(resolved_experiment)[1]

    with pytest.raises(LlamaBenchParseError, match="individual samples_ts"):
        normalize_records(
            [record],
            run_id="run-1",
            experiment=resolved_experiment,
            case=generation_case,
        )
