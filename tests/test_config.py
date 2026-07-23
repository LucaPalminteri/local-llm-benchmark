from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import ConfigError, load_experiment


def test_load_experiment_resolves_referenced_paths(
    resolved_experiment: object,
) -> None:
    experiment = resolved_experiment
    assert experiment.model.id == "test-model"  # type: ignore[attr-defined]
    assert experiment.model.gguf_path.is_absolute()  # type: ignore[attr-defined]
    assert experiment.runtime.llama_bench_path.is_absolute()  # type: ignore[attr-defined]
    assert experiment.benchmark.output_directory.is_absolute()  # type: ignore[attr-defined]


def test_rejects_micro_batch_larger_than_batch(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: model",
                "display_name: Model",
                "gguf_path: model.gguf",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: runtime",
                "llama_bench_path: llama-bench.exe",
                "batch_size: 128",
                "micro_batch_size: 256",
            ]
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: invalid",
                "model: model.yaml",
                "runtime: runtime.yaml",
                "benchmark:",
                "  prompt_tokens: [512]",
                "  context_depths: [0]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="micro_batch_size cannot exceed batch_size"):
        load_experiment(experiment)
