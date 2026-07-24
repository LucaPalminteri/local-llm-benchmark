from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import ResolvedExperiment, load_experiment


@pytest.fixture()
def resolved_experiment(tmp_path: Path) -> ResolvedExperiment:
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    executable = tmp_path / "llama-bench.exe"
    executable.touch()
    server_executable = tmp_path / "llama-server.exe"
    server_executable.touch()

    models = tmp_path / "models"
    profiles = tmp_path / "profiles"
    experiments = tmp_path / "experiments"
    models.mkdir()
    profiles.mkdir()
    experiments.mkdir()

    (models / "model.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: test-model",
                "display_name: Test Model",
                f"gguf_path: '{model_file}'",
                "metadata:",
                "  quantization: Q4_K_M",
            ]
        ),
        encoding="utf-8",
    )
    (profiles / "runtime.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: test-runtime",
                f"llama_bench_path: '{executable}'",
                f"llama_server_path: '{server_executable}'",
                "expected_commit: abc123",
                "backend: Vulkan",
                "gpu_layers: 99",
                'flash_attention: "on"',
                "kv_cache:",
                "  key: f16",
                "  value: f16",
                "threads: 4",
                "batch_size: 512",
                "micro_batch_size: 256",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path = experiments / "raw.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: test-experiment",
                "model: ../models/model.yaml",
                "runtime: ../profiles/runtime.yaml",
                "benchmark:",
                "  prompt_tokens: [512]",
                "  generation_tokens: [128, 256]",
                "  prompt_generation_pairs:",
                "    - prompt: 64",
                "      generation: 16",
                "  context_depths: [0, 2048]",
                "  repetitions: 3",
                "  timeout_seconds: 60",
                "  output_directory: ../runs",
            ]
        ),
        encoding="utf-8",
    )
    return load_experiment(experiment_path)
