from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

from llmbench.config import ResolvedExperiment
from llmbench.llama_bench import ProcessResult
from llmbench.runner import run_raw_experiment
from llmbench.storage import read_jsonl


def test_run_writes_samples_and_resume_skips_terminal_case(
    resolved_experiment: ResolvedExperiment,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    benchmark = resolved_experiment.benchmark.model_copy(
        update={
            "prompt_tokens": [],
            "generation_tokens": [128],
            "prompt_generation_pairs": [],
            "context_depths": [0],
            "output_directory": tmp_path / "runs",
        }
    )
    experiment = replace(resolved_experiment, benchmark=benchmark)

    def fake_execute(command: list[str], timeout_seconds: float) -> ProcessResult:
        assert timeout_seconds == 60
        record = {
            "build_commit": "abc123",
            "backends": "Vulkan",
            "model_filename": "model.gguf",
            "model_type": "test model",
            "n_prompt": 0,
            "n_gen": 128,
            "n_depth": 0,
            "samples_ns": [100, 110, 90],
            "samples_ts": [20.0, 19.0, 21.0],
        }
        return ProcessResult(command, 0, json.dumps([record]), "", False)

    monkeypatch.setattr("llmbench.runner.execute_command", fake_execute)
    first = run_raw_experiment(experiment)

    assert first.summary["status"] == "completed"
    assert len(read_jsonl(first.paths.samples)) == 3
    assert first.paths.combined_raw.is_file()

    def unexpected_execute(command: list[str], timeout_seconds: float) -> ProcessResult:
        raise AssertionError("a terminal case must not execute during resume")

    monkeypatch.setattr("llmbench.runner.execute_command", unexpected_execute)
    resumed = run_raw_experiment(experiment, resume_run_id=first.run_id)

    assert resumed.run_id == first.run_id
    assert len(read_jsonl(resumed.paths.samples)) == 3
