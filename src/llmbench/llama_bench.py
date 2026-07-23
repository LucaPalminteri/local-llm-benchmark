from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from llmbench.config import ResolvedExperiment
from llmbench.models import NormalizedSample
from llmbench.planner import BenchCase


class LlamaBenchParseError(ValueError):
    """Raised when llama-bench output cannot be normalized."""


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None


def build_command(experiment: ResolvedExperiment, case: BenchCase) -> list[str]:
    runtime = experiment.runtime
    benchmark = experiment.benchmark

    command = [
        str(runtime.llama_bench_path),
        "-m",
        str(experiment.model.gguf_path),
        "-ngl",
        str(runtime.gpu_layers),
        "-fa",
        runtime.flash_attention,
        "-ctk",
        runtime.kv_cache.key,
        "-ctv",
        runtime.kv_cache.value,
        "-b",
        str(runtime.batch_size),
        "-ub",
        str(runtime.micro_batch_size),
        "-d",
        str(case.context_depth),
        "-r",
        str(benchmark.repetitions),
        "-o",
        "json",
    ]

    if runtime.threads is not None:
        command.extend(["-t", str(runtime.threads)])
    if runtime.delay_seconds:
        command.extend(["--delay", str(runtime.delay_seconds)])

    if case.test_type == "prompt":
        command.extend(["-p", str(case.prompt_tokens), "-n", "0"])
    elif case.test_type == "generation":
        command.extend(["-p", "0", "-n", str(case.generation_tokens)])
    else:
        command.extend(["-pg", f"{case.prompt_tokens},{case.generation_tokens}"])

    return command


def execute_command(command: list[str], timeout_seconds: float) -> ProcessResult:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return ProcessResult(
            command=command,
            exit_code=None,
            stdout=_timeout_text(error.stdout),
            stderr=_timeout_text(error.stderr),
            timed_out=True,
            error=f"llama-bench exceeded the {timeout_seconds:g} second timeout",
        )
    except OSError as error:
        return ProcessResult(
            command=command,
            exit_code=None,
            stdout="",
            stderr="",
            timed_out=False,
            error=str(error),
        )

    return ProcessResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_json_output(output: str) -> list[dict[str, Any]]:
    stripped = output.strip()
    if not stripped:
        raise LlamaBenchParseError("llama-bench produced empty JSON output")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        try:
            for line in stripped.splitlines():
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise LlamaBenchParseError("each JSONL row must be an object")
                    records.append(item)
        except json.JSONDecodeError as error:
            raise LlamaBenchParseError(
                f"llama-bench output is neither valid JSON nor JSONL: {error}"
            ) from error
        if not records:
            raise LlamaBenchParseError("llama-bench JSONL output contained no rows") from None
        return records

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise LlamaBenchParseError("llama-bench JSON must be an object or array of objects")


def normalize_records(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    experiment: ResolvedExperiment,
    case: BenchCase,
) -> list[NormalizedSample]:
    matching = [
        record
        for record in records
        if int(record.get("n_prompt", -1)) == case.prompt_tokens
        and int(record.get("n_gen", -1)) == case.generation_tokens
        and int(record.get("n_depth", -1)) == case.context_depth
    ]
    if len(matching) != 1:
        raise LlamaBenchParseError(f"expected one result for {case.case_id}, found {len(matching)}")

    record = matching[0]
    samples_ts = record.get("samples_ts")
    samples_ns = record.get("samples_ns")
    if not isinstance(samples_ts, list) or not samples_ts:
        raise LlamaBenchParseError("result does not contain individual samples_ts")
    if not isinstance(samples_ns, list) or len(samples_ns) != len(samples_ts):
        raise LlamaBenchParseError("result samples_ns must match the number of samples_ts")

    if len(samples_ts) != experiment.benchmark.repetitions:
        raise LlamaBenchParseError(
            "result repetition count does not match the configured repetitions"
        )

    raw_metadata = {
        key: record.get(key)
        for key in (
            "build_number",
            "cpu_info",
            "gpu_info",
            "model_size",
            "model_n_params",
            "n_batch",
            "n_ubatch",
            "n_threads",
            "flash_attn",
            "test_time",
        )
        if key in record
    }

    return [
        NormalizedSample(
            run_id=run_id,
            case_id=case.case_id,
            repetition=index,
            status="completed",
            test_type=case.test_type,
            prompt_tokens=case.prompt_tokens,
            generation_tokens=case.generation_tokens,
            context_depth=case.context_depth,
            tokens_per_second=float(tokens_per_second),
            duration_ns=int(samples_ns[index - 1]),
            model_id=experiment.model.id,
            runtime_id=experiment.runtime.id,
            model_filename=_optional_string(record.get("model_filename")),
            model_type=_optional_string(record.get("model_type")),
            build_commit=_optional_string(record.get("build_commit")),
            backend=_optional_string(record.get("backends")),
            kv_cache_key=experiment.runtime.kv_cache.key,
            kv_cache_value=experiment.runtime.kv_cache.value,
            gpu_layers=experiment.runtime.gpu_layers,
            raw_metadata=raw_metadata,
        )
        for index, tokens_per_second in enumerate(samples_ts, start=1)
    ]


def failure_sample(
    *,
    run_id: str,
    experiment: ResolvedExperiment,
    case: BenchCase,
    kind: str,
    message: str,
) -> NormalizedSample:
    return NormalizedSample(
        run_id=run_id,
        case_id=case.case_id,
        repetition=None,
        status="failed",
        failure_kind=kind,
        failure_message=message,
        test_type=case.test_type,
        prompt_tokens=case.prompt_tokens,
        generation_tokens=case.generation_tokens,
        context_depth=case.context_depth,
        model_id=experiment.model.id,
        runtime_id=experiment.runtime.id,
        kv_cache_key=experiment.runtime.kv_cache.key,
        kv_cache_value=experiment.runtime.kv_cache.value,
        gpu_layers=experiment.runtime.gpu_layers,
    )


def validate_expected_commit(
    experiment: ResolvedExperiment, samples: list[NormalizedSample]
) -> None:
    expected = experiment.runtime.expected_commit
    if expected is None:
        return
    observed = {sample.build_commit for sample in samples}
    matches = all(
        value is not None and (value.startswith(expected) or expected.startswith(value))
        for value in observed
    )
    if not matches:
        values = ", ".join(sorted(value or "<missing>" for value in observed))
        raise LlamaBenchParseError(f"expected llama.cpp commit {expected}, observed {values}")


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
