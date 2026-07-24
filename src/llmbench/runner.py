from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmbench.config import ResolvedExperiment, require_raw_benchmark
from llmbench.llama_bench import (
    LlamaBenchParseError,
    build_command,
    execute_command,
    failure_sample,
    normalize_records,
    parse_json_output,
    validate_expected_commit,
)
from llmbench.models import NormalizedSample
from llmbench.planner import BenchCase, expand_cases
from llmbench.reporting import refresh_reports
from llmbench.storage import (
    RunPaths,
    append_event,
    append_process_log,
    create_run,
    paths_for_run,
    read_json,
    rebuild_combined_raw,
    store_case_samples,
    terminal_case_ids,
    utc_now,
    write_json,
    write_text,
)


class ResumeError(ValueError):
    """Raised when a run cannot safely be resumed."""


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    paths: RunPaths
    summary: dict[str, Any]


def run_raw_experiment(
    experiment: ResolvedExperiment, *, resume_run_id: str | None = None
) -> RunOutcome:
    benchmark = require_raw_benchmark(experiment)
    cases = expand_cases(experiment)
    commands = {case.case_id: build_command(experiment, case) for case in cases}

    if resume_run_id is None:
        run_id, paths = create_run(experiment, cases, commands)
    else:
        run_id = resume_run_id
        paths = paths_for_run(benchmark.output_directory / run_id)
        _validate_resume(paths, experiment, cases, commands)
        append_event(paths, "run_resumed")

    finished_cases = terminal_case_ids(paths)
    try:
        for case in cases:
            if case.case_id in finished_cases:
                append_event(paths, "case_skipped", {"case_id": case.case_id})
                continue
            _run_case(
                run_id=run_id,
                paths=paths,
                experiment=experiment,
                case=case,
                command=commands[case.case_id],
            )
            rebuild_combined_raw(paths)
            refresh_reports(paths)
    except KeyboardInterrupt:
        append_event(paths, "run_interrupted")
        rebuild_combined_raw(paths)
        refresh_reports(paths)
        raise KeyboardInterrupt from None

    rebuild_combined_raw(paths)
    summary = refresh_reports(paths)
    append_event(paths, "run_finished", {"status": summary["status"]})
    return RunOutcome(run_id=run_id, paths=paths, summary=summary)


def _run_case(
    *,
    run_id: str,
    paths: RunPaths,
    experiment: ResolvedExperiment,
    case: BenchCase,
    command: list[str],
) -> None:
    benchmark = require_raw_benchmark(experiment)
    append_event(
        paths,
        "case_started",
        {"case_id": case.case_id, "command": command, "started_at": utc_now()},
    )
    result = execute_command(command, benchmark.timeout_seconds)

    raw_path = paths.raw_cases / f"{case.case_id}.json"
    write_text(raw_path, result.stdout)
    write_text(paths.logs / f"{case.case_id}.stderr.log", result.stderr)
    append_process_log(paths.stdout_log, case.case_id, result.stdout)
    append_process_log(paths.stderr_log, case.case_id, result.stderr)
    write_json(
        paths.logs / f"{case.case_id}.process.json",
        {
            "command": result.command,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "error": result.error,
        },
    )

    samples: list[NormalizedSample]
    if result.timed_out:
        samples = [
            failure_sample(
                run_id=run_id,
                experiment=experiment,
                case=case,
                kind="timeout",
                message=result.error or "llama-bench timed out",
            )
        ]
    elif result.error is not None:
        samples = [
            failure_sample(
                run_id=run_id,
                experiment=experiment,
                case=case,
                kind="process_error",
                message=result.error,
            )
        ]
    elif result.exit_code != 0:
        samples = [
            failure_sample(
                run_id=run_id,
                experiment=experiment,
                case=case,
                kind="nonzero_exit",
                message=f"llama-bench exited with code {result.exit_code}",
            )
        ]
    else:
        try:
            records = parse_json_output(result.stdout)
            samples = normalize_records(
                records,
                run_id=run_id,
                experiment=experiment,
                case=case,
            )
            validate_expected_commit(experiment, samples)
        except LlamaBenchParseError as error:
            samples = [
                failure_sample(
                    run_id=run_id,
                    experiment=experiment,
                    case=case,
                    kind="parse_error",
                    message=str(error),
                )
            ]

    store_case_samples(paths, samples)
    status = "failed" if samples[0].status == "failed" else "completed"
    append_event(
        paths,
        f"case_{status}",
        {
            "case_id": case.case_id,
            "exit_code": result.exit_code,
            "sample_count": len(samples),
        },
    )


def _validate_resume(
    paths: RunPaths,
    experiment: ResolvedExperiment,
    cases: list[BenchCase],
    commands: dict[str, list[str]],
) -> None:
    if not paths.manifest.is_file():
        raise ResumeError(f"run manifest does not exist: {paths.manifest}")
    manifest = read_json(paths.manifest)
    existing_experiment = manifest.get("experiment", {})

    expected_model = experiment.model.model_dump(mode="json")
    expected_runtime = experiment.runtime.model_dump(mode="json")
    expected_benchmark = experiment.benchmark.model_dump(mode="json")
    existing_benchmark = existing_experiment.get("benchmark")
    if isinstance(existing_benchmark, dict) and "track" not in existing_benchmark:
        existing_benchmark = {**existing_benchmark, "track": "raw"}
    if existing_experiment.get("model") != expected_model:
        raise ResumeError("model configuration differs from the original run")
    if existing_experiment.get("runtime") != expected_runtime:
        raise ResumeError("runtime configuration differs from the original run")
    if existing_benchmark != expected_benchmark:
        raise ResumeError("benchmark configuration differs from the original run")

    expected_cases = [{**case.to_dict(), "command": commands[case.case_id]} for case in cases]
    if manifest.get("cases") != expected_cases:
        raise ResumeError("planned cases or commands differ from the original run")
