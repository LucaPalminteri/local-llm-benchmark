from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml

from llmbench import APP_VERSION
from llmbench.config import (
    ConfigError,
    InteractiveBenchmarkConfig,
    RawBenchmarkConfig,
    ResolvedExperiment,
    load_experiment,
    require_interactive_benchmark,
    require_raw_benchmark,
)
from llmbench.discovery import (
    DiscoveryError,
    default_model_roots,
    discover_model_roots,
    info_as_dict,
    inspect_gguf,
    write_model_configs,
)
from llmbench.doctor import run_doctor
from llmbench.llama_bench import build_command
from llmbench.planner import expand_cases, expand_interactive_cases
from llmbench.reporting import (
    compare_summaries,
    find_latest_run,
    refresh_reports,
    summary_to_markdown,
)
from llmbench.runner import ResumeError, run_raw_experiment
from llmbench.storage import paths_for_run, write_text

DEFAULT_CONFIG = Path("configs/experiments/raw-smoke.yaml")

app = typer.Typer(
    name="llmbench",
    help="Reproducible benchmarks for local LLM configurations.",
    no_args_is_help=True,
)
raw_app = typer.Typer(help="Run and report raw llama-bench experiments.")
models_app = typer.Typer(help="Discover and inspect local GGUF models.")
app.add_typer(raw_app, name="raw")
app.add_typer(models_app, name="models")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(APP_VERSION)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the application version.",
        ),
    ] = False,
) -> None:
    """Run reproducible local LLM benchmarks."""


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Experiment configuration to validate.",
        ),
    ] = DEFAULT_CONFIG,
) -> None:
    """Check whether the configured benchmark can run on this machine."""
    experiment = _load_or_exit(config)
    checks = run_doctor(experiment)
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        typer.echo(f"[{marker}] {check.name}: {check.detail}")
    if not all(check.passed for check in checks):
        raise typer.Exit(code=1)


@app.command()
def plan(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Experiment configuration to expand.",
        ),
    ] = DEFAULT_CONFIG,
) -> None:
    """Show the complete benchmark matrix without executing it."""
    experiment = _load_or_exit(config)
    if experiment.track == "interactive":
        _show_interactive_plan(experiment)
        return

    benchmark = _raw_benchmark_or_exit(experiment)
    cases = expand_cases(experiment)
    typer.echo(
        f"Experiment {experiment.id}: {len(cases)} cases x {benchmark.repetitions} repetitions"
    )
    typer.echo("CASE ID                         TEST               PROMPT  GEN  DEPTH")
    for case in cases:
        typer.echo(
            f"{case.case_id:<31} {case.test_type:<18} "
            f"{case.prompt_tokens:>6} {case.generation_tokens:>4} "
            f"{case.context_depth:>6}"
        )
    typer.echo("")
    typer.echo("Commands:")
    for case in cases:
        typer.echo(f"  {case.case_id}")
        typer.echo(f"    {_display_command(build_command(experiment, case))}")


@models_app.command("inspect")
def models_inspect(
    model: Annotated[
        Path,
        typer.Argument(help="GGUF file to inspect."),
    ],
) -> None:
    """Print embedded metadata for one local GGUF file."""
    try:
        info = inspect_gguf(model)
    except DiscoveryError as error:
        _fail(str(error))
    typer.echo(yaml.safe_dump(info_as_dict(info), sort_keys=False, allow_unicode=True))


@models_app.command("discover")
def models_discover(
    root: Annotated[
        Path | None,
        typer.Argument(
            help="Directory to scan. Omit to search configured and standard model roots."
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory for generated local model configurations.",
        ),
    ] = Path("configs/local/models"),
    write: Annotated[
        bool,
        typer.Option(
            "--write",
            help="Write model configurations. The default is a safe dry run.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing generated configurations. Requires --write.",
        ),
    ] = False,
) -> None:
    """Discover runnable models and optionally generate local YAML configs."""
    if overwrite and not write:
        _fail("--overwrite requires --write")
    roots = (root,) if root is not None else default_model_roots()
    if not roots:
        _fail("no model roots were detected; pass ROOT or set LLMBENCH_MODEL_PATHS")

    typer.echo("Searching model root(s):")
    for search_root in roots:
        typer.echo(f"  {search_root.resolve()}")
    typer.echo("")

    try:
        result = discover_model_roots(roots)
    except DiscoveryError as error:
        _fail(str(error))

    typer.echo(f"Discovered {len(result.models)} runnable model(s)")
    for model in result.models:
        metadata = model.config.metadata
        details = " | ".join(
            value
            for value in (
                metadata.size_label,
                metadata.quantization,
                metadata.architecture,
            )
            if value
        )
        typer.echo(f"  {model.config.id}: {details}")
        typer.echo(f"    {model.config.gguf_path}")

    if result.issues:
        typer.echo("")
        typer.echo("Scan notes:")
        for issue in result.issues:
            typer.echo(f"  [{issue.kind.upper()}] {issue.path}: {issue.message}")

    if not result.models:
        _fail("no runnable GGUF models were discovered")

    if write:
        try:
            paths = write_model_configs(
                result.models,
                output,
                overwrite=overwrite,
            )
        except DiscoveryError as error:
            _fail(str(error))
        typer.echo("")
        typer.echo(f"Wrote {len(paths)} model config(s) to {output.resolve()}")
    else:
        typer.echo("")
        typer.echo("Dry run: no files written. Re-run with --write to generate configs.")


@raw_app.command("run")
def raw_run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Experiment configuration to execute.",
        ),
    ] = DEFAULT_CONFIG,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Resume a prior run ID and skip its terminal cases.",
        ),
    ] = None,
) -> None:
    """Execute a raw llama-bench experiment."""
    experiment = _load_or_exit(config)
    _raw_benchmark_or_exit(experiment)
    missing = [
        path
        for path in (
            experiment.runtime.llama_bench_path,
            experiment.model.gguf_path,
        )
        if not path.is_file()
    ]
    if missing:
        typer.echo("Cannot start benchmark; required files are missing:", err=True)
        for path in missing:
            typer.echo(f"  {path}", err=True)
        raise typer.Exit(code=1)

    try:
        outcome = run_raw_experiment(experiment, resume_run_id=resume)
    except ResumeError as error:
        _fail(str(error))
    except KeyboardInterrupt:
        typer.echo("Benchmark interrupted. Completed cases were preserved.", err=True)
        raise typer.Exit(code=130) from None

    typer.echo(f"Run: {outcome.run_id}")
    typer.echo(f"Status: {outcome.summary['status']}")
    typer.echo(f"Artifacts: {outcome.paths.root}")
    typer.echo("")
    typer.echo(summary_to_markdown(outcome.summary))


@raw_app.command("report")
def raw_report(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run ID. Omit when using --latest."),
    ] = None,
    latest: Annotated[
        bool,
        typer.Option(
            "--latest",
            help="Report the most recently modified run.",
        ),
    ] = False,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Configuration used to locate the runs directory.",
        ),
    ] = DEFAULT_CONFIG,
) -> None:
    """Regenerate terminal, Markdown, and CSV reports for a run."""
    experiment = _load_or_exit(config)
    benchmark = _raw_benchmark_or_exit(experiment)
    if latest and run_id is not None:
        _fail("provide a run ID or --latest, not both")
    if latest:
        try:
            run_directory = find_latest_run(benchmark.output_directory, track="raw")
        except (FileNotFoundError, ValueError) as error:
            _fail(str(error))
    elif run_id is not None:
        run_directory = benchmark.output_directory / run_id
    else:
        _fail("provide a run ID or use --latest")

    paths = paths_for_run(run_directory)
    if not paths.manifest.is_file():
        _fail(f"run manifest not found: {paths.manifest}")
    try:
        summary = refresh_reports(paths)
    except ValueError as error:
        _fail(str(error))
    typer.echo(summary_to_markdown(summary))
    typer.echo(f"Markdown: {paths.root / 'report.md'}")
    typer.echo(f"CSV: {paths.root / 'report.csv'}")


@app.command()
def compare(
    run_id_a: Annotated[str, typer.Argument(help="First run ID.")],
    run_id_b: Annotated[str, typer.Argument(help="Second run ID.")],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Configuration used to locate runs and reports.",
        ),
    ] = DEFAULT_CONFIG,
) -> None:
    """Compare two compatible raw-performance runs."""
    experiment = _load_or_exit(config)
    benchmark = _raw_benchmark_or_exit(experiment)
    summaries = []
    for run_id in (run_id_a, run_id_b):
        paths = paths_for_run(benchmark.output_directory / run_id)
        if not paths.manifest.is_file():
            _fail(f"run manifest not found: {paths.manifest}")
        summaries.append(refresh_reports(paths))

    try:
        report = compare_summaries(summaries)
    except ValueError as error:
        _fail(str(error))

    reports_directory = benchmark.output_directory.parent / "reports"
    report_path = reports_directory / f"compare-{run_id_a}-vs-{run_id_b}.md"
    write_text(report_path, report)
    typer.echo(report)
    typer.echo(f"Markdown: {report_path}")


def _load_or_exit(config: Path) -> ResolvedExperiment:
    try:
        return load_experiment(config)
    except ConfigError as error:
        _fail(str(error))


def _raw_benchmark_or_exit(experiment: ResolvedExperiment) -> RawBenchmarkConfig:
    try:
        return require_raw_benchmark(experiment)
    except ConfigError as error:
        _fail(str(error))


def _interactive_benchmark_or_exit(
    experiment: ResolvedExperiment,
) -> InteractiveBenchmarkConfig:
    try:
        return require_interactive_benchmark(experiment)
    except ConfigError as error:
        _fail(str(error))


def _show_interactive_plan(experiment: ResolvedExperiment) -> None:
    benchmark = _interactive_benchmark_or_exit(experiment)
    cases = expand_interactive_cases(experiment)
    request_count = sum(case.repetitions for case in cases)
    measured_count = sum(case.repetitions for case in cases if case.phase != "warmup")
    endpoint = f"{benchmark.server.host}:{benchmark.server.port}"

    typer.echo(
        f"Experiment {experiment.id}: {len(cases)} cases, "
        f"{request_count} requests ({measured_count} measured)"
    )
    typer.echo(
        f"Server: {endpoint} | context {benchmark.server.context_size} | "
        f"ready timeout {benchmark.server.readiness_timeout_seconds:g}s | "
        f"request timeout {benchmark.server.request_timeout_seconds:g}s"
    )
    typer.echo(
        f"Sampling: temperature {benchmark.sampling.temperature:g} | "
        f"top_p {benchmark.sampling.top_p:g} | top_k {benchmark.sampling.top_k} | "
        f"seed {benchmark.sampling.seed}"
    )
    typer.echo("CASE ID                         WORKLOAD  PHASE            INPUT  OUTPUT  REQUESTS")
    for case in cases:
        typer.echo(
            f"{case.case_id:<31} {case.workload_size:<9} {case.phase:<16} "
            f"{case.target_prompt_tokens:>5} {case.requested_output_tokens:>7} "
            f"{case.repetitions:>9}"
        )


def _display_command(command: list[str]) -> str:
    return " ".join(
        f'"{argument}"' if any(character.isspace() for character in argument) else argument
        for argument in command
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)
