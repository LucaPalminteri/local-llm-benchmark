from __future__ import annotations

import csv
import io
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from llmbench import (
    RAW_BENCHMARK_PROTOCOL_VERSION,
    RESULT_SCHEMA_VERSION,
)
from llmbench.config import BenchmarkTrack
from llmbench.storage import RunPaths, read_json, read_jsonl, write_json, write_text


def build_summary(paths: RunPaths) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    track = _manifest_track(manifest)
    if track != "raw":
        raise ValueError(f"raw reporting requires a raw benchmark run, received track {track!r}")
    rows = read_jsonl(paths.samples)
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_case[str(row["case_id"])].append(row)

    case_summaries: list[dict[str, Any]] = []
    pending_cases = 0
    failed_cases = 0
    for planned_case in manifest["cases"]:
        case_id = str(planned_case["case_id"])
        case_rows = rows_by_case.get(case_id, [])
        completed = [
            row
            for row in case_rows
            if row.get("status") == "completed" and row.get("tokens_per_second") is not None
        ]
        failures = [row for row in case_rows if row.get("status") == "failed"]
        values = [float(row["tokens_per_second"]) for row in completed]

        if not case_rows:
            status = "pending"
            pending_cases += 1
        elif failures:
            status = "failed"
            failed_cases += 1
        else:
            status = "completed"

        case_summaries.append(
            {
                "case_id": case_id,
                "status": status,
                "test_type": planned_case["test_type"],
                "prompt_tokens": planned_case["prompt_tokens"],
                "generation_tokens": planned_case["generation_tokens"],
                "context_depth": planned_case["context_depth"],
                "sample_count": len(values),
                "failed_sample_count": len(failures),
                "median_tokens_per_second": (statistics.median(values) if values else None),
                "mean_tokens_per_second": (statistics.fmean(values) if values else None),
                "stddev_tokens_per_second": (
                    statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
                ),
                "minimum_tokens_per_second": min(values) if values else None,
                "maximum_tokens_per_second": max(values) if values else None,
                "failure_kind": (str(failures[0].get("failure_kind")) if failures else None),
                "failure_message": (str(failures[0].get("failure_message")) if failures else None),
            }
        )

    if pending_cases:
        status = "partial"
    elif failed_cases:
        status = "completed_with_failures"
    else:
        status = "completed"

    experiment = manifest["experiment"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_track": track,
        "benchmark_protocol_version": manifest.get(
            "benchmark_protocol_version", RAW_BENCHMARK_PROTOCOL_VERSION
        ),
        "run_id": manifest["run_id"],
        "model_id": experiment["model"]["id"],
        "model_name": experiment["model"]["display_name"],
        "runtime_id": experiment["runtime"]["id"],
        "status": status,
        "planned_case_count": len(manifest["cases"]),
        "completed_case_count": len(manifest["cases"]) - pending_cases - failed_cases,
        "failed_case_count": failed_cases,
        "pending_case_count": pending_cases,
        "cases": case_summaries,
    }


def refresh_reports(paths: RunPaths) -> dict[str, Any]:
    summary = build_summary(paths)
    write_json(paths.summary, summary)
    write_text(paths.root / "report.md", summary_to_markdown(summary))
    write_text(paths.root / "report.csv", summary_to_csv(summary))
    return summary


def summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Raw benchmark report: {summary['run_id']}",
        "",
        f"- Model: {summary['model_name']} (`{summary['model_id']}`)",
        f"- Runtime: `{summary['runtime_id']}`",
        f"- Status: {summary['status']}",
        "",
        "| Case | Test | Prompt | Generation | Depth | Samples | Median t/s | Stddev | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in summary["cases"]:
        lines.append(
            "| {case_id} | {test_type} | {prompt_tokens} | "
            "{generation_tokens} | {context_depth} | {sample_count} | "
            "{median} | {stddev} | {status} |".format(
                **case,
                median=_format_number(case["median_tokens_per_second"]),
                stddev=_format_number(case["stddev_tokens_per_second"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def summary_to_csv(summary: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "run_id",
        "model_id",
        "runtime_id",
        "case_id",
        "test_type",
        "prompt_tokens",
        "generation_tokens",
        "context_depth",
        "sample_count",
        "median_tokens_per_second",
        "mean_tokens_per_second",
        "stddev_tokens_per_second",
        "minimum_tokens_per_second",
        "maximum_tokens_per_second",
        "status",
        "failure_kind",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for case in summary["cases"]:
        writer.writerow(
            {
                "run_id": summary["run_id"],
                "model_id": summary["model_id"],
                "runtime_id": summary["runtime_id"],
                **{name: case.get(name) for name in fieldnames if name in case},
            }
        )
    return output.getvalue()


def compare_summaries(summaries: list[dict[str, Any]]) -> str:
    tracks = {_summary_track(summary) for summary in summaries}
    if len(tracks) != 1:
        raise ValueError("cannot compare different benchmark tracks")
    if tracks and tracks != {"raw"}:
        track = next(iter(tracks))
        raise ValueError(f"raw comparison requires raw benchmark runs, received track {track!r}")

    protocol_versions = {str(summary["benchmark_protocol_version"]) for summary in summaries}
    if len(protocol_versions) != 1:
        raise ValueError("cannot compare different benchmark protocol versions")

    case_signatures = [
        {
            (
                case["test_type"],
                case["prompt_tokens"],
                case["generation_tokens"],
                case["context_depth"],
            )
            for case in summary["cases"]
        }
        for summary in summaries
    ]
    if case_signatures and any(
        signature != case_signatures[0] for signature in case_signatures[1:]
    ):
        raise ValueError("cannot compare runs with different benchmark case matrices")

    lines = [
        "# Raw benchmark comparison",
        "",
        "| Run | Model | Runtime | Test | Prompt | Generation | Depth | Median t/s | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        for case in summary["cases"]:
            lines.append(
                "| {run_id} | {model_id} | {runtime_id} | {test_type} | "
                "{prompt_tokens} | {generation_tokens} | {context_depth} | "
                "{median} | {status} |".format(
                    run_id=summary["run_id"],
                    model_id=summary["model_id"],
                    runtime_id=summary["runtime_id"],
                    median=_format_number(case["median_tokens_per_second"]),
                    **case,
                )
            )
    lines.append("")
    return "\n".join(lines)


def find_latest_run(
    runs_directory: Path,
    *,
    track: BenchmarkTrack | None = None,
) -> Path:
    candidates = (
        [
            path
            for path in runs_directory.iterdir()
            if path.is_dir()
            and (path / "manifest.json").is_file()
            and (
                track is None
                or _manifest_track(cast(dict[str, Any], read_json(path / "manifest.json"))) == track
            )
        ]
        if runs_directory.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(f"no benchmark runs found in {runs_directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _manifest_track(manifest: dict[str, Any]) -> BenchmarkTrack:
    track = manifest.get("benchmark_track")
    if track in ("raw", "interactive"):
        return cast(BenchmarkTrack, track)

    protocol = manifest.get("benchmark_protocol_version")
    if isinstance(protocol, str) and protocol.startswith("raw-"):
        return "raw"
    if isinstance(protocol, str) and protocol.startswith("interactive-"):
        return "interactive"
    raise ValueError("run manifest does not identify a supported benchmark track")


def _summary_track(summary: dict[str, Any]) -> BenchmarkTrack:
    track = summary.get("benchmark_track")
    if track in ("raw", "interactive"):
        return cast(BenchmarkTrack, track)

    protocol = summary.get("benchmark_protocol_version")
    if isinstance(protocol, str) and protocol.startswith("raw-"):
        return "raw"
    if isinstance(protocol, str) and protocol.startswith("interactive-"):
        return "interactive"
    raise ValueError("summary does not identify a supported benchmark track")


def _format_number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    raise TypeError(f"expected a numeric report value, received {type(value).__name__}")
