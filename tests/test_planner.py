from __future__ import annotations

from llmbench.config import ResolvedExperiment
from llmbench.llama_bench import build_command
from llmbench.planner import expand_cases


def test_expands_each_workload_across_context_depths(
    resolved_experiment: ResolvedExperiment,
) -> None:
    cases = expand_cases(resolved_experiment)

    assert len(cases) == 8
    assert [case.case_id for case in cases] == [
        "pp-p512-d0",
        "tg-n128-d0",
        "tg-n256-d0",
        "pg-p64-n16-d0",
        "pp-p512-d2048",
        "tg-n128-d2048",
        "tg-n256-d2048",
        "pg-p64-n16-d2048",
    ]


def test_builds_generation_command_without_shell_string(
    resolved_experiment: ResolvedExperiment,
) -> None:
    generation_case = expand_cases(resolved_experiment)[1]

    command = build_command(resolved_experiment, generation_case)

    assert isinstance(command, list)
    assert command[0].endswith("llama-bench.exe")
    assert command[command.index("-p") + 1] == "0"
    assert command[command.index("-n") + 1] == "128"
    assert command[command.index("-d") + 1] == "0"
    assert command[command.index("-r") + 1] == "3"
    assert command[command.index("-o") + 1] == "json"
    assert command[command.index("-fa") + 1] == "on"
