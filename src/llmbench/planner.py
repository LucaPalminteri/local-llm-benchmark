from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from llmbench.config import (
    ResolvedExperiment,
    require_interactive_benchmark,
    require_raw_benchmark,
)
from llmbench.models import MeasurementPhase

TestType = Literal["prompt", "generation", "prompt_generation"]


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    test_type: TestType
    prompt_tokens: int
    generation_tokens: int
    context_depth: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InteractiveCase:
    case_id: str
    workload_id: str
    workload_size: Literal["small", "medium", "large"]
    target_prompt_tokens: int
    requested_output_tokens: int
    phase: MeasurementPhase
    repetitions: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def expand_cases(experiment: ResolvedExperiment) -> list[BenchCase]:
    benchmark = require_raw_benchmark(experiment)
    cases: list[BenchCase] = []

    for depth in benchmark.context_depths:
        for prompt_tokens in benchmark.prompt_tokens:
            cases.append(
                BenchCase(
                    case_id=f"pp-p{prompt_tokens}-d{depth}",
                    test_type="prompt",
                    prompt_tokens=prompt_tokens,
                    generation_tokens=0,
                    context_depth=depth,
                )
            )

        for generation_tokens in benchmark.generation_tokens:
            cases.append(
                BenchCase(
                    case_id=f"tg-n{generation_tokens}-d{depth}",
                    test_type="generation",
                    prompt_tokens=0,
                    generation_tokens=generation_tokens,
                    context_depth=depth,
                )
            )

        for pair in benchmark.prompt_generation_pairs:
            cases.append(
                BenchCase(
                    case_id=f"pg-p{pair.prompt}-n{pair.generation}-d{depth}",
                    test_type="prompt_generation",
                    prompt_tokens=pair.prompt,
                    generation_tokens=pair.generation,
                    context_depth=depth,
                )
            )

    return cases


def expand_interactive_cases(experiment: ResolvedExperiment) -> list[InteractiveCase]:
    benchmark = require_interactive_benchmark(experiment)
    workloads = {workload.id: workload for workload in benchmark.workloads}
    cases: list[InteractiveCase] = []

    def add_case(workload_id: str, phase: MeasurementPhase, repetitions: int) -> None:
        workload = workloads[workload_id]
        cases.append(
            InteractiveCase(
                case_id=f"{workload.id}-{phase.replace('_', '-')}",
                workload_id=workload.id,
                workload_size=workload.size,
                target_prompt_tokens=workload.target_prompt_tokens,
                requested_output_tokens=benchmark.requested_output_tokens,
                phase=phase,
                repetitions=repetitions,
            )
        )

    if benchmark.measure_cold:
        assert benchmark.cold_workload_id is not None
        add_case(benchmark.cold_workload_id, "cold", 1)

    for workload in benchmark.workloads:
        if benchmark.warmup_requests:
            add_case(workload.id, "warmup", benchmark.warmup_requests)
        if benchmark.measure_warm_uncached:
            add_case(workload.id, "warm_uncached", benchmark.repetitions)
        if benchmark.measure_warm_cached:
            add_case(workload.id, "warm_cached", benchmark.repetitions)

    return cases
