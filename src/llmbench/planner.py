from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from llmbench.config import ResolvedExperiment

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


def expand_cases(experiment: ResolvedExperiment) -> list[BenchCase]:
    benchmark = experiment.benchmark
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
