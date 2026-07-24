from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmbench import INTERACTIVE_BENCHMARK_PROTOCOL_VERSION, RAW_BENCHMARK_PROTOCOL_VERSION

BenchmarkTrack = Literal["raw", "interactive"]


class ConfigError(ValueError):
    """Raised when a benchmark configuration cannot be loaded."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelMetadata(StrictModel):
    family: str | None = None
    parameters_billion: float | None = Field(default=None, gt=0)
    quantization: str | None = None
    architecture: str | None = None
    model_type: str | None = None
    file_type: int | None = Field(default=None, ge=0)
    quantization_version: int | None = Field(default=None, gt=0)
    size_label: str | None = None
    finetune: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    embedding_length: int | None = Field(default=None, gt=0)
    block_count: int | None = Field(default=None, gt=0)
    expert_count: int | None = Field(default=None, gt=0)
    expert_used_count: int | None = Field(default=None, gt=0)
    file_size_bytes: int | None = Field(default=None, gt=0)
    shard_count: int | None = Field(default=None, gt=0)
    gguf_version: int | None = Field(default=None, gt=0)
    tensor_count: int | None = Field(default=None, ge=0)
    source_url: str | None = None
    license: str | None = None


class ModelConfig(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str = Field(min_length=1)
    gguf_path: Path
    metadata: ModelMetadata = Field(default_factory=ModelMetadata)


class KvCacheConfig(StrictModel):
    key: str = Field(default="f16", min_length=1)
    value: str = Field(default="f16", min_length=1)


class RuntimeProfile(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    llama_bench_path: Path
    expected_commit: str | None = None
    backend: str = "Vulkan"
    gpu_layers: int = Field(default=99, ge=-1)
    flash_attention: Literal["on", "off", "auto"] = "on"
    kv_cache: KvCacheConfig = Field(default_factory=KvCacheConfig)
    threads: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=2048, gt=0)
    micro_batch_size: int = Field(default=512, gt=0)
    delay_seconds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> RuntimeProfile:
        if self.micro_batch_size > self.batch_size:
            raise ValueError("micro_batch_size cannot exceed batch_size")
        return self


class PromptGenerationPair(StrictModel):
    prompt: int = Field(gt=0)
    generation: int = Field(gt=0)


class BenchmarkConfigBase(StrictModel):
    output_directory: Path = Path("../../runs")


class RawBenchmarkConfig(BenchmarkConfigBase):
    track: Literal["raw"] = "raw"
    prompt_tokens: list[int] = Field(default_factory=list)
    generation_tokens: list[int] = Field(default_factory=list)
    prompt_generation_pairs: list[PromptGenerationPair] = Field(default_factory=list)
    context_depths: list[int] = Field(default_factory=lambda: [0])
    repetitions: int = Field(default=7, gt=0)
    timeout_seconds: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def validate_matrix(self) -> RawBenchmarkConfig:
        if not (self.prompt_tokens or self.generation_tokens or self.prompt_generation_pairs):
            raise ValueError("at least one benchmark workload must be configured")

        positive_lists = {
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
        }
        for name, values in positive_lists.items():
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be greater than zero")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique")

        if any(depth < 0 for depth in self.context_depths):
            raise ValueError("context_depths values cannot be negative")
        if len(self.context_depths) != len(set(self.context_depths)):
            raise ValueError("context_depths values must be unique")
        if not self.context_depths:
            raise ValueError("context_depths cannot be empty")
        return self


class InteractiveBenchmarkConfig(BenchmarkConfigBase):
    """Minimal interactive track contract extended by the v0.2 implementation."""

    track: Literal["interactive"]


BenchmarkConfig = Annotated[
    RawBenchmarkConfig | InteractiveBenchmarkConfig,
    Field(discriminator="track"),
]


class ExperimentReferences(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model: Path
    runtime: Path
    benchmark: BenchmarkConfig

    @model_validator(mode="before")
    @classmethod
    def default_legacy_benchmark_track(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        benchmark = value.get("benchmark")
        if not isinstance(benchmark, dict) or "track" in benchmark:
            return value
        updated = dict(value)
        updated["benchmark"] = {**benchmark, "track": "raw"}
        return updated


@dataclass(frozen=True)
class ResolvedExperiment:
    id: str
    model: ModelConfig
    runtime: RuntimeProfile
    benchmark: RawBenchmarkConfig | InteractiveBenchmarkConfig
    experiment_path: Path
    model_path: Path
    runtime_path: Path

    @property
    def track(self) -> BenchmarkTrack:
        return self.benchmark.track


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"could not read configuration {path}: {error}") from error

    try:
        document = yaml.safe_load(contents)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error

    if not isinstance(document, dict):
        raise ConfigError(f"configuration {path} must contain a YAML object")
    return document


def _resolve_path(path: Path, base_directory: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_directory / path).resolve()


def benchmark_protocol_version(track: BenchmarkTrack) -> str:
    if track == "raw":
        return RAW_BENCHMARK_PROTOCOL_VERSION
    return INTERACTIVE_BENCHMARK_PROTOCOL_VERSION


def require_raw_benchmark(experiment: ResolvedExperiment) -> RawBenchmarkConfig:
    benchmark = experiment.benchmark
    if not isinstance(benchmark, RawBenchmarkConfig):
        raise ConfigError(
            f"expected a raw benchmark experiment, received track {benchmark.track!r}"
        )
    return benchmark


def load_experiment(path: Path) -> ResolvedExperiment:
    experiment_path = path.resolve()
    try:
        references = ExperimentReferences.model_validate(_read_yaml(experiment_path))

        model_path = _resolve_path(references.model, experiment_path.parent)
        runtime_path = _resolve_path(references.runtime, experiment_path.parent)
        model = ModelConfig.model_validate(_read_yaml(model_path))
        runtime = RuntimeProfile.model_validate(_read_yaml(runtime_path))
    except ValidationError as error:
        raise ConfigError(str(error)) from error

    model = model.model_copy(
        update={"gguf_path": _resolve_path(model.gguf_path, model_path.parent)}
    )
    runtime = runtime.model_copy(
        update={"llama_bench_path": _resolve_path(runtime.llama_bench_path, runtime_path.parent)}
    )
    benchmark = references.benchmark.model_copy(
        update={
            "output_directory": _resolve_path(
                references.benchmark.output_directory, experiment_path.parent
            )
        }
    )

    return ResolvedExperiment(
        id=references.id,
        model=model,
        runtime=runtime,
        benchmark=benchmark,
        experiment_path=experiment_path,
        model_path=model_path,
        runtime_path=runtime_path,
    )
