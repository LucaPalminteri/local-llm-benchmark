from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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
    llama_server_path: Path | None = None
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


class InteractiveServerConfig(StrictModel):
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int | Literal["auto"] = "auto"
    readiness_timeout_seconds: float = Field(default=300, gt=0)
    request_timeout_seconds: float = Field(default=300, gt=0)
    context_size: int = Field(default=16384, gt=0)

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(value, int) and not 1 <= value <= 65535:
            raise ValueError("server port must be between 1 and 65535 or 'auto'")
        return value


class SamplingConfig(StrictModel):
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=40, ge=0)
    seed: int = 42


class InteractiveWorkloadConfig(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    size: Literal["small", "medium", "large"]
    target_prompt_tokens: int = Field(gt=0)


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
    track: Literal["interactive"]
    mode: Literal["smoke", "full"] = "full"
    server: InteractiveServerConfig
    sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig())
    workloads: list[InteractiveWorkloadConfig] = Field(min_length=1)
    requested_output_tokens: int = Field(gt=0)
    warmup_requests: int = Field(default=1, ge=0)
    repetitions: int = Field(default=10, gt=0)
    measure_cold: bool = True
    cold_workload_id: str | None = "small"
    measure_warm_uncached: bool = True
    measure_warm_cached: bool = True
    cached_prefix_ratio: float = Field(default=0.75, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_interactive_plan(self) -> InteractiveBenchmarkConfig:
        ids = [workload.id for workload in self.workloads]
        sizes = [workload.size for workload in self.workloads]
        if len(ids) != len(set(ids)):
            raise ValueError("interactive workload ids must be unique")
        if len(sizes) != len(set(sizes)):
            raise ValueError("interactive workload sizes must be unique")
        if self.mode == "full" and set(sizes) != {"small", "medium", "large"}:
            raise ValueError("full interactive experiments require small, medium, and large")
        if self.mode == "full" and self.repetitions < 10:
            raise ValueError("full interactive experiments require at least 10 repetitions")
        if not (self.measure_warm_uncached or self.measure_warm_cached):
            raise ValueError("at least one warm measurement phase must be enabled")

        if self.measure_cold:
            if self.cold_workload_id not in set(ids):
                raise ValueError("cold_workload_id must identify a configured workload")
        elif self.cold_workload_id is not None:
            raise ValueError("cold_workload_id must be null when measure_cold is false")

        oversized = [
            workload.id
            for workload in self.workloads
            if workload.target_prompt_tokens + self.requested_output_tokens
            > self.server.context_size
        ]
        if oversized:
            raise ValueError(
                "target prompt plus requested output exceeds server context for: "
                + ", ".join(oversized)
            )
        return self


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


def require_interactive_benchmark(
    experiment: ResolvedExperiment,
) -> InteractiveBenchmarkConfig:
    benchmark = experiment.benchmark
    if not isinstance(benchmark, InteractiveBenchmarkConfig):
        raise ConfigError(
            f"expected an interactive benchmark experiment, received track {benchmark.track!r}"
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
    runtime_update: dict[str, Any] = {
        "llama_bench_path": _resolve_path(runtime.llama_bench_path, runtime_path.parent)
    }
    if runtime.llama_server_path is not None:
        runtime_update["llama_server_path"] = _resolve_path(
            runtime.llama_server_path, runtime_path.parent
        )
    runtime = runtime.model_copy(update=runtime_update)
    if (
        isinstance(references.benchmark, InteractiveBenchmarkConfig)
        and runtime.llama_server_path is None
    ):
        raise ConfigError(
            f"interactive experiment {experiment_path} requires runtime llama_server_path"
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
