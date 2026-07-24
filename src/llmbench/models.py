from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from llmbench import RESULT_SCHEMA_VERSION


class NormalizedSampleBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = RESULT_SCHEMA_VERSION
    run_id: str
    case_id: str
    repetition: int | None
    status: Literal["completed", "failed"]
    failure_kind: str | None = None
    failure_message: str | None = None
    model_id: str
    runtime_id: str
    raw_metadata: dict[str, Any] | None = None


class RawNormalizedSample(NormalizedSampleBase):
    track: Literal["raw"] = "raw"
    test_type: Literal["prompt", "generation", "prompt_generation"]
    prompt_tokens: int
    generation_tokens: int
    context_depth: int
    tokens_per_second: float | None = None
    duration_ns: int | None = None
    model_filename: str | None = None
    model_type: str | None = None
    build_commit: str | None = None
    backend: str | None = None
    kv_cache_key: str
    kv_cache_value: str
    gpu_layers: int


class InteractiveNormalizedSample(NormalizedSampleBase):
    track: Literal["interactive"] = "interactive"


# Backward-compatible v0.1 name.
NormalizedSample = RawNormalizedSample
