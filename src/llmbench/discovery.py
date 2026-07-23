from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from llmbench.config import ModelConfig, ModelMetadata
from llmbench.gguf_metadata import GGUFMetadataError, read_gguf_metadata

_SHARD_PATTERN = re.compile(
    r"^(?P<base>.+)-(?P<number>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)
_SIZE_BILLION_PATTERN = re.compile(r"^(?P<count>\d+(?:\.\d+)?)B$", re.IGNORECASE)
_NON_ID_CHARACTERS = re.compile(r"[^a-z0-9._-]+")
_REPEATED_DASHES = re.compile(r"-+")
_SIDECAR_TYPES = {"adapter", "clip-vision", "lora", "mmproj", "vocab"}
_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.basename",
        "general.file_type",
        "general.finetune",
        "general.license",
        "general.name",
        "general.quantization_version",
        "general.size_label",
        "general.source.repo_url",
        "general.source.url",
        "general.type",
        "general.url",
        "split.count",
        "split.no",
    }
)
_FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}


class DiscoveryError(ValueError):
    """Raised when local GGUF discovery cannot proceed safely."""


@dataclass(frozen=True)
class GGUFInfo:
    path: Path
    name: str
    basename: str | None
    model_type: str | None
    architecture: str | None
    size_label: str | None
    finetune: str | None
    quantization: str | None
    file_type: int | None
    quantization_version: int | None
    context_length: int | None
    embedding_length: int | None
    block_count: int | None
    expert_count: int | None
    expert_used_count: int | None
    gguf_version: int | None
    tensor_count: int | None
    split_number: int | None
    split_count: int | None
    source_url: str | None
    license: str | None
    file_size_bytes: int

    @property
    def is_runnable_model(self) -> bool:
        model_type = (self.model_type or "model").casefold()
        filename = self.path.name.casefold()
        return (
            model_type not in _SIDECAR_TYPES
            and (self.architecture or "").casefold() != "clip"
            and "mmproj" not in filename
        )


@dataclass(frozen=True)
class DiscoveryIssue:
    path: Path
    kind: Literal["skipped", "warning", "error"]
    message: str


@dataclass(frozen=True)
class DiscoveredModel:
    config: ModelConfig
    source: GGUFInfo


@dataclass(frozen=True)
class DiscoveryResult:
    models: tuple[DiscoveredModel, ...]
    issues: tuple[DiscoveryIssue, ...]


def inspect_gguf(path: Path) -> GGUFInfo:
    """Read useful scalar metadata from a GGUF without loading its tensor data."""
    model_path = path.resolve()
    if not model_path.is_file():
        raise DiscoveryError(f"GGUF file does not exist: {model_path}")
    if model_path.suffix.casefold() != ".gguf":
        raise DiscoveryError(f"expected a .gguf file: {model_path}")

    try:
        initial = read_gguf_metadata(
            model_path,
            frozenset({"general.architecture"}),
            stop_when_found=True,
        )
        architecture = _string_value(initial.values, "general.architecture")
        architecture_keys = frozenset(
            f"{architecture}.{suffix}"
            for suffix in (
                "block_count",
                "context_length",
                "embedding_length",
                "expert_count",
                "expert_used_count",
            )
        )
        metadata = read_gguf_metadata(model_path, _METADATA_KEYS | architecture_keys)
    except GGUFMetadataError as error:
        raise DiscoveryError(f"could not read GGUF metadata from {model_path}: {error}") from error

    values = metadata.values
    architecture = _string_value(values, "general.architecture")
    file_type = _integer_value(values, "general.file_type")
    quantization = _quantization_name(file_type, model_path)
    name = _string_value(values, "general.name")
    if name is None or name.casefold() in {"gguf model", "model", "vlm model"}:
        name = _name_from_filename(model_path, quantization)

    return GGUFInfo(
        path=model_path,
        name=name,
        basename=_string_value(values, "general.basename"),
        model_type=_string_value(values, "general.type"),
        architecture=architecture,
        size_label=_string_value(values, "general.size_label"),
        finetune=_string_value(values, "general.finetune"),
        quantization=quantization,
        file_type=file_type,
        quantization_version=_integer_value(values, "general.quantization_version"),
        context_length=_architecture_integer_value(values, architecture, "context_length"),
        embedding_length=_architecture_integer_value(values, architecture, "embedding_length"),
        block_count=_architecture_integer_value(values, architecture, "block_count"),
        expert_count=_architecture_integer_value(values, architecture, "expert_count"),
        expert_used_count=_architecture_integer_value(values, architecture, "expert_used_count"),
        gguf_version=metadata.version,
        tensor_count=metadata.tensor_count,
        split_number=_integer_value(values, "split.no"),
        split_count=_integer_value(values, "split.count"),
        source_url=(
            _string_value(values, "general.source.repo_url")
            or _string_value(values, "general.source.url")
            or _string_value(values, "general.url")
        ),
        license=_string_value(values, "general.license"),
        file_size_bytes=model_path.stat().st_size,
    )


def discover_models(root: Path) -> DiscoveryResult:
    """Discover runnable GGUF models below a local directory."""
    return discover_model_roots((root,))


def discover_model_roots(roots: tuple[Path, ...]) -> DiscoveryResult:
    """Discover runnable GGUF models below one or more local directories."""
    discovery_roots = tuple(root.resolve() for root in roots)
    if not discovery_roots:
        raise DiscoveryError("no model roots were detected; pass ROOT or set LLMBENCH_MODEL_PATHS")
    invalid_roots = tuple(root for root in discovery_roots if not root.is_dir())
    if invalid_roots:
        invalid_description = ", ".join(str(root) for root in invalid_roots)
        raise DiscoveryError(f"model root is not a directory: {invalid_description}")

    models: list[DiscoveredModel] = []
    issues: list[DiscoveryIssue] = []
    seen_files: set[tuple[int, int] | str] = set()
    used_ids: set[str] = set()

    try:
        paths_by_name = {
            str(candidate.resolve()).casefold(): candidate
            for root in discovery_roots
            for candidate in root.rglob("*.gguf")
        }
        paths = sorted(paths_by_name.values(), key=lambda candidate: str(candidate).casefold())
    except OSError as error:
        raise DiscoveryError(f"could not scan model roots: {error}") from error

    for path in paths:
        shard_match = _SHARD_PATTERN.match(path.name)
        if shard_match and int(shard_match.group("number")) != 1:
            issues.append(
                DiscoveryIssue(path, "skipped", "non-primary shard; represented by shard 00001")
            )
            continue

        try:
            identity = _file_identity(path)
        except OSError as error:
            issues.append(DiscoveryIssue(path, "error", f"could not stat file: {error}"))
            continue
        if identity in seen_files:
            issues.append(DiscoveryIssue(path, "skipped", "duplicate or hard-linked GGUF"))
            continue
        seen_files.add(identity)

        try:
            info = inspect_gguf(path)
        except DiscoveryError as error:
            issues.append(DiscoveryIssue(path, "error", str(error)))
            continue

        if not info.is_runnable_model:
            issues.append(
                DiscoveryIssue(
                    path,
                    "skipped",
                    f"GGUF sidecar ({info.model_type or info.architecture or 'unknown type'})",
                )
            )
            continue

        shard_count = info.split_count or _shard_count_from_filename(path) or 1
        total_size, missing_shards = _model_file_size(path, shard_count)
        if missing_shards:
            missing = ", ".join(f"{number:05d}" for number in missing_shards)
            issues.append(DiscoveryIssue(path, "warning", f"missing shard numbers: {missing}"))

        config = _model_config(info, total_size=total_size, shard_count=shard_count)
        if config.id in used_ids:
            suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
            config = config.model_copy(update={"id": f"{config.id}-{suffix}"})
            issues.append(
                DiscoveryIssue(path, "warning", f"duplicate model ID; added suffix {suffix}")
            )
        used_ids.add(config.id)
        models.append(DiscoveredModel(config=config, source=info))

    return DiscoveryResult(models=tuple(models), issues=tuple(issues))


def default_model_roots(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> tuple[Path, ...]:
    """Return existing, conventional local model roots in priority order."""
    environment = os.environ if environ is None else environ
    home_directory = Path.home() if home is None else home
    working_directory = Path.cwd() if cwd is None else cwd

    configured = environment.get("LLMBENCH_MODEL_PATHS")
    if configured:
        configured_candidates = tuple(
            _environment_path(value) for value in configured.split(os.pathsep) if value.strip()
        )
        return _existing_unique_directories(configured_candidates)

    candidates: list[Path] = []
    llama_cache = environment.get("LLAMA_CACHE")
    if llama_cache:
        candidates.append(_environment_path(llama_cache))

    hugging_face_cache = environment.get("HF_HUB_CACHE")
    if hugging_face_cache:
        candidates.append(_environment_path(hugging_face_cache))
    else:
        hugging_face_home = environment.get("HF_HOME")
        if hugging_face_home:
            candidates.append(_environment_path(hugging_face_home) / "hub")
        else:
            xdg_cache_home = environment.get("XDG_CACHE_HOME")
            cache_home = (
                _environment_path(xdg_cache_home) if xdg_cache_home else home_directory / ".cache"
            )
            candidates.append(cache_home / "huggingface" / "hub")

    candidates.extend(
        (
            working_directory / "models",
            home_directory / "models",
            home_directory / "Models",
        )
    )
    return _existing_unique_directories(tuple(candidates))


def write_model_configs(
    models: tuple[DiscoveredModel, ...],
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write discovered model configs after checking all overwrite conflicts."""
    output = output_directory.resolve()
    destinations = tuple(output / f"{model.config.id}.yaml" for model in models)
    conflicts = tuple(path for path in destinations if path.exists())
    if conflicts and not overwrite:
        conflict_list = ", ".join(str(path) for path in conflicts)
        raise DiscoveryError(
            f"refusing to overwrite existing model configs: {conflict_list}; "
            "use --overwrite to replace them"
        )

    output.mkdir(parents=True, exist_ok=True)
    for model, destination in zip(models, destinations, strict=True):
        document = model.config.model_dump(mode="json", exclude_none=True)
        destination.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return destinations


def info_as_dict(info: GGUFInfo) -> dict[str, Any]:
    """Return inspection metadata suitable for terminal YAML output."""
    return {
        "path": str(info.path),
        "runnable_model": info.is_runnable_model,
        "name": info.name,
        "basename": info.basename,
        "model_type": info.model_type,
        "architecture": info.architecture,
        "size_label": info.size_label,
        "finetune": info.finetune,
        "quantization": info.quantization,
        "file_type": info.file_type,
        "quantization_version": info.quantization_version,
        "context_length": info.context_length,
        "embedding_length": info.embedding_length,
        "block_count": info.block_count,
        "expert_count": info.expert_count,
        "expert_used_count": info.expert_used_count,
        "gguf_version": info.gguf_version,
        "tensor_count": info.tensor_count,
        "split_number": info.split_number,
        "split_count": info.split_count,
        "source_url": info.source_url,
        "license": info.license,
        "file_size_bytes": info.file_size_bytes,
    }


def _model_config(info: GGUFInfo, *, total_size: int, shard_count: int) -> ModelConfig:
    display_name = info.name
    if info.quantization:
        display_name = f"{display_name} {info.quantization}"
    parameters_billion = _parameters_billion(info.size_label)
    family = info.basename or info.name.split(maxsplit=1)[0] or info.architecture
    return ModelConfig(
        id=_slugify(display_name),
        display_name=display_name,
        gguf_path=info.path,
        metadata=ModelMetadata(
            family=family,
            parameters_billion=parameters_billion,
            quantization=info.quantization,
            architecture=info.architecture,
            model_type=info.model_type,
            file_type=info.file_type,
            quantization_version=info.quantization_version,
            size_label=info.size_label,
            finetune=info.finetune,
            context_length=info.context_length,
            embedding_length=info.embedding_length,
            block_count=info.block_count,
            expert_count=info.expert_count,
            expert_used_count=info.expert_used_count,
            file_size_bytes=total_size,
            shard_count=shard_count,
            gguf_version=info.gguf_version,
            tensor_count=info.tensor_count,
            source_url=info.source_url,
            license=info.license,
        ),
    )


def _string_value(values: dict[str, object], key: str) -> str | None:
    value = values.get(key)
    return value if isinstance(value, str) and value else None


def _integer_value(values: dict[str, object], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _architecture_integer_value(
    values: dict[str, object],
    architecture: str | None,
    suffix: str,
) -> int | None:
    if architecture is None:
        return None
    return _integer_value(values, f"{architecture}.{suffix}")


def _quantization_name(file_type: int | None, path: Path) -> str | None:
    if file_type is not None:
        name = _FILE_TYPE_NAMES.get(file_type)
        if name is not None:
            return name

    filename_match = re.search(
        r"(?:^|[._-])((?:I?Q\d(?:_[A-Z0-9]+)+)|BF16|F16|F32)(?:[._-]|$)",
        path.name.upper(),
    )
    return filename_match.group(1) if filename_match else None


def _parameters_billion(size_label: str | None) -> float | None:
    if size_label is None:
        return None
    match = _SIZE_BILLION_PATTERN.fullmatch(size_label.strip())
    return float(match.group("count")) if match else None


def _slugify(value: str) -> str:
    normalized = value.casefold().replace("_", "-").replace(" ", "-")
    normalized = _NON_ID_CHARACTERS.sub("-", normalized)
    normalized = _REPEATED_DASHES.sub("-", normalized).strip("-.")
    return normalized or "discovered-model"


def _name_from_filename(path: Path, quantization: str | None) -> str:
    stem = _SHARD_PATTERN.sub(lambda match: match.group("base"), path.name)
    if stem.casefold().endswith(".gguf"):
        stem = stem[:-5]
    if quantization is not None:
        for separator in ("-", ".", "_"):
            suffix = f"{separator}{quantization}"
            if stem.casefold().endswith(suffix.casefold()):
                stem = stem[: -len(suffix)]
                break
    return stem.replace("_", " ").replace("-", " ").strip() or "Discovered Model"


def _file_identity(path: Path) -> tuple[int, int] | str:
    stat = path.stat()
    if stat.st_ino:
        return stat.st_dev, stat.st_ino
    return str(path.resolve()).casefold()


def _environment_path(value: str) -> Path:
    return Path(os.path.expandvars(value.strip())).expanduser()


def _existing_unique_directories(candidates: tuple[Path, ...]) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        identity = os.path.normcase(str(resolved))
        if resolved.is_dir() and identity not in seen:
            roots.append(resolved)
            seen.add(identity)
    return tuple(roots)


def _shard_count_from_filename(path: Path) -> int | None:
    match = _SHARD_PATTERN.match(path.name)
    return int(match.group("count")) if match else None


def _model_file_size(path: Path, shard_count: int) -> tuple[int, tuple[int, ...]]:
    match = _SHARD_PATTERN.match(path.name)
    if match is None or shard_count == 1:
        return path.stat().st_size, ()

    total_size = 0
    missing: list[int] = []
    for number in range(1, shard_count + 1):
        shard = path.with_name(f"{match.group('base')}-{number:05d}-of-{shard_count:05d}.gguf")
        if shard.is_file():
            total_size += shard.stat().st_size
        else:
            missing.append(number)
    return total_size, tuple(missing)
