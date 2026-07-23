from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO

import pytest
import yaml
from typer.testing import CliRunner

from llmbench.cli import app
from llmbench.config import ModelConfig
from llmbench.discovery import (
    DiscoveryError,
    default_model_roots,
    discover_models,
    inspect_gguf,
    write_model_configs,
)

_UINT32 = 4
_STRING = 8
_ARRAY = 9


def test_inspects_embedded_model_metadata(tmp_path: Path) -> None:
    model = tmp_path / "Qwen-Test-7B-Q4_K_M.gguf"
    _write_test_gguf(
        model,
        [
            ("general.architecture", _STRING, "qwen3"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Qwen Test 7B Instruct"),
            ("general.basename", _STRING, "Qwen-Test"),
            ("general.size_label", _STRING, "7B"),
            ("general.finetune", _STRING, "Instruct"),
            ("general.file_type", _UINT32, 15),
            ("general.quantization_version", _UINT32, 2),
            ("qwen3.context_length", _UINT32, 32768),
            ("qwen3.embedding_length", _UINT32, 4096),
            ("qwen3.block_count", _UINT32, 32),
            ("general.source.repo_url", _STRING, "https://example.com/model"),
            ("general.license", _STRING, "apache-2.0"),
            ("tokenizer.ggml.tokens", _ARRAY, ["one", "two"]),
        ],
        tensor_count=321,
    )

    info = inspect_gguf(model)

    assert info.name == "Qwen Test 7B Instruct"
    assert info.architecture == "qwen3"
    assert info.quantization == "Q4_K_M"
    assert info.quantization_version == 2
    assert info.context_length == 32768
    assert info.embedding_length == 4096
    assert info.block_count == 32
    assert info.gguf_version == 3
    assert info.tensor_count == 321
    assert info.source_url == "https://example.com/model"
    assert info.license == "apache-2.0"
    assert info.is_runnable_model


def test_uses_descriptive_filename_when_embedded_name_is_generic(tmp_path: Path) -> None:
    model = tmp_path / "DreamOmni2-Vlm-Model-7.6B-Q4_K_M.gguf"
    _write_test_gguf(
        model,
        [
            ("general.architecture", _STRING, "qwen2vl"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Vlm Model"),
            ("general.size_label", _STRING, "7.6B"),
            ("general.file_type", _UINT32, 15),
        ],
    )

    result = discover_models(tmp_path)

    assert result.models[0].config.id == "dreamomni2-vlm-model-7.6b-q4-k-m"
    assert result.models[0].config.display_name == "DreamOmni2 Vlm Model 7.6B Q4_K_M"
    assert result.models[0].config.metadata.family == "DreamOmni2"


def test_discovers_primary_shard_and_skips_projector(tmp_path: Path) -> None:
    primary = tmp_path / "qwen-test-q4_k_m-00001-of-00002.gguf"
    secondary = tmp_path / "qwen-test-q4_k_m-00002-of-00002.gguf"
    projector = tmp_path / "mmproj-qwen-test-f16.gguf"
    _write_test_gguf(
        primary,
        [
            ("general.architecture", _STRING, "qwen3"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Qwen Test 7B"),
            ("general.basename", _STRING, "Qwen-Test"),
            ("general.size_label", _STRING, "7B"),
            ("general.file_type", _UINT32, 15),
            ("split.no", _UINT32, 0),
            ("split.count", _UINT32, 2),
        ],
    )
    secondary.write_bytes(b"secondary shard contents")
    _write_test_gguf(
        projector,
        [
            ("general.architecture", _STRING, "clip"),
            ("general.type", _STRING, "mmproj"),
            ("general.name", _STRING, "Qwen Test Projector"),
            ("general.file_type", _UINT32, 1),
        ],
    )

    result = discover_models(tmp_path)

    assert len(result.models) == 1
    discovered = result.models[0]
    assert discovered.config.id == "qwen-test-7b-q4-k-m"
    assert discovered.config.gguf_path == primary.resolve()
    assert discovered.config.metadata.parameters_billion == 7
    assert discovered.config.metadata.shard_count == 2
    assert discovered.config.metadata.file_size_bytes == (
        primary.stat().st_size + secondary.stat().st_size
    )
    assert any("non-primary shard" in issue.message for issue in result.issues)
    assert any("sidecar" in issue.message for issue in result.issues)


def test_writes_valid_configs_and_protects_existing_files(tmp_path: Path) -> None:
    model = tmp_path / "model-q6_k.gguf"
    _write_test_gguf(
        model,
        [
            ("general.architecture", _STRING, "qwen35"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Example 9B"),
            ("general.size_label", _STRING, "9B"),
            ("general.file_type", _UINT32, 18),
        ],
    )
    result = discover_models(tmp_path)
    output = tmp_path / "configs"

    paths = write_model_configs(result.models, output)

    assert len(paths) == 1
    document = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    config = ModelConfig.model_validate(document)
    assert config.id == "example-9b-q6-k"
    assert config.gguf_path == model.resolve()
    assert config.metadata.architecture == "qwen35"

    with pytest.raises(DiscoveryError, match="refusing to overwrite"):
        write_model_configs(result.models, output)

    write_model_configs(result.models, output, overwrite=True)


def test_discover_command_is_dry_run_until_write_is_requested(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model-q4_k_m.gguf"
    _write_test_gguf(
        model,
        [
            ("general.architecture", _STRING, "llama"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Example 8B"),
            ("general.size_label", _STRING, "8B"),
            ("general.file_type", _UINT32, 15),
        ],
    )
    output = tmp_path / "local-configs"
    runner = CliRunner()

    dry_run = runner.invoke(
        app,
        ["models", "discover", str(model_root), "--output", str(output)],
    )

    assert dry_run.exit_code == 0
    assert "Dry run: no files written" in dry_run.stdout
    assert not output.exists()

    written = runner.invoke(
        app,
        [
            "models",
            "discover",
            str(model_root),
            "--output",
            str(output),
            "--write",
        ],
    )

    assert written.exit_code == 0
    assert "Wrote 1 model config" in written.stdout
    assert (output / "example-8b-q4-k-m.yaml").is_file()


def test_detects_model_roots_from_portable_environment_variable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    roots = default_model_roots(
        environ={"LLMBENCH_MODEL_PATHS": f"{first}{os.pathsep}{second}"},
        home=tmp_path / "home",
        cwd=tmp_path / "work",
    )

    assert roots == (first.resolve(), second.resolve())


def test_discover_command_can_use_automatic_model_roots(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    _write_test_gguf(
        model_root / "automatic-3b-q4_k_m.gguf",
        [
            ("general.architecture", _STRING, "llama"),
            ("general.type", _STRING, "model"),
            ("general.name", _STRING, "Automatic 3B"),
            ("general.size_label", _STRING, "3B"),
            ("general.file_type", _UINT32, 15),
        ],
    )

    result = CliRunner().invoke(
        app,
        ["models", "discover"],
        env={"LLMBENCH_MODEL_PATHS": str(model_root)},
    )

    assert result.exit_code == 0
    assert "Searching model root(s):" in result.stdout
    assert str(model_root.resolve()) in result.stdout
    assert "automatic-3b-q4-k-m" in result.stdout


def _write_test_gguf(
    path: Path,
    metadata: list[tuple[str, int, object]],
    *,
    tensor_count: int = 0,
) -> None:
    with path.open("wb") as stream:
        stream.write(b"GGUF")
        stream.write(struct.pack("<IQQ", 3, tensor_count, len(metadata)))
        for key, value_type, value in metadata:
            _write_string(stream, key)
            stream.write(struct.pack("<I", value_type))
            _write_value(stream, value_type, value)


def _write_value(stream: BinaryIO, value_type: int, value: object) -> None:
    if value_type == _STRING:
        assert isinstance(value, str)
        _write_string(stream, value)
    elif value_type == _UINT32:
        assert isinstance(value, int)
        stream.write(struct.pack("<I", value))
    elif value_type == _ARRAY:
        assert isinstance(value, list)
        stream.write(struct.pack("<IQ", _STRING, len(value)))
        for item in value:
            assert isinstance(item, str)
            _write_string(stream, item)
    else:
        raise AssertionError(f"unsupported test GGUF type: {value_type}")


def _write_string(stream: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)
