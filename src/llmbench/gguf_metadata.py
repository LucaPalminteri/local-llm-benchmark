from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_GGUF_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = {2, 3}

_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

_SCALAR_FORMATS = {
    _UINT8: "B",
    _INT8: "b",
    _UINT16: "H",
    _INT16: "h",
    _UINT32: "I",
    _INT32: "i",
    _FLOAT32: "f",
    _BOOL: "?",
    _UINT64: "Q",
    _INT64: "q",
    _FLOAT64: "d",
}


class GGUFMetadataError(ValueError):
    """Raised when a GGUF header or metadata table is invalid or unsupported."""


@dataclass(frozen=True)
class GGUFMetadata:
    version: int
    tensor_count: int
    values: dict[str, object]


def read_gguf_metadata(
    path: Path,
    keys: frozenset[str],
    *,
    stop_when_found: bool = False,
) -> GGUFMetadata:
    """Read selected GGUF metadata values without mapping or loading tensors."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            if _read_exact(stream, 4) != _GGUF_MAGIC:
                raise GGUFMetadataError("file does not start with the GGUF magic bytes")
            version = _read_scalar(stream, _UINT32)
            if version not in _SUPPORTED_VERSIONS:
                raise GGUFMetadataError(f"unsupported GGUF version: {version}")
            tensor_count = _read_scalar(stream, _UINT64)
            metadata_count = _read_scalar(stream, _UINT64)
            values: dict[str, object] = {}
            for _ in range(metadata_count):
                key = _read_string(stream)
                value_type = _read_scalar(stream, _UINT32)
                if key in keys:
                    values[key] = _read_value(stream, value_type, file_size)
                    if stop_when_found and values.keys() >= keys:
                        break
                else:
                    _skip_value(stream, value_type, file_size)
    except OSError as error:
        raise GGUFMetadataError(str(error)) from error
    return GGUFMetadata(version=version, tensor_count=tensor_count, values=values)


def _read_value(stream: BinaryIO, value_type: int, file_size: int) -> object:
    if value_type in _SCALAR_FORMATS:
        return _read_scalar(stream, value_type)
    if value_type == _STRING:
        return _read_string(stream)
    if value_type == _ARRAY:
        element_type = _read_scalar(stream, _UINT32)
        count = _read_scalar(stream, _UINT64)
        return tuple(_read_value(stream, element_type, file_size) for _ in range(count))
    raise GGUFMetadataError(f"unsupported GGUF metadata value type: {value_type}")


def _skip_value(stream: BinaryIO, value_type: int, file_size: int) -> None:
    if value_type in _SCALAR_FORMATS:
        _skip_exact(stream, struct.calcsize("<" + _SCALAR_FORMATS[value_type]), file_size)
        return
    if value_type == _STRING:
        _skip_exact(stream, _read_scalar(stream, _UINT64), file_size)
        return
    if value_type == _ARRAY:
        element_type = _read_scalar(stream, _UINT32)
        count = _read_scalar(stream, _UINT64)
        if element_type in _SCALAR_FORMATS:
            element_size = struct.calcsize("<" + _SCALAR_FORMATS[element_type])
            _skip_exact(stream, element_size * count, file_size)
            return
        for _ in range(count):
            _skip_value(stream, element_type, file_size)
        return
    raise GGUFMetadataError(f"unsupported GGUF metadata value type: {value_type}")


def _read_string(stream: BinaryIO) -> str:
    length = _read_scalar(stream, _UINT64)
    try:
        return _read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise GGUFMetadataError("GGUF metadata contains invalid UTF-8") from error


def _read_scalar(stream: BinaryIO, value_type: int) -> int:
    format_code = _SCALAR_FORMATS[value_type]
    size = struct.calcsize(format_code)
    return int(struct.unpack("<" + format_code, _read_exact(stream, size))[0])


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        raise GGUFMetadataError("unexpected end of GGUF metadata")
    return data


def _skip_exact(stream: BinaryIO, length: int, file_size: int) -> None:
    destination = stream.tell() + length
    if destination > file_size:
        raise GGUFMetadataError("GGUF metadata extends beyond the end of the file")
    stream.seek(length, 1)
