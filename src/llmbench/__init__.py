"""Local LLM benchmark runner."""

APP_VERSION = "0.2.0.dev0"
RESULT_SCHEMA_VERSION = 1
RAW_BENCHMARK_PROTOCOL_VERSION = "raw-v1"
INTERACTIVE_BENCHMARK_PROTOCOL_VERSION = "interactive-v1"

# Kept as an import-compatible alias for v0.1 integrations. New code should
# select a protocol from the benchmark track instead.
BENCHMARK_PROTOCOL_VERSION = RAW_BENCHMARK_PROTOCOL_VERSION

__all__ = [
    "APP_VERSION",
    "BENCHMARK_PROTOCOL_VERSION",
    "INTERACTIVE_BENCHMARK_PROTOCOL_VERSION",
    "RAW_BENCHMARK_PROTOCOL_VERSION",
    "RESULT_SCHEMA_VERSION",
]
