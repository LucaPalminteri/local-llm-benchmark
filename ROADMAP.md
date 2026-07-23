# Local LLM Benchmark Roadmap

This roadmap turns the full benchmark vision in
[local-llm-benchmark-plan.md](local-llm-benchmark-plan.md) into small,
independently useful releases.

The project should produce trustworthy benchmark data before it produces a
large task suite or a polished dashboard. Each version therefore adds one new
capability while preserving the raw evidence needed to audit earlier results.

## Release overview

| Version | Status | Name | Main outcome |
|---|---|---|---|
| [v0.1](docs/plans/v0.1-raw-performance.md) | In progress | Raw Performance MVP | Run reproducible `llama-bench` experiments from a Python CLI |
| [v0.2](docs/plans/v0.2-interactive-performance.md) | Planned | Interactive Performance | Measure real requests through `llama-server`, including TTFT |
| [v0.3](docs/plans/v0.3-system-telemetry.md) | Planned | System Telemetry | Correlate performance with RAM, VRAM, and process measurements |
| [v0.4](docs/plans/v0.4-model-quality.md) | Planned | Model-Only Quality | Evaluate reasoning and one-shot coding with deterministic graders |
| [v0.5](docs/plans/v0.5-agentic-coding.md) | Planned | Agentic Coding | Evaluate models working in clean repositories through one fixed agent |
| [v0.6](docs/plans/v0.6-dashboard.md) | Planned | Results Dashboard | Explore models, configurations, runs, and regressions visually |
| [v1.0](docs/plans/v1.0-stable-benchmark.md) | Planned | Stable Benchmark | Publish a documented, repeatable end-to-end benchmark workflow |

### Status values

- **Planned:** work has not started.
- **In progress:** this is the current active release.
- **Blocked:** work started but cannot currently continue.
- **Completed:** all acceptance criteria and the definition of done are met.

## Version dependency

```text
v0.1 Raw performance
  |
  v
v0.2 Interactive performance
  |
  v
v0.3 System telemetry
  |
  v
v0.4 Model-only quality
  |
  v
v0.5 Agentic coding
  |
  v
v0.6 Dashboard
  |
  v
v1.0 Stable benchmark
```

The sequence is deliberate:

1. Prove that configurations, execution, artifacts, and statistics work.
2. Add server lifecycle and client-observed latency.
3. Add hardware telemetry without mixing it into inference logic.
4. Reuse the same run system for deterministic quality evaluation.
5. Add the more complex repository-agent environment.
6. Build the dashboard on top of stable schemas and useful accumulated data.

## Rules that apply to every version

- A benchmark target is the complete model and runtime configuration, not only
  a model name.
- Raw artifacts are immutable and summaries can be regenerated from them.
- Missing measurements use `null`; zero is reserved for a measured zero.
- Failed, timed-out, and out-of-memory attempts are stored as results.
- Every stored result has a schema version.
- Every run records the exact command, configuration, model file, engine build,
  timestamps, and relevant environment information.
- Reports never become the source of truth.
- New functionality includes automated tests for parsing, validation, scoring,
  or aggregation as appropriate.

## Suggested release policy

Use semantic versions while the project is experimental:

- Patch releases fix behavior without changing result meaning.
- Minor releases add benchmark tracks or new schema capabilities.
- Any incompatible result-schema change increments the schema version even
  when the application remains in `0.x`.

Do not promise result comparability across a change that affects prompts,
sampling, chat templates, grading, or runtime settings. Mark such changes as a
new benchmark protocol version.

## What is intentionally deferred

The following should not be part of `v0.1`:

- A web dashboard
- A custom coding agent
- Public benchmark integrations
- A large task library
- Distributed execution
- Automatic model downloading
- A single combined model score

These may be useful later, but none are necessary to establish a reliable raw
performance baseline.
