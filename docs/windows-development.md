# Windows development setup

The benchmark uses a project-local Python virtual environment. The examples
below use PowerShell from the repository root.

## Create and activate the environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation scripts, the virtual environment can still be
used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\llmbench.exe --help
```

## Configure local paths

The committed smoke configuration contains placeholder paths:

- `configs/models/example-model.yaml`
- `configs/profiles/vulkan-f16.yaml`

Discover local GGUF files before creating the experiment. Discovery is a dry
run unless `--write` is supplied:

```powershell
llmbench models discover
llmbench models discover --write
```

Automatic discovery checks `LLMBENCH_MODEL_PATHS`, `LLAMA_CACHE`, the
configured or default Hugging Face Hub cache, the repository's `models`
directory, and `models` or `Models` under the user home directory. It does not
scan entire drives.

Pass a nonstandard directory explicitly:

```powershell
llmbench models discover "D:\Models"
```

Or configure multiple directories for the current shell:

```powershell
$env:LLMBENCH_MODEL_PATHS = "D:\Models;E:\Shared\Models"
llmbench models discover
```

The generated model configurations are stored in `configs/local/models/`.
That directory is ignored by Git, so absolute local paths are not published.
Projectors, adapters, duplicate hard links, and non-primary model shards are
reported but not emitted as runnable model configurations.

Inspect the complete embedded metadata for one GGUF at any time:

```powershell
llmbench models inspect "D:\Models\model.gguf"
```

Create local runtime and experiment copies:

```powershell
Copy-Item configs\profiles\vulkan-f16.yaml configs\local\runtime.yaml
Copy-Item configs\experiments\raw-smoke.yaml configs\local\raw-smoke.yaml
```

In `configs/local/raw-smoke.yaml`, change the references to:

```yaml
model: models/<discovered-model-id>.yaml
runtime: runtime.yaml
```

Then edit:

- `configs/local/runtime.yaml`: set `llama_bench_path` and optionally
  `expected_commit`.

As a manual fallback, copy `configs/models/example-model.yaml` into
`configs/local/models/` and set its `gguf_path` yourself.

## Validate and run

```powershell
llmbench doctor --config configs\local\raw-smoke.yaml
llmbench plan --config configs\local\raw-smoke.yaml
llmbench raw run --config configs\local\raw-smoke.yaml
llmbench raw report --latest --config configs\local\raw-smoke.yaml
```

The smoke experiment uses only two repetitions. Increase it to seven after the
complete execution and reporting pipeline works with the selected model.

## Development checks

```powershell
pytest
ruff check .
ruff format --check .
mypy
```
