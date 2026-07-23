# Local LLM Benchmark

A reproducible benchmark for comparing local coding models on a Windows 11 PC
with an AMD Radeon RX 6700 XT (12 GB VRAM), using `llama.cpp` and the Vulkan
backend.

The project focuses on a practical question:

> Which local model most reliably completes realistic React, Next.js,
> TypeScript, and Node.js tasks within acceptable time and VRAM limits?

This project is currently a work in progress. See
[local-llm-benchmark-plan.md](local-llm-benchmark-plan.md) for the complete
benchmark design.

Implementation is divided into independently useful releases. See
[ROADMAP.md](ROADMAP.md) for the version overview and detailed milestone plans.

## Benchmark tracks

The benchmark keeps quality and performance results separate instead of
combining them into one opaque score.

1. **Raw inference performance** - prompt processing, generation speed, context
   scaling, RAM, and VRAM.
2. **Interactive performance** - time to first token, total response time, and
   prompt-cache behavior through `llama-server`.
3. **General reasoning** - reasoning, instruction following, comprehension, and
   long-context retrieval.
4. **Model-only coding** - one-shot TypeScript, React, Next.js, and Node.js
   tasks with executable tests.
5. **Agentic repository coding** - realistic repository tasks completed with
   the same tools, permissions, token budget, and iteration limit.

## Test system

| Component | Configuration |
|---|---|
| Operating system | Windows 11 |
| GPU | AMD Radeon RX 6700 XT |
| VRAM | 12 GB |
| Inference engine | llama.cpp |
| Backend | Vulkan |
| Primary workload | React, Next.js, TypeScript, Node.js, debugging, and refactoring |

## Planned project structure

```text
local-llm-benchmark/
|-- tasks/          # Public task descriptions and starter repositories
|-- hidden-tests/   # Private functional and regression tests
|-- runner/         # Benchmark and scoring automation
|-- results/        # Raw and summarized benchmark results
`-- local-llm-benchmark-plan.md
```

Each repository task will start from a clean Git commit. Models may inspect the
source, public tests, and task description, but not hidden tests, reference
implementations, or exact scoring conditions.

## Getting started

The benchmark will be built in four phases:

1. Establish a raw performance baseline with `llama-bench`.
2. Add a small deterministic reasoning and one-shot coding suite.
3. Add repository-level coding tasks with hidden tests.
4. Run deeper evaluations only for finalist models.

### Raw performance example

Prompt processing:

```powershell
.\llama-bench.exe `
  -m "D:\Models\model.gguf" `
  -ngl 99 `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -n 0 `
  -p 512,2048,8192 `
  -d 0,2048,8192,16384 `
  -r 7 `
  -o json > "results\model-prompt.json"
```

Generation:

```powershell
.\llama-bench.exe `
  -m "D:\Models\model.gguf" `
  -ngl 99 `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -p 0 `
  -n 128,512 `
  -d 0,2048,8192,16384 `
  -r 7 `
  -o json > "results\model-generation.json"
```

Interactive testing:

```powershell
.\llama-server.exe `
  -m "D:\Models\model.gguf" `
  -ngl 99 `
  -c 16384 `
  -fa on `
  --jinja `
  --metrics `
  --port 8080
```

Model paths and result filenames should be changed for each test. Models that
barely fit in VRAM will also be tested with a quantized `q8_0` KV cache.

## Primary metrics

The final comparison will prioritize:

- Agentic web-development Pass@1
- Model-only coding Pass@1
- Successfully completed repository tasks per hour
- General-reasoning accuracy
- Warm time to first token (TTFT)
- Generation speed at 8K context
- Peak VRAM usage

Raw performance tests should use at least seven repetitions. Quality tasks
should use Pass@1 as the primary metric and three runs when practical to measure
consistency.

## Fair comparison rules

Every model must use the same:

- `llama.cpp` build and commit
- Vulkan backend and context limit
- Chat template, system prompt, and tool definitions
- Sampling profile and output-token limit
- Timeout, agent iteration budget, and repository commit
- Public and hidden tests

Each benchmark entry represents the complete configuration, not just the model:

```text
model + quantization + context size + KV cache + GPU layers
+ flash attention + llama.cpp build
```

Runs that disable validation, modify tests, bypass TypeScript or lint rules, or
hardcode visible examples will be rejected or penalized.

## Results

Results will remain multidimensional so that speed, reliability, reasoning, and
resource usage can be compared directly.

| Model | General reasoning | Coding Pass@1 | Agent Pass@1 | TTFT | Decode t/s | 8K decode t/s | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| _To be tested_ | - | - | - | - | - | - | - |

Raw result files will include the model, GGUF quantization, `llama.cpp` commit,
context and KV-cache settings, sampling configuration, timing data, quality
scores, and hardware utilization required to reproduce each run.

## License

No license has been selected yet.
