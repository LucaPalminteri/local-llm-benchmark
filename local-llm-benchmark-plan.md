# Local LLM Benchmark Plan

## My setup

- **Operating system:** Windows 11
- **Inference engine:** llama.cpp
- **GPU:** AMD Radeon RX 6700 XT
- **VRAM:** 12 GB
- **Backend:** Vulkan
- **Main use case:** coding, especially React, Next.js, TypeScript, Node.js, debugging, refactoring, and repository-level work

## Benchmark goal

The benchmark should separate four different questions:

1. **How intelligent is the model in general?**
2. **How good is it at coding?**
3. **How useful is it as a coding agent inside a real repository?**
4. **How fast and resource-efficient is it on my hardware?**

These should not be compressed into one generic score. A model can be faster but less reliable, or slower but substantially better at completing real coding tasks.

---

# Benchmark tracks

## 1. Raw inference performance

Use `llama-bench` to measure the raw performance of each GGUF model and configuration.

### Metrics

- Prompt-processing speed in tokens per second
- Generation speed in tokens per second
- Performance at different context depths
- Standard deviation between runs
- GPU-offloaded layers
- Peak VRAM usage
- Peak system RAM usage

### Prompt-processing benchmark

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

### Generation benchmark

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

### Why context depth matters

Coding agents gradually accumulate source code, tool output, patches, test logs, and conversation history. A model that generates quickly with an empty context may become much slower after 8K or 16K tokens.

For coding use, record performance at least at:

- Empty context
- 2K context
- 8K context
- 16K context

### KV-cache configurations

If a model barely fits in 12 GB of VRAM, also test quantized KV cache:

```powershell
-ctk q8_0 -ctv q8_0
```

Treat each configuration as a separate benchmark entry:

```text
Qwen 14B Q4_K_M / F16 KV cache
Qwen 14B Q4_K_M / Q8_0 KV cache
```

The real benchmarked unit is:

```text
Model + quantization + context size + KV cache type + GPU layers
+ flash attention setting + llama.cpp build
```

---

## 2. Real interactive performance

Use `llama-server` to measure how the model actually feels through a chat UI, editor integration, or coding agent.

### Example server command

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

### Metrics

- Cold startup time
- First request after loading
- Warm time to first token
- Prompt-processing speed
- Generation speed
- End-to-end completion time
- Prompt-cache reuse
- Peak RAM and VRAM

### Time to first token

Measure TTFT in the client:

```text
TTFT = timestamp of first streamed content token - request start timestamp
```

Run at least ten warm requests and record:

- Median TTFT
- p95 TTFT
- Median total response time
- Median generation speed

Do not rely only on averages because they can hide occasional slow responses.

### Prompt sizes

Use three fixed prompt sizes:

```text
Small: approximately 500 input tokens
Medium: approximately 4,000 input tokens
Large: approximately 12,000 input tokens
```

Generate a fixed number of output tokens, such as 256, for every model.

---

## 3. General reasoning and intelligence

Use a combination of standardized tests and parametric reasoning tests.

### Standardized evaluation

A framework such as `lm-evaluation-harness` can run tasks through a local OpenAI-compatible `llama-server` endpoint.

Evaluate a compact selection covering:

- Logical reasoning
- Mathematical reasoning
- Instruction following
- Reading comprehension
- Factual knowledge
- Long-context retrieval

A practical suite should contain around 300 to 1,000 questions. Running many overlapping benchmarks usually adds cost without adding much useful information.

### Parametric reasoning

ReasonScape is useful when comparing:

- Different model families
- Thinking and non-thinking models
- Large quantization differences
- Models that look tied on traditional benchmarks

Instead of only giving one accuracy score, parametric benchmarks can reveal where the model starts failing as problem difficulty increases.

### General-reasoning metrics

Record:

- Accuracy
- Pass@1
- Consistency across three runs
- Truncation rate
- Median generated tokens for correct answers
- Median generated tokens for incorrect answers
- Correct answers per 100,000 generated tokens

The last metric helps compare reasoning efficiency. A model that gains a few accuracy points by producing thousands of reasoning tokens may not be the best local option.

---

## 4. Coding intelligence

Separate coding evaluation into two categories.

## 4.1 Model-only coding

The model receives one prompt and returns code or a patch. It cannot inspect files, run commands, or retry.

This measures:

- First-attempt correctness
- TypeScript knowledge
- React and Next.js understanding
- Code comprehension
- Instruction following
- Ability to detect bugs

### Suggested task distribution

| Category | Tasks |
|---|---:|
| TypeScript functions and types | 10 |
| React bugs and hooks | 8 |
| Next.js App Router | 8 |
| Node.js and APIs | 6 |
| Code review and security | 4 |
| Test generation | 4 |

Total: approximately 40 tasks.

### Example tasks

- Fix a stale closure inside `useEffect`
- Correct a discriminated union
- Fix a hydration mismatch
- Synchronize filters with URL search parameters
- Implement a validated API route using Zod
- Prevent stale search results from overwriting newer ones
- Detect an authorization vulnerability
- Write Vitest tests for existing behavior

Score the returned code with hidden executable tests whenever possible.

---

## 4.2 Agentic repository coding

Give every model the same coding agent, repository tools, command permissions, iteration limit, and token budget.

This is the most important benchmark for real coding usefulness.

### Recommended tools

- llama.cpp server
- Aider, OpenCode, or a custom coding agent
- Docker or isolated repository copies
- TypeScript
- ESLint
- Vitest
- Playwright
- Next.js production build

### Suggested repository structure

```text
benchmark/
├── tasks/
│   ├── react-stale-closure/
│   ├── nextjs-search-params/
│   ├── nextjs-cache-revalidation/
│   ├── node-auth-bug/
│   └── typescript-generic-table/
├── hidden-tests/
├── runner/
└── results/
```

Each task starts from a clean Git commit.

The model may see:

- Source code
- Existing public tests
- Task description

The model must not see:

- Hidden tests
- Reference implementation
- Exact scoring conditions
- Expected patch

### Validation commands

```bash
npm install
npm run typecheck
npm run lint
npm test
npm run build
npm run test:e2e
```

### Suggested score per task

| Check | Points |
|---|---:|
| Hidden functional tests | 50 |
| Typecheck and production build | 15 |
| Playwright browser tests | 15 |
| Explicit requirement checks | 10 |
| No forbidden test/config changes | 5 |
| Patch quality | 5 |

A full pass should require:

- Hidden tests pass
- Production build passes
- No regressions
- No tests are removed or disabled
- TypeScript and linting are not bypassed

### Prevent benchmark cheating

Reject or penalize runs that:

- Modify hidden or public tests
- Disable TypeScript checks
- Add broad `eslint-disable` comments
- Change package scripts to skip validation
- Delete failing tests
- Hardcode values specifically for visible examples

Inspect `git diff` after every run.

---

# Build a private web-development benchmark

Public coding benchmarks are useful, but many focus on competitive programming, Python repositories, or isolated algorithms. They do not accurately measure real Next.js, React, TypeScript, and Node.js work.

Create private tasks based on:

- Bugs previously solved
- Simplified Jira tickets
- Common PR-review comments
- Problems from old personal projects
- Artificially broken Next.js repositories
- Features that would normally take 20 to 60 minutes manually

Do not copy proprietary company code. Recreate only the underlying technical problem in a small artificial project.

### Recommended benchmark composition

For 30 repository tasks:

- 6 React tasks
- 8 Next.js tasks
- 6 Node.js tasks
- 7 debugging tasks
- 3 refactoring tasks

---

# Fair-comparison rules

Every model must use the same:

- llama.cpp build and commit
- Vulkan backend
- Context limit
- Chat template
- System prompt
- Tool definitions
- Maximum output tokens
- Timeout
- Agent iteration budget
- Repository commit
- Hidden tests
- Sampling profile

Do not compare one model through Open WebUI and another through Aider. That would compare both the model and the agent implementation.

## Sampling profiles

### Deterministic profile

```text
temperature = 0
fixed seed
one attempt
```

Use this for:

- Pass@1 comparisons
- Quantization comparisons
- Regression tests

### Recommended/native profile

Use the model author's recommended sampling settings and run three attempts.

Use this to evaluate how the model performs under its intended configuration.

Always report output-token limits. A thinking model allowed 8,000 tokens should not be directly compared with another model limited to 512 tokens without showing that difference.

---

# Repetitions and stability

Run each quality task three times when practical.

Record:

- **Pass@1:** whether the first attempt succeeds
- **Average score:** mean score across attempts
- **Consistency:** percentage of tasks passed in every run
- **Best-of-3:** useful, but secondary

Pass@1 should be the primary quality metric because it reflects how often the model produces a usable first solution.

For raw performance benchmarks, use at least seven repetitions and record the median and standard deviation.

---

# Final dashboard

Do not create one opaque benchmark score. Keep the important dimensions visible.

| Model | General reasoning | Coding one-shot | Web agent Pass@1 | TTFT | Decode t/s | 8K decode t/s | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A | 68% | 62% | 48% | 1.2 s | 31 | 25 | 10.8 GB |
| Model B | 65% | 70% | 63% | 1.8 s | 22 | 17 | 11.5 GB |

Also calculate:

```text
Successfully completed repository tasks per hour
```

This is a useful practical metric because it naturally combines correctness and total execution time.

## Recommended ranking priority

For my use case, rank models primarily by:

1. Agentic web-development Pass@1
2. Model-only coding Pass@1
3. Successfully completed tasks per hour
4. General-reasoning accuracy
5. Warm TTFT
6. Generation speed at 8K context
7. VRAM usage

---

# Suggested results schema

Each result should include enough configuration data to reproduce it.

```json
{
  "model": "Qwen model name",
  "quantization": "Q5_K_M",
  "gguf_file": "model-file.gguf",
  "llama_cpp_commit": "commit-hash",
  "backend": "Vulkan",
  "gpu": "AMD Radeon RX 6700 XT",
  "vram_gb": 12,
  "context_size": 16384,
  "gpu_layers": 99,
  "flash_attention": true,
  "kv_cache_key": "f16",
  "kv_cache_value": "f16",
  "temperature": 0,
  "seed": 42,
  "prompt_tokens": 4096,
  "output_limit": 512,
  "ttft_ms_median": 0,
  "ttft_ms_p95": 0,
  "prompt_tokens_per_second": 0,
  "generation_tokens_per_second": 0,
  "peak_vram_mb": 0,
  "general_reasoning_accuracy": 0,
  "coding_one_shot_pass_at_1": 0,
  "agentic_web_pass_at_1": 0,
  "tasks_completed_per_hour": 0
}
```

---

# Practical implementation order

## Phase 1: performance baseline

1. Create a fixed folder for models and results.
2. Record the llama.cpp build or commit.
3. Benchmark prompt processing with `llama-bench`.
4. Benchmark generation at empty, 2K, 8K, and 16K context.
5. Measure warm TTFT through `llama-server`.
6. Record VRAM and RAM usage.

## Phase 2: small quality suite

1. Create 10 general-reasoning tasks.
2. Create 15 TypeScript/React/Next.js one-shot tasks.
3. Add hidden tests.
4. Run every model once with deterministic settings.
5. Investigate major differences manually.

## Phase 3: repository benchmark

1. Create 5 small repository-level tasks.
2. Run every model through the same agent.
3. Score with typecheck, tests, build, and Playwright.
4. Record tokens, time, iterations, and final patch.
5. Expand gradually to 30 tasks.

## Phase 4: deeper evaluation

Use larger general-reasoning suites or ReasonScape only for finalists or models whose results are difficult to distinguish.

---

# Main principle

The most important question is not:

> Which model has the highest public benchmark score?

It is:

> Which model most reliably completes realistic React, Next.js, TypeScript, and Node.js tasks on my RX 6700 XT within an acceptable amount of time and VRAM?

For this setup, the most meaningful final metrics are:

```text
Agentic coding Pass@1
Successful repository tasks per hour
Generation speed at 8K context
Warm time to first token
Peak VRAM usage
```
