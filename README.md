# nearai-bench

Benchmarking harness for evaluating AI agents. Extracted from [ironclaw](https://github.com/nearai/ironclaw).

## Available Suites

| Suite | Description |
|-------|-------------|
| `trajectory` | Multi-turn trajectory scenarios with per-turn assertions (supersedes `spot`) |
| `spot` | End-to-end spot checks: conversation, tool use, chaining, robustness |
| `custom` | Custom JSONL tasks with flexible scoring (exact, contains, regex, LLM) |
| `gaia` | GAIA benchmark (knowledge and reasoning) |
| `tau_bench` | Tau-bench (multi-turn tool-calling dialog) |
| `swe_bench` | SWE-bench Pro (real-world software engineering) |

## Quick Start

```bash
# 1. Configure your LLM provider (pick one)
cp .env.example .env
# Edit .env with your API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or LLM_* vars)

# 2. List available suites
nearai-bench list

# 3. Run trajectory scenarios
nearai-bench run --suite trajectory --config suites/trajectory.toml

# Run with a specific model
nearai-bench run --suite trajectory --config suites/trajectory.toml --model gpt-4o

# View latest results
nearai-bench results latest

# Compare two runs
nearai-bench compare <baseline-uuid> <comparison-uuid>
```

## LLM Provider Setup

Copy `.env.example` to `.env` and set your provider credentials. The harness
supports any OpenAI-compatible API endpoint.

**OpenAI** (simplest):
```bash
OPENAI_API_KEY=sk-...
```

**Anthropic**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Any OpenAI-compatible provider** (OpenRouter, Together, vLLM, Ollama, etc.):
```bash
LLM_BACKEND=openai_compatible
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=anthropic/claude-sonnet-4
```

**NEAR AI** (requires ironclaw onboarding):
```bash
LLM_BACKEND=nearai
```

## Project Structure

```
benchmarks/
  datasets/          Versioned benchmark datasets
    spot/v1/           21 spot-check tasks
    swe-bench-lite/v1/ SWE-bench Lite dataset (astropy subset)
  suites/            Suite configuration files (TOML)
  baselines/         Curated reference results by suite
  results/           Run output, namespaced by harness
    ironclaw/          Results from the ironclaw harness
  src/               Harness source code
    adapters/          Suite adapter implementations
```

## Datasets

Datasets live under `datasets/{suite-name}/v{N}/tasks.jsonl`. The versioning scheme lets
datasets evolve without invalidating older results that reference a prior version.

## Adding a New Dataset

1. Create `datasets/{name}/v1/tasks.jsonl` in the appropriate JSONL format.
2. Create `suites/{name}.toml` pointing `suite_config.dataset_path` at the new file.
3. If the suite type doesn't exist, implement a new adapter in `src/adapters/`.

## Results

Results are written to `results/{harness}/{run-uuid}/` containing:

- `run.json`: aggregate metrics (pass rate, cost, timing, model, harness)
- `tasks.jsonl`: per-task results with scores, traces, and responses

The `harness` field in `run.json` identifies which agent implementation produced the results,
allowing multiple harnesses to share the same results directory structure.

## Configuration

Suite configs are TOML files with this structure:

```toml
task_timeout = "120s"
parallelism = 1

[[matrix]]
label = "default"
# model = "openai/gpt-4o"  # optional model override

[suite_config]
dataset_path = "datasets/spot/v1/tasks.jsonl"
```

## License

MIT OR Apache-2.0
