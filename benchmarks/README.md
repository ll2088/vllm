# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Unified baseline snapshot

`benchmarks/run_baseline.py` runs the throughput, latency and prefix caching
benchmarks as subprocesses and aggregates their structured JSON output into a
single file with stable top-level fields (`throughput`, `latency`,
`prefix_cache_hit_rate`):

```bash
python benchmarks/run_baseline.py --output /tmp/baseline.json
```

Each benchmark script also supports `--output-json <path>` for standalone use.
