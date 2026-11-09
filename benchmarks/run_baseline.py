# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified baseline runner for vLLM benchmarks.

Runs the throughput, latency and prefix caching benchmarks as subprocesses,
collects their structured JSON output and aggregates it into a single JSON
file with stable top-level fields.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = {
    "throughput": "benchmark_throughput.py",
    "latency": "benchmark_latency.py",
    "prefix_cache_hit_rate": "benchmark_prefix_caching.py",
}


def _run_benchmark(
    name: str,
    script: str,
    extra_args: list[str],
    timeout: float | None = None,
) -> dict:
    script_path = os.path.join(BENCHMARKS_DIR, script)
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cmd = [sys.executable, script_path, "--output-json", tmp_path]
        cmd.extend(extra_args)
        env = os.environ.copy()
        # Avoid triggering CUDA detection / source builds in lightweight CI.
        # These defaults are applied only to the benchmark subprocesses; the
        # parent process environment is left untouched.
        env.setdefault("VLLM_TARGET_DEVICE", "cpu")
        # setup.py asserts CUDA_HOME is set when probing nvcc; provide a
        # harmless placeholder so CPU-only environments do not fail during
        # any incidental build/import path.
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Benchmark '{name}' timed out after {timeout}s"
            ) from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or "<no output>"
            raise RuntimeError(
                f"Benchmark '{name}' failed with exit code "
                f"{proc.returncode}: {detail}"
            )
        try:
            with open(tmp_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Benchmark '{name}' did not produce valid JSON: {exc}"
            ) from exc
        return data
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_baseline(
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> dict:
    extra_args = extra_args or []
    results: dict = {}
    for field, script in BENCHMARKS.items():
        data = _run_benchmark(field, script, extra_args, timeout=timeout)
        results[field] = data.get(field)
    return results


def _write_results(path: str, results: dict) -> None:
    """Atomically write aggregated results so partial files are never left."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all vLLM baseline benchmarks and aggregate results "
        "into a single JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the aggregated JSON results to.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-benchmark timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to every benchmark script.",
    )
    return parser


def main(argv=None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    try:
        results = run_baseline(extra_args=args.extra_args, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _write_results(args.output, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
