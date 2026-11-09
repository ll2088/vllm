# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys

import pytest

BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(BENCHMARKS_DIR, script), *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "script,field",
    [
        ("benchmark_throughput.py", "throughput"),
        ("benchmark_latency.py", "latency"),
        ("benchmark_prefix_caching.py", "prefix_cache_hit_rate"),
    ],
)
def test_output_json_option(tmp_path, script, field):
    out = tmp_path / "out.json"
    proc = _run(script, "--output-json", str(out))
    assert proc.returncode in (0, 1)
    assert out.exists()
    data = json.loads(out.read_text())
    assert field in data


def test_help_contains_output_json():
    for script in (
        "benchmark_throughput.py",
        "benchmark_latency.py",
        "benchmark_prefix_caching.py",
    ):
        proc = _run(script, "--help")
        assert proc.returncode == 0
        assert "--output-json" in proc.stdout


def test_run_baseline_help():
    proc = _run("run_baseline.py", "--help")
    assert proc.returncode == 0
    assert "--output" in proc.stdout


def test_run_baseline_aggregates(tmp_path):
    out = tmp_path / "baseline.json"
    proc = _run("run_baseline.py", "--output", str(out))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    for key in ("throughput", "latency", "prefix_cache_hit_rate"):
        assert key in data


def test_run_baseline_failure_returns_nonzero(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baseline", os.path.join(BENCHMARKS_DIR, "run_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "run_baseline", _fail)
    out = tmp_path / "baseline.json"
    rc = module.main(["--output", str(out)])
    assert rc != 0
    assert not out.exists()
