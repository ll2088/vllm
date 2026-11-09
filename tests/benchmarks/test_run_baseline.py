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
    env = os.environ.copy()
    env.setdefault("VLLM_TARGET_DEVICE", "cpu")
    return subprocess.run(
        [sys.executable, os.path.join(BENCHMARKS_DIR, script), *args],
        capture_output=True,
        text=True,
        env=env,
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


def test_run_baseline_sets_pythonpath_for_subprocess(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baseline", os.path.join(BENCHMARKS_DIR, "run_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured = {}

    class _FakeCompleted:
        returncode = 0
        stderr = ""

    def _fake_run(cmd, capture_output=True, text=True, env=None):
        captured["env"] = env
        # Write valid JSON to the --output-json path.
        out_idx = cmd.index("--output-json") + 1
        with open(cmd[out_idx], "w") as f:
            json.dump({"throughput": 1.0}, f)
        return _FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    module._run_benchmark("throughput", "benchmark_throughput.py", [])
    assert captured["env"]["VLLM_TARGET_DEVICE"] == "cpu"
    assert module.BENCHMARKS_DIR in captured["env"]["PYTHONPATH"].split(
        os.pathsep
    )


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


def test_run_baseline_subprocess_failure(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baseline", os.path.join(BENCHMARKS_DIR, "run_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_run_benchmark", _fail)
    out = tmp_path / "baseline.json"
    rc = module.main(["--output", str(out)])
    assert rc != 0
    assert not out.exists()


def test_run_baseline_aggregates_fields(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baseline", os.path.join(BENCHMARKS_DIR, "run_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values = {
        "throughput": 12.5,
        "latency": 0.25,
        "prefix_cache_hit_rate": 0.75,
    }

    def _fake(name, script, extra_args):
        return {name: values[name]}

    monkeypatch.setattr(module, "_run_benchmark", _fake)
    out = tmp_path / "baseline.json"
    rc = module.main(["--output", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data == values


def test_run_baseline_no_partial_file_on_failure(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baseline", os.path.join(BENCHMARKS_DIR, "run_baseline.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = {"n": 0}

    def _partial(name, script, extra_args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second failed")
        return {name: 1.0}

    monkeypatch.setattr(module, "_run_benchmark", _partial)
    out = tmp_path / "baseline.json"
    rc = module.main(["--output", str(out)])
    assert rc != 0
    assert not out.exists()
