# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight site hook for benchmark subprocesses.

When the benchmarks directory is on PYTHONPATH, Python imports this module
at startup. It forces the CPU target so that importing vllm does not require
a CUDA toolchain, which keeps the baseline runner usable in CPU-only CI.
"""

import os

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
# setup.py's get_nvcc_cuda_version asserts CUDA_HOME is set; provide a
# placeholder so CPU-only benchmark subprocesses never trip that assertion.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
