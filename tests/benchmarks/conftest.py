# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test configuration for benchmark script tests.

These tests exercise standalone benchmark scripts and must not require a
built vLLM extension or a CUDA toolchain. We therefore avoid importing the
vllm package at collection time and force the CPU target for any subprocess
that may import vllm.
"""

import os
import sys

BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
)

if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

# Prevent subprocesses from attempting CUDA detection during import.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
