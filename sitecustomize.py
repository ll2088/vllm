# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repository-level sitecustomize for lightweight CI environments.

When the repository root is on ``sys.path`` (e.g. ``pip install -e .`` with
build isolation, or running scripts from the repo), Python imports this
module at startup. It provides harmless defaults so that ``setup.py`` does
not fail when probing the CUDA toolchain in CPU-only environments.

Explicitly set environment variables always take precedence.
"""

import os

# setup.py asserts CUDA_HOME is set when probing nvcc; provide a harmless
# placeholder so CPU-only environments do not fail during build/import.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

# Avoid triggering CUDA detection / source builds in lightweight CI.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
