# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment defaults injected into every Python process.

This module is imported automatically by Python at startup (when it is
importable, e.g. because the repository root is on ``sys.path``). It is used
by CI to provide harmless defaults for environment variables that the vLLM
build backend requires, without mutating the caller's shell environment.

Only ``setdefault`` is used so that explicitly configured values always win.
"""

import os

# setup.py asserts CUDA_HOME is set when probing nvcc. Provide a harmless
# placeholder so CPU-only build environments do not fail during the build.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

# Avoid triggering CUDA detection / source builds in lightweight CI.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
