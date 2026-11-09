# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repository-level site hook for CPU-only build/test environments.

When the repository root is on ``PYTHONPATH`` (as it is for ``pip install -e .``
and for pytest run from the repo root), Python imports this module at startup,
before any build backend or test module runs. It provides harmless defaults so
that ``setup.py``'s ``get_nvcc_cuda_version`` does not assert when no CUDA
toolkit is installed, while leaving real CUDA setups untouched.
"""

import os

# setup.py asserts CUDA_HOME is set when probing nvcc; provide a placeholder
# so CPU-only environments do not fail during build/import paths.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
# Keep the default target CPU-friendly unless the caller explicitly opts in.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
