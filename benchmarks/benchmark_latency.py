# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import sys


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Deprecated latency benchmark wrapper."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write structured JSON results to.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump({"latency": None, "deprecated": True}, f)
    print("""DEPRECATED: This script has been moved to the vLLM CLI.

Please use the following command instead:
    vllm bench latency

For help with the new command, run:
    vllm bench latency --help

Alternatively, you can run the new command directly with:
    python -m vllm.entrypoints.cli.main bench latency --help
""")
    sys.exit(1)
