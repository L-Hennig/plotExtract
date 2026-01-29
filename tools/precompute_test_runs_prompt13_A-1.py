"""Precompute 3 saved runs for the WebExtract "Test results" mode.

This script runs the v2 runner 3 times on:
  - plots/first_examples/A/A-1/A-1.png
  - prompt_13

It creates version folders like:
  A-1_png.pv2_prompt_13.v1/
  A-1_png.pv2_prompt_13.v2/
  A-1_png.pv2_prompt_13.v3/

Usage (from repo root):
  python tools/precompute_test_runs_prompt13_A-1.py

Notes:
- Requires your normal provider/API configuration for v2 runs.
- This is intentionally separate from the UI so "Test results" never triggers extraction.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
RUNNER = os.path.join(REPO_ROOT, "plot_extract_v2", "runner.py")

IMAGE_REL = os.path.join("first_examples", "A", "A-1", "A-1.png")
PROMPT = "prompt_13"
N_RUNS = 3


def main() -> int:
    image_abs = os.path.join(PLOTS_DIR, IMAGE_REL)
    if not os.path.isfile(image_abs):
        print(f"Image not found: {image_abs}")
        return 2

    if not os.path.isfile(RUNNER):
        print(f"Runner not found: {RUNNER}")
        return 2

    py = sys.executable
    print(f"Python: {py}")
    print(f"Runner: {RUNNER}")
    print(f"Image:  {image_abs}")
    print(f"Prompt: {PROMPT}")

    for i in range(N_RUNS):
        print(f"\n=== Precompute run {i + 1} / {N_RUNS} ===")
        # runner.py expects: <input_plot> <prompt_name> <article_info>
        args = [py, RUNNER, image_abs, PROMPT, ""]
        proc = subprocess.run(args, cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"Run {i + 1} failed with exit code {proc.returncode}")
            return int(proc.returncode) or 1

    print("\nDone. You can now enable 'Test results' in the WebExtract UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
