"""CLI contract only: no model training or LLM calls in the scaffold."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous ML research harness (scaffold)")
    parser.add_argument("task", nargs="?", default=Path("task.md"), type=Path)
    parser.parse_args()
    parser.exit(2, "Research workflow is not implemented yet. See docs/PLAN.md.\n")
