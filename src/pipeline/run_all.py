"""CLI entrypoint for the full CPose pipeline."""
from __future__ import annotations

import argparse

from src import DATA_TEST_DIR, DEFAULT_CONFIG, print_module_console
from src.common.paths import DATA_DIR
from src.pipeline.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPose TFCS-PAR terminal pipeline")
    parser.add_argument("--input", default=str(DATA_TEST_DIR))
    parser.add_argument("--output", default=str(DATA_DIR / "runs"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--config", default=None, help="Runtime config path")
    parser.add_argument("--models", default=str(DEFAULT_CONFIG), help="Model registry path")
    parser.add_argument("--gt", default=None, help="Ground-truth annotation root; omit for proxy metrics only")
    parser.add_argument("--run-id", default="run", help="Tag appended to the timestamped run directory")
    args = parser.parse_args()
    print_module_console("pipeline", args)
    run_pipeline(args.input, args.output, args.manifest, args.topology, args.config, args.gt, args.run_id, args.models)


if __name__ == "__main__":
    main()
