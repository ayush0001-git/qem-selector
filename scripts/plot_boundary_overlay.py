"""Driver script to generate the Angle 3 boundary overlay figure."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import yaml

# Set non-interactive backend to avoid _tkinter memory errors
matplotlib.use("Agg")

from qemsel import boundary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Angle 3 boundary overlay figure.")
    parser.add_argument("--model", type=Path, default=Path("results/boundary/model_significant.joblib"))
    parser.add_argument("--config", type=Path, default=Path("configs/boundary.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/boundary"))
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.model.exists():
        print(f"error: model not found: {args.model}")
        return 1
    if not args.config.exists():
        print(f"error: config not found: {args.config}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # Make sure 'shots_list' is in grid_spec (qemsel.boundary expects it if not named 'shots')
    if "shots_list" not in config and "shots" in config:
        config["shots_list"] = config["shots"] if isinstance(config["shots"], list) else [config["shots"]]

    print("Generating boundary overlay...")
    try:
        out_dict = boundary.overlay_selector_vs_theory(
            model_bundle=args.model,
            grid_spec=config,
            out_dir=args.out
        )
    except Exception as e:
        print(f"error generating overlay: {e}")
        return 1

    json_path = args.out / "boundary_overlay.json"
    json_path.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")

    print(f"Overlay complete! Image saved to {args.out / 'boundary_overlay.png'}")
    print(f"Data saved to {json_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
