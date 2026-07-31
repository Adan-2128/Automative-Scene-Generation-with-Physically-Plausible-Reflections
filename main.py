#!/usr/bin/env python3
"""
CLI entry point for the Automotive Scene Generation Pipeline (PS1).

Usage:
    # First run (or whenever environment images change): fit PCA + run
    python main.py --car-model-id GT-R35 --geometry-hash abc123 --fit-pca

    # Subsequent runs: reuse the saved PCA state
    python main.py --car-model-id GT-R35 --geometry-hash abc123
"""
from __future__ import annotations

import argparse

from environments.car_constraints import CarLockProfile
from pipeline import ScenePipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automotive Scene Generation Pipeline (PS1)")
    p.add_argument("--car-model-id", required=True, help="Identifier for the locked 3D car asset")
    p.add_argument("--geometry-hash", required=True, help="Hash/version pin of the locked car geometry")
    p.add_argument("--paint-code", default="unspecified")
    p.add_argument("--wheel-spec", default="unspecified")
    p.add_argument("--fit-pca", action="store_true", help="Force refit PCA on current environment images")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    car_lock = CarLockProfile(
        model_id=args.car_model_id,
        geometry_hash=args.geometry_hash,
        paint_code=args.paint_code,
        wheel_spec=args.wheel_spec,
    )

    pipeline = ScenePipeline(car_lock=car_lock)
    results = pipeline.run(fit_pca=args.fit_pca)

    print(
        f"\nProcessed {len(results)} environment scene(s). "
        f"JSON descriptors written to outputs/scenes/."
    )


if __name__ == "__main__":
    main()