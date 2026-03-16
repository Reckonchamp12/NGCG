#!/usr/bin/env python3
"""
experiments/run_benchmark.py
============================
Run NGCG against all baselines on the full 9-system benchmark.

Usage
-----
    python experiments/run_benchmark.py --data ngcg_data_clean.h5 --seed 0
    python experiments/run_benchmark.py --systems mass_spring henon_heiles --seed 0
    python experiments/run_benchmark.py --seeds 0 1 2   # multi-seed run
"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ngcg import NGCG
from ngcg.data import ALL_SYSTEMS, TRUE_LAWS

OUT_DIR = "results"
os.makedirs(f"{OUT_DIR}/seed0", exist_ok=True)


def run_one(system: str, data_path: str, seed: int) -> dict:
    print(f"\n{'═'*60}\n  {system.upper()}  seed={seed}\n{'═'*60}")
    try:
        model  = NGCG(system=system, data_path=data_path)
        result = model.fit(seed=seed)
        result["seed"] = seed
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "method": "NGCG", "system": system, "seed": seed,
            "error": str(e)[:120],
            "DR": 0.0, "FDR": 0.0, "F1": 0.0,
            "MSE_16": 999.0, "CV": 999.0,
            "best_constancy": 999.0, "complexity": 999.0,
            "has_true_law": 1.0 if TRUE_LAWS.get(system) else 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="ngcg_data_clean.h5")
    parser.add_argument("--systems", nargs="*", default=ALL_SYSTEMS)
    parser.add_argument("--seed",    type=int,  default=0)
    parser.add_argument("--seeds",   nargs="*", type=int, default=None)
    args = parser.parse_args()

    seeds   = args.seeds or [args.seed]
    all_rows = []

    for seed in seeds:
        for system in args.systems:
            row = run_one(system, args.data, seed)
            all_rows.append(row)

            # Save after every system
            df = pd.DataFrame(all_rows)
            df.to_csv(f"{OUT_DIR}/results_all.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUT_DIR}/results_all.csv", index=False)

    # Summary
    print(f"\n\n{'═'*60}")
    print("  NGCG BENCHMARK — FINAL SUMMARY")
    print(f"{'═'*60}\n")
    cols = ["system", "DR", "DR_strict", "FDR", "F1",
            "MSE_16", "CV", "best_constancy", "true_law_constancy",
            "complexity", "fit_time_s"]
    cols = [c for c in cols if c in df.columns]
    print(df[df.seed == seeds[0]][cols].round(4).to_string(index=False))

    if len(seeds) > 1:
        print(f"\n  Multi-seed summary (mean ± std):")
        for system in args.systems:
            sub = df[df.system == system]
            print(f"  {system:<18}  "
                  f"DR={sub.DR.mean():.2f}±{sub.DR.std():.2f}  "
                  f"F1={sub.F1.mean():.2f}±{sub.F1.std():.2f}")

    print(f"\n  Results → {OUT_DIR}/results_all.csv")


if __name__ == "__main__":
    main()
