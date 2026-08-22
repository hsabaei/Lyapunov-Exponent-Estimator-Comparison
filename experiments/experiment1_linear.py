import csv
import time
from pathlib import Path

import numpy as np

from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import proposed_le_estimator


# ============================================================
# Frozen Experiment 1 parameters
# ============================================================

N = 60
L = 0.0
X0 = 1.0

A_VALUES = [
    -0.95,
    -0.80,
    -0.60,
    -0.40,
    -0.20,
     0.20,
     0.40,
     0.60,
     0.80,
     0.95,
]

K_VALUES = [10, 20, 40]


def run_experiment():

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "experiment1_linear_proposed.csv"

    rows = []

    print()
    print("=" * 95)
    print("Experiment 1: Linear Fixed-Point Contraction")
    print("=" * 95)
    print(f"N  = {N}")
    print(f"L  = {L}")
    print(f"x0 = {X0}")
    print(f"K  = {K_VALUES}")
    print()

    header = (
        f"{'a':>7} "
        f"{'K':>5} "
        f"{'True LE':>14} "
        f"{'Estimated LE':>14} "
        f"{'Abs. Error':>14} "
        f"{'Time (s)':>12} "
        f"{'Valid':>7}"
    )

    print(header)
    print("-" * len(header))

    for a in A_VALUES:

        # Generate one common trajectory for this value of a.
        # The same trajectory is then evaluated at every K.
        x = generate_linear_trajectory(
            a=a,
            L=L,
            x0=X0,
            N=N,
        )

        lambda_true = linear_true_lyapunov(a)

        for K in K_VALUES:

            result = proposed_le_estimator(
                x=x,
                K=K,
            )

            if result["valid"]:
                lambda_hat = result["lambda_hat"]

                absolute_error = abs(
                    lambda_hat - lambda_true
                )

            else:
                lambda_hat = np.nan
                absolute_error = np.nan

            row = {
                "a": a,
                "K": K,
                "N": N,
                "L": L,
                "x0": X0,
                "lambda_true": lambda_true,
                "lambda_hat": lambda_hat,
                "absolute_error": absolute_error,
                "runtime_sec": result["runtime_sec"],
                "ell_err_hat": result["ell_err_hat"],
                "id_fie_hat": result["id_fie_hat"],
                "L_hat": result["L_hat"],
                "t0": result["t0"],
                "valid": result["valid"],
                "failure_reason": result["failure_reason"],
            }

            rows.append(row)

            print(
                f"{a:7.2f} "
                f"{K:5d} "
                f"{lambda_true:14.8f} "
                f"{lambda_hat:14.8f} "
                f"{absolute_error:14.8e} "
                f"{result['runtime_sec']:12.4e} "
                f"{str(result['valid']):>7}"
            )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    fieldnames = [
        "a",
        "K",
        "N",
        "L",
        "x0",
        "lambda_true",
        "lambda_hat",
        "absolute_error",
        "runtime_sec",
        "ell_err_hat",
        "id_fie_hat",
        "L_hat",
        "t0",
        "valid",
        "failure_reason",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 95)
    print(f"Results saved to: {output_file}")
    print(f"Number of cases:   {len(rows)}")
    print("=" * 95)
    print()


if __name__ == "__main__":
    run_experiment()