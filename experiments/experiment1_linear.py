import csv
from pathlib import Path

import numpy as np

from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import (
    rolling_proposed_le_estimator,
)


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
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_file = (
        output_dir
        / "experiment1_linear_rolling.csv"
    )

    rows = []

    print()
    print("=" * 100)
    print("Experiment 1: Rolling Linear Fixed-Point Estimation")
    print("=" * 100)

    print(f"N  = {N}")
    print(f"L  = {L}")
    print(f"x0 = {X0}")
    print(f"K  = {K_VALUES}")
    print()

    header = (
        f"{'a':>7} "
        f"{'K':>5} "
        f"{'windows':>8} "
        f"{'True LE':>14} "
        f"{'First LE':>14} "
        f"{'Last LE':>14} "
        f"{'Last error':>14}"
    )

    print(header)
    print("-" * len(header))

    for a in A_VALUES:

        x = generate_linear_trajectory(
            a=a,
            L=L,
            x0=X0,
            N=N,
        )

        lambda_true = (
            linear_true_lyapunov(a)
        )

        for K in K_VALUES:

            results = (
                rolling_proposed_le_estimator(
                    x=x,
                    K=K,
                )
            )

            valid_results = [
                result
                for result in results
                if result["valid"]
            ]

            for result in results:

                if result["valid"]:
                    absolute_error = abs(
                        result["lambda_hat"]
                        - lambda_true
                    )
                else:
                    absolute_error = np.nan

                rows.append({
                    "a": a,
                    "K": K,
                    "N": N,
                    "L": L,
                    "x0": X0,

                    "end_index":
                        result["end_index"],

                    "window_start":
                        result["window_start"],

                    "window_end":
                        result["window_end"],

                    "boundary_index":
                        result["boundary_index"],

                    "L_hat_index":
                        result["L_hat_index"],

                    "L_hat":
                        result["L_hat"],

                    "lambda_true":
                        lambda_true,

                    "lambda_hat":
                        result["lambda_hat"],

                    "absolute_error":
                        absolute_error,

                    "ell_err_hat":
                        result["ell_err_hat"],

                    "id_fie_hat":
                        result["id_fie_hat"],

                    "runtime_sec":
                        result["runtime_sec"],

                    "valid":
                        result["valid"],

                    "failure_reason":
                        result["failure_reason"],
                })

            if len(valid_results) > 0:

                first = valid_results[0]
                last = valid_results[-1]

                last_error = abs(
                    last["lambda_hat"]
                    - lambda_true
                )

                print(
                    f"{a:7.2f} "
                    f"{K:5d} "
                    f"{len(valid_results):8d} "
                    f"{lambda_true:14.8f} "
                    f"{first['lambda_hat']:14.8f} "
                    f"{last['lambda_hat']:14.8f} "
                    f"{last_error:14.8e}"
                )

            else:

                print(
                    f"{a:7.2f} "
                    f"{K:5d} "
                    f"{0:8d} "
                    f"{lambda_true:14.8f} "
                    f"{'nan':>14} "
                    f"{'nan':>14} "
                    f"{'nan':>14}"
                )

    fieldnames = [
        "a",
        "K",
        "N",
        "L",
        "x0",
        "end_index",
        "window_start",
        "window_end",
        "boundary_index",
        "L_hat_index",
        "L_hat",
        "lambda_true",
        "lambda_hat",
        "absolute_error",
        "ell_err_hat",
        "id_fie_hat",
        "runtime_sec",
        "valid",
        "failure_reason",
    ]

    with open(
        detailed_file,
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
    print("=" * 100)
    print(
        f"Rolling results saved to: "
        f"{detailed_file}"
    )
    print(
        f"Total rolling estimates: "
        f"{len(rows)}"
    )
    print("=" * 100)
    print()


if __name__ == "__main__":
    run_experiment()