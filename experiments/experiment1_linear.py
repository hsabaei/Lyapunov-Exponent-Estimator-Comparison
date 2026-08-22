import csv
from pathlib import Path

import numpy as np
import mpmath as mp

from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import (
    rolling_proposed_le_estimator,
)


# ============================================================
# Experiment 1 parameters
# ============================================================

N = 200

# Nonzero fixed point
L = 2.0


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


K_VALUES = [
    10,
    20,
    40,
    50,
]


LIMIT_METHODS = [
    "next",
    "aitken",
]


# ============================================================
# High precision
# ============================================================

# We need enough precision to retain the smallest residual
# occurring near iteration N.
#
# For the present experiment:
#
#     min |a| = 0.2
#     N = 200
#
# so residuals can be around 10^(-140).
#
# 200 decimal digits comfortably resolves them.

MP_DPS = 200

mp.mp.dps = MP_DPS


# ============================================================
# Random initial condition
# ============================================================

# Reproducible random draw.
#
# x0 is sampled once and then held fixed for every a,
# so only the contraction factor changes between systems.

RANDOM_SEED = 2026

rng = np.random.default_rng(
    RANDOM_SEED
)

X0 = float(
    rng.uniform(
        L - 1.0,
        L + 1.0,
    )
)

# Probability of exact equality is essentially zero,
# but enforce x0 != L logically.

while X0 == L:

    X0 = float(
        rng.uniform(
            L - 1.0,
            L + 1.0,
        )
    )


# ============================================================
# Formatting helpers
# ============================================================

def to_float(value):

    if not mp.isfinite(value):
        return np.nan

    return float(value)


def mp_string(
    value,
    digits=50,
):

    if not mp.isfinite(value):
        return "nan"

    return mp.nstr(
        value,
        digits,
    )


# ============================================================
# Experiment
# ============================================================

def run_experiment():

    output_dir = Path(
        "results/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rolling_file = (
        output_dir
        / "experiment1_linear_rolling.csv"
    )

    summary_file = (
        output_dir
        / "experiment1_linear_summary.csv"
    )

    rolling_rows = []
    summary_rows = []

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 154)

    print(
        "Experiment 1: High-Precision "
        "Linear Fixed-Point Benchmark"
    )

    print("=" * 154)

    print(f"N          = {N}")
    print(f"L          = {L}")
    print(f"x0         = {X0:.16f}")
    print(f"random seed= {RANDOM_SEED}")
    print(f"K          = {K_VALUES}")
    print(f"precision  = {MP_DPS} decimal digits")
    print(
        f"methods    = {LIMIT_METHODS}"
    )

    print()

    header = (
        f"{'a':>7} "
        f"{'K':>5} "
        f"{'method':>8} "
        f"{'n':>5} "
        f"{'windows':>8} "
        f"{'True LE':>14} "
        f"{'Estimated LE':>14} "
        f"{'ell_err':>14} "
        f"{'FIE hat':>12} "
        f"{'|Lhat-L|':>14} "
        f"{'Abs. Error':>14}"
    )

    print(header)
    print("-" * len(header))

    # ========================================================
    # Systems
    # ========================================================

    for a in A_VALUES:

        x = generate_linear_trajectory(
            a=a,
            L=L,
            x0=X0,
            N=N,
            high_precision=True,
        )

        lambda_true = (
            linear_true_lyapunov(
                a=a,
                high_precision=True,
            )
        )

        L_mp = mp.mpf(
            str(L)
        )

        # ====================================================
        # K
        # ====================================================

        for K in K_VALUES:

            # ================================================
            # Limit methods
            # ================================================

            for limit_method in LIMIT_METHODS:

                results = (
                    rolling_proposed_le_estimator(
                        x=x,
                        K=K,
                        limit_method=limit_method,
                        numeric_backend="mpmath",
                    )
                )

                valid_results = [
                    r
                    for r in results
                    if r["valid"]
                ]

                # ============================================
                # Save all rolling results
                # ============================================

                for result in results:

                    if result["valid"]:

                        absolute_error = mp.fabs(
                            result["lambda_hat"]
                            - lambda_true
                        )

                        L_hat_error = mp.fabs(
                            result["L_hat"]
                            - L_mp
                        )

                    else:

                        absolute_error = mp.nan
                        L_hat_error = mp.nan

                    rolling_rows.append({

                        "a":
                            a,

                        "K":
                            K,

                        "N":
                            N,

                        "L":
                            L,

                        "x0":
                            X0,

                        "random_seed":
                            RANDOM_SEED,

                        "precision_dps":
                            MP_DPS,

                        "limit_method":
                            limit_method,

                        "end_index":
                            result["end_index"],

                        "window_start":
                            result["window_start"],

                        "window_end":
                            result["window_end"],

                        "latest_observation_index":
                            result[
                                "latest_observation_index"
                            ],

                        "L_hat":
                            mp_string(
                                result["L_hat"]
                            ),

                        "L_hat_error":
                            mp_string(
                                L_hat_error
                            ),

                        "lambda_true":
                            mp_string(
                                lambda_true
                            ),

                        "lambda_hat":
                            mp_string(
                                result[
                                    "lambda_hat"
                                ]
                            ),

                        "absolute_error":
                            mp_string(
                                absolute_error
                            ),

                        "ell_err_hat":
                            mp_string(
                                result[
                                    "ell_err_hat"
                                ]
                            ),

                        "id_fie_hat":
                            mp_string(
                                result[
                                    "id_fie_hat"
                                ]
                            ),

                        "runtime_sec":
                            result[
                                "runtime_sec"
                            ],

                        "valid":
                            result["valid"],

                        "failure_reason":
                            result[
                                "failure_reason"
                            ],
                    })

                # ============================================
                # Representative result:
                # FINAL rolling window
                # ============================================

                if len(valid_results) == 0:

                    print(
                        f"{a:7.2f} "
                        f"{K:5d} "
                        f"{limit_method:>8} "
                        f"{'--':>5} "
                        f"{0:8d} "
                        f"{float(lambda_true):14.8f} "
                        f"{'nan':>14} "
                        f"{'nan':>14} "
                        f"{'nan':>12} "
                        f"{'nan':>14} "
                        f"{'nan':>14}"
                    )

                    continue

                final_result = (
                    valid_results[-1]
                )

                L_hat_error = mp.fabs(
                    final_result["L_hat"]
                    - L_mp
                )

                absolute_error = mp.fabs(
                    final_result["lambda_hat"]
                    - lambda_true
                )

                print(
                    f"{a:7.2f} "
                    f"{K:5d} "
                    f"{limit_method:>8} "
                    f"{final_result['end_index']:5d} "
                    f"{len(valid_results):8d} "
                    f"{float(lambda_true):14.8f} "
                    f"{float(final_result['lambda_hat']):14.8f} "
                    f"{float(final_result['ell_err_hat']):14.8f} "
                    f"{float(final_result['id_fie_hat']):12.8f} "
                    f"{float(L_hat_error):14.4e} "
                    f"{float(absolute_error):14.8e}"
                )

                # ============================================
                # Summary CSV
                # ============================================

                summary_rows.append({

                    "a":
                        a,

                    "K":
                        K,

                    "N":
                        N,

                    "L":
                        L,

                    "x0":
                        X0,

                    "random_seed":
                        RANDOM_SEED,

                    "precision_dps":
                        MP_DPS,

                    "limit_method":
                        limit_method,

                    "end_index":
                        final_result[
                            "end_index"
                        ],

                    "valid_windows":
                        len(valid_results),

                    "total_windows":
                        len(results),

                    "lambda_true":
                        mp_string(
                            lambda_true
                        ),

                    "lambda_hat":
                        mp_string(
                            final_result[
                                "lambda_hat"
                            ]
                        ),

                    "absolute_error":
                        mp_string(
                            absolute_error
                        ),

                    "ell_err_hat":
                        mp_string(
                            final_result[
                                "ell_err_hat"
                            ]
                        ),

                    "id_fie_hat":
                        mp_string(
                            final_result[
                                "id_fie_hat"
                            ]
                        ),

                    "L_hat":
                        mp_string(
                            final_result[
                                "L_hat"
                            ]
                        ),

                    "L_hat_error":
                        mp_string(
                            L_hat_error
                        ),

                    "runtime_sec":
                        final_result[
                            "runtime_sec"
                        ],

                    "valid":
                        final_result[
                            "valid"
                        ],
                })

    # ========================================================
    # Save rolling results
    # ========================================================

    if rolling_rows:

        with open(
            rolling_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rolling_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                rolling_rows
            )

    # ========================================================
    # Save summary
    # ========================================================

    if summary_rows:

        with open(
            summary_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 154)

    print(
        f"Summary saved to: "
        f"{summary_file}"
    )

    print(
        f"Rolling results saved to: "
        f"{rolling_file}"
    )

    print("=" * 154)
    print()


if __name__ == "__main__":
    run_experiment()