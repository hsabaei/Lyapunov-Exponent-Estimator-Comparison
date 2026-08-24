"""
Experiment 1: first complete baseline comparison on affine fixed-point maps.

All methods receive the same final K+2 observations

    x_{N-K}, ..., x_N, x_{N+1}

for each contraction factor a and each K.

Scientific target
-----------------
For

    U(x) = L + a (x - L),

the fixed-point Lyapunov exponent is

    lambda_true = log(|a|).

Estimators compared
-------------------
1. Proposed - Next limit estimate
2. Proposed - Aitken limit estimate
3. Yang-Wu scalar second-order local mapping
4. Sun-Wu RBF reproducible adaptation
5. Rosenstein fixed-point adaptation

Numerics
--------
The synthetic trajectory is generated with mpmath at MP_DPS digits.
The proposed estimator, Yang-Wu, and Rosenstein use their mpmath backends.
The Sun-Wu implementation preserves the high-precision relative geometry
before its internally normalized float64 RBF fit.

Runtime is recorded only as a diagnostic. Do not interpret the runtimes as
a fair computational benchmark because the arithmetic backends differ.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from mpmath import mp

from src.estimators.proposed import (
    proposed_le_estimator,
)

from src.estimators.yang_wu import (
    yang_wu_scalar_le_estimator,
)

from src.estimators.sun_wu import (
    sun_wu_scalar_rbf_le_estimator,
)

from src.estimators.rosenstein import (
    rosenstein_le_estimator,
)


MP_DPS = 200

L = mp.mpf("2.0")
X0 = mp.mpf("1.3578696273508724")

N = 200

A_VALUES = [
    "-0.95",
    "-0.8",
    "-0.6",
    "-0.4",
    "-0.2",
    "0.2",
    "0.4",
    "0.6",
    "0.8",
    "0.95",
]

K_VALUES = [
    10,
    20,
    40,
    50,
]


OUTPUT_DIR = Path(
    "results/tables"
)

DETAIL_PATH = (
    OUTPUT_DIR
    / "experiment1_baseline_comparison.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "experiment1_baseline_comparison_summary.csv"
)


def _as_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_error(
    estimate,
    truth,
    valid,
):
    if not valid:
        return math.nan

    try:
        return float(
            mp.fabs(
                mp.mpf(str(estimate))
                - truth
            )
        )
    except Exception:
        return math.nan


def affine_trajectory(a):
    a = mp.mpf(a)

    x = [X0]

    # Generate x_0, ..., x_N, x_{N+1}.
    for _ in range(N + 1):
        x.append(
            L
            + a * (x[-1] - L)
        )

    return x


def final_observation_record(
    x,
    K,
):
    # Same observations for every method:
    #
    # x_{N-K}, ..., x_N, x_{N+1}
    #
    # Length = K + 2.
    return x[
        N - K:
        N + 2
    ]


def _base_row(
    a,
    K,
    estimator,
    truth,
    result,
):
    estimate = result.get(
        "lambda_hat"
    )

    valid = bool(
        result.get(
            "valid",
            False,
        )
    )

    return {
        "experiment":
            "affine",

        "a":
            float(a),

        "K":
            K,

        "estimator":
            estimator,

        "lambda_true":
            float(truth),

        "lambda_hat":
            _as_float(
                estimate
            ),

        "abs_error":
            _safe_error(
                estimate=estimate,
                truth=truth,
                valid=valid,
            ),

        "valid":
            valid,

        "failure_reason":
            result.get(
                "failure_reason"
            ),

        "runtime_sec_diagnostic_only":
            _as_float(
                result.get(
                    "runtime_sec"
                )
            ),

        # Proposed-specific diagnostics.
        "L_hat":
            _as_float(
                result.get(
                    "L_hat"
                )
            ),

        "ell_err_hat":
            _as_float(
                result.get(
                    "ell_err_hat"
                )
            ),

        "id_fie_hat":
            _as_float(
                result.get(
                    "id_fie_hat"
                )
            ),

        # Yang-Wu diagnostics.
        "n_valid_local_maps":
            result.get(
                "n_valid_local_maps"
            ),

        "n_attempted_local_maps":
            result.get(
                "n_attempted_local_maps"
            ),

        # Sun-Wu diagnostics.
        "selected_hidden_nodes":
            result.get(
                "selected_hidden_nodes"
            ),

        "selected_width_neighbors":
            result.get(
                "selected_width_neighbors"
            ),

        "selected_cv_mse":
            _as_float(
                result.get(
                    "selected_cv_mse"
                )
            ),

        "training_mse":
            _as_float(
                result.get(
                    "training_mse"
                )
            ),

        # Rosenstein diagnostics.
        "n_neighbor_pairs":
            result.get(
                "n_neighbor_pairs"
            ),

        "n_horizons":
            (
                len(
                    result.get(
                        "horizons",
                        [],
                    )
                )
                if result.get(
                    "horizons"
                ) is not None
                else None
            ),
    }


def run_case(
    a,
    K,
    x,
):
    truth = mp.log(
        mp.fabs(a)
    )

    data = final_observation_record(
        x=x,
        K=K,
    )

    rows = []

    proposed_next = (
        proposed_le_estimator(
            x=data,
            K=K,
            end_index=K,
            limit_method="next",
            numeric_backend="mpmath",
        )
    )

    rows.append(
        _base_row(
            a=a,
            K=K,
            estimator="Proposed-Next",
            truth=truth,
            result=proposed_next,
        )
    )

    proposed_aitken = (
        proposed_le_estimator(
            x=data,
            K=K,
            end_index=K,
            limit_method="aitken",
            numeric_backend="mpmath",
        )
    )

    rows.append(
        _base_row(
            a=a,
            K=K,
            estimator="Proposed-Aitken",
            truth=truth,
            result=proposed_aitken,
        )
    )

    yang_wu = (
        yang_wu_scalar_le_estimator(
            x=data,
            evolution=1,
            r_min=None,
            numeric_backend="mpmath",
        )
    )

    rows.append(
        _base_row(
            a=a,
            K=K,
            estimator="Yang-Wu",
            truth=truth,
            result=yang_wu,
        )
    )

    sun_wu = (
        sun_wu_scalar_rbf_le_estimator(
            x=data,
            max_hidden_nodes=20,
            width_neighbors=None,
        )
    )

    rows.append(
        _base_row(
            a=a,
            K=K,
            estimator="Sun-Wu",
            truth=truth,
            result=sun_wu,
        )
    )

    rosenstein = (
        rosenstein_le_estimator(
            x=data,
            embedding_dim=1,
            lag=1,
            temporal_exclusion=0,
            sample_time=mp.mpf("1.0"),
            max_horizon=None,
            numeric_backend="mpmath",
        )
    )

    rows.append(
        _base_row(
            a=a,
            K=K,
            estimator="Rosenstein",
            truth=truth,
            result=rosenstein,
        )
    )

    return rows


def summarize(rows):
    estimators = sorted(
        {
            row["estimator"]
            for row in rows
        }
    )

    summary_rows = []

    for estimator in estimators:
        subset = [
            row
            for row in rows
            if row["estimator"]
            == estimator
        ]

        valid_rows = [
            row
            for row in subset
            if row["valid"]
            and math.isfinite(
                row["abs_error"]
            )
        ]

        errors = [
            row["abs_error"]
            for row in valid_rows
        ]

        sign_correct = 0

        for row in valid_rows:
            estimate = row[
                "lambda_hat"
            ]

            truth = row[
                "lambda_true"
            ]

            if (
                math.isfinite(
                    estimate
                )
                and estimate * truth > 0
            ):
                sign_correct += 1

        if errors:
            mean_error = (
                statistics.fmean(
                    errors
                )
            )

            median_error = (
                statistics.median(
                    errors
                )
            )

            max_error = max(
                errors
            )
        else:
            mean_error = math.nan
            median_error = math.nan
            max_error = math.nan

        summary_rows.append(
            {
                "estimator":
                    estimator,

                "n_cases":
                    len(subset),

                "n_valid":
                    len(valid_rows),

                "valid_rate":
                    (
                        len(valid_rows)
                        / len(subset)
                    ),

                "mean_abs_error":
                    mean_error,

                "median_abs_error":
                    median_error,

                "max_abs_error":
                    max_error,

                "sign_correct_rate":
                    (
                        sign_correct
                        / len(valid_rows)
                        if valid_rows
                        else math.nan
                    ),
            }
        )

    return summary_rows


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    mp.dps = MP_DPS

    rows = []

    total_cases = (
        len(A_VALUES)
        * len(K_VALUES)
    )

    completed = 0

    for a_text in A_VALUES:
        a = mp.mpf(
            a_text
        )

        x = affine_trajectory(
            a
        )

        for K in K_VALUES:
            rows.extend(
                run_case(
                    a=a,
                    K=K,
                    x=x,
                )
            )

            completed += 1

            print(
                f"completed "
                f"{completed}/{total_cases} "
                f"system-window cases "
                f"(a={float(a):+.2f}, "
                f"K={K})"
            )

    summary_rows = summarize(
        rows
    )

    write_csv(
        DETAIL_PATH,
        rows,
    )

    write_csv(
        SUMMARY_PATH,
        summary_rows,
    )

    print(
        "\n=== EXPERIMENT 1 "
        "BASELINE COMPARISON SUMMARY ==="
    )

    for row in summary_rows:
        print(
            f"{row['estimator']:<18} "
            f"valid="
            f"{row['n_valid']}/"
            f"{row['n_cases']} "
            f"mean_abs_error="
            f"{row['mean_abs_error']:.6e} "
            f"median_abs_error="
            f"{row['median_abs_error']:.6e} "
            f"max_abs_error="
            f"{row['max_abs_error']:.6e} "
            f"sign_correct="
            f"{row['sign_correct_rate']:.3f}"
        )

    print(
        "\nSaved:"
    )

    print(
        DETAIL_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nNote: runtime_sec_diagnostic_only "
        "must not be used as a fair runtime "
        "comparison because the estimators "
        "currently use different arithmetic "
        "backends."
    )


if __name__ == "__main__":
    main()
